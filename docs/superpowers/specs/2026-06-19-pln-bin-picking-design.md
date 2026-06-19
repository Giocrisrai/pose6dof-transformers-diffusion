# Diseño — Bin picking guiado por lenguaje natural (PLN open-source)

> Fecha: 2026-06-19 · Estado: aprobado para implementación
> Autor: TFM Pose 6-DoF · Enfoque elegido: **A** (Parser + Grounder, default determinista, LLM en Fase 2)

## 1. Motivación

El proyecto ya tiene un cuerpo de trabajo de lenguaje validado como exploraciones
post-TFM (exp16–26): VLA-lite con CLIP text-conditioning (98.6%), robustez
lingüística (100% en familias no vistas), multi-atributo color/forma/tamaño
(~99.9%), multi-objeto N=2..5 (100%), instrucciones secuenciales, CLIP-image
grounding y razonamiento espacial. Pero **vive como experimentos y docs sueltas**:

- `src/pipeline.py` (`BinPickingPipeline.run`) **no acepta instrucción de texto**.
- La CLI de demo (`run_pick_*`, `run_e2e_live.py`) no tiene flag de instrucción.
- El dashboard muestra exp16–26 como tarjetas de solo-lectura.
- No hay una *feature* open-source coherente, documentada y comparada con el
  estado del arte (CLIPort, VoxPoser, SayCan, OWL-ViT/CLIP-Fields).

**Objetivo**: convertir ese trabajo en una capacidad de producto —"pick guiado
por lenguaje natural"— integrada, testeada y documentada, competitiva frente a
soluciones de bin picking, manteniendo el ethos **100% open-license** y la
ejecución **local en M1 Pro 16 GB**.

## 2. Alcance por fases

- **Fase 1 — Consolidar e integrar (default determinista).** Núcleo `src/language/`,
  integración en pipeline, CLI de demo, tab de dashboard, docs + comparativa SOTA.
  Reusa el grounding ya validado (CLIP-image + filtros de atributo/espacial).
- **Fase 2 — Lenguaje libre (LLM local).** Backend `InstructionParser` con LLM
  open-source local (Ollama/MLX) para frases fuera de gramática, con fallback
  determinista. Backend API enchufable como punto de extensión (no por defecto).

El salto de capacidad de la Fase 2 es en **comprensión** (parser), no en
**grounding** (que se mantiene a 98–100% y no se toca).

## 3. Arquitectura y modelo de datos

Paquete nuevo `src/language/`:

```
src/language/
  __init__.py
  schema.py        # Instruction, TargetSpec, SpatialRelation, GroundingResult
  parser.py        # InstructionParser (interfaz) + make_parser factory
  grounding.py     # Grounder: Instruction + objetos detectados -> target(s)
  vocab.py         # léxico controlado ES/EN (colores, formas, tamaños, relaciones)
  backends/
    __init__.py
    deterministic.py   # gramática/regex (Fase 1, default)
    llm_local.py       # Ollama/MLX (Fase 2)
    llm_api.py         # stub configurable (Fase 2)
```

### Modelo de datos (`schema.py`)

```python
@dataclass
class TargetSpec:
    color: str | None        # normalizado desde "red"/"rojo"
    shape: str | None        # cube/sphere/cylinder/box
    size: str | None         # small/large (atributo continuo, exp22)
    raw_noun: str | None     # "pieza", "objeto"

@dataclass
class SpatialRelation:        # exp26
    relation: str            # left_of/right_of/nearest/farthest/on_top
    anchor: TargetSpec | None

@dataclass
class Instruction:
    raw_text: str
    intent: str              # "pick" | "pick_then_place" | "sequence"
    target: TargetSpec
    spatial: SpatialRelation | None
    steps: list["Instruction"]   # instrucciones secuenciales (exp23)
    confidence: float
    backend: str             # qué parser lo produjo

@dataclass
class GroundingResult:
    target_obj_id: int | None      # id en la List[PoseResult] del pipeline
    scores: dict[int, float]       # similitud por objeto (CLIP + filtros)
    method: str                    # "clip_image" | "attribute" | "spatial"
    rejected: list[int]
    ambiguous: bool                # >1 candidato con score similar
```

### Contratos de interfaz

- `InstructionParser.parse(text: str) -> Instruction` — puro texto→estructura, sin
  estado de escena. Sustituible por backend.
- `Grounder.ground(instruction, objects, rgb, K) -> GroundingResult` — `objects`
  es la `List[PoseResult]` que ya produce el pipeline (con `bbox`/`mask` para
  recortar crops). CLIP image-text para color/forma + reglas geométricas para
  relación espacial/tamaño.

Cada unidad es testeable por separado: parser con strings; grounder con escenas
sintéticas con seed (generadores de exp16/24).

## 4. Integración

### Pipeline (`src/pipeline.py`)

- `PipelineConfig` gana: `language_enabled: bool = False`,
  `parser_backend: str = "deterministic"`, `grounder_method: str = "clip_image"`.
- `run(rgb, depth, K, masks=None, instruction: str | None = None)`:
  1. detección + `estimate_poses` (igual que hoy)
  2. si `instruction`: `parse` → `ground` → filtra `poses` al target (o lista
     ordenada por score si `ambiguous`)
  3. `plan_grasps` solo sobre el/los target(s)
  4. `PipelineResult` gana campos opcionales `instruction` y `grounding`
     (no rompe consumidores existentes).
- Carga perezosa: parser/grounder solo se inicializan si `language_enabled`; CLIP
  se importa lazy (como `cv2`).

### CLI de demo — `experiments/run_pick_language.py` (nuevo)

- `--instruction "dame el cubo rojo"` `--parser-backend deterministic|llm`
  `--scene clutter|multi` `--render`.
- Reusa `src/simulation/multi_object_scene.py`, ejecuta E2E en CoppeliaSim y
  guarda `experiments/results/language_pick/<run>.json` + render con el target
  resaltado. No toca los `run_pick_*` existentes.

### Dashboard (`dashboard.py`) — pestaña "🗣️ Lenguaje natural (PLN)"

- Input de texto + selector de escena de ejemplo.
- Muestra `Instruction` parseada (JSON), scores de grounding por objeto, imagen
  con target resaltado y trayectoria resultante.
- Modo en vivo (lee `run_pick_language.py`) + modo demo con escenas pre-renderizadas
  cacheadas (funciona sin CoppeliaSim, como el resto del dashboard).
- Las tarjetas exp16–26 se reorganizan bajo esta pestaña como "validación
  experimental" de la feature.

## 5. Fase 2 — LLM local y API enchufable

- `backends/llm_local.py`: `InstructionParser` con LLM open-source local.
  Default **Ollama** (`qwen2.5:3b-instruct` o `llama3.2:3b`) vía
  `http://localhost:11434`; alternativa **MLX** nativa M1. Estrategia
  *constrained JSON output* (prompt few-shot pidiendo el esquema `Instruction`);
  valida contra schema; ante JSON inválido o fallo de conexión → **fallback
  automático a `DeterministicParser`**. Registra `confidence` y
  `backend="llm_local:<model>"`.
- `backends/llm_api.py`: misma interfaz, stub configurable por entorno
  (`LANGUAGE_API_PROVIDER` / `*_API_KEY`); si no hay clave, no se registra como
  disponible. Sin claves commiteadas. Punto de extensión documentado.
- Factory `make_parser(backend="deterministic", **kw) -> InstructionParser`:
  `"deterministic"` (default) | `"llm_local"` | `"llm_api"`; cada uno cae al
  determinista si su dependencia no está disponible.
- Dependencias `ollama`/`mlx-lm`/proveedor API como **extras opcionales** en
  `pyproject.toml` (`[project.optional-dependencies] language-llm = [...]`); el
  core y la Fase 1 no requieren ningún LLM.

## 6. Documentación y comparativa SOTA

- `docs/LENGUAJE_NATURAL.md`: documento unificado de la feature (motivación,
  arquitectura, API, backends, ejemplos ES/EN, instalación del extra, tabla
  consolidada de resultados exp16–26).
- Comparativa SOTA (sección o `docs/COMPARATIVA_SOTA_LENGUAJE.md`): tabla frente
  a CLIPort, VoxPoser, OWL-ViT/CLIP-Fields, SayCan en ejes: licencia/open-source,
  hardware (M1 local vs. clúster), re-entrenamiento por vocabulario,
  interpretabilidad, accuracy. **Cifras de terceros citadas de sus papers, con
  referencia; nada inventado.**
- `README.md`: subsección "Natural-language bin picking" con quickstart de 3
  líneas + figura del grounding.
- `docs/exploraciones/`: nota de que exp16–26 quedan consolidados en la feature.
- Memoria del proyecto: nueva entrada enlazada con `[[tfm-docs-ubicacion]]`.

## 7. Testing (pytest)

- `test_language_parser.py`: `DeterministicParser` sobre corpus ES/EN (colores,
  formas, tamaños, relaciones, secuencias) + casos límite (frase vacía, atributo
  desconocido, multi-atributo).
- `test_grounding.py`: `Grounder` sobre escenas sintéticas con seed (target
  correcto, detección de ambigüedad, relación espacial).
- `test_pipeline_language.py`: `run(instruction=...)` con poses mock — filtra al
  target; sin `instruction` el comportamiento es idéntico al actual (no-regresión).
- `test_llm_fallback.py`: con Ollama ausente, `make_parser("llm_local")` cae al
  determinista sin error (ausencia mockeada).
- CLIP y LLM marcados `@pytest.mark.slow`/`skipif`: la suite base corre sin pesos
  ni red.

## 8. Criterios de éxito

- **Fase 1**: `run(instruction=...)` selecciona el target correcto en ≥95% de un
  set de escenas de test; suite base verde sin deps de LLM/red; CLI + tab de
  dashboard operativos con escenas cacheadas.
- **Fase 2**: `llm_local` parsea ≥90% de un corpus de frases libres fuera de
  gramática a `Instruction` válida, con fallback determinista verificado.

## 9. No-objetivos (YAGNI)

- No re-entrenar la diffusion policy para text-conditioning (exp16 queda como ruta
  opcional, no productiva).
- No commitear claves ni hacer del backend API el camino por defecto.
- No refactor no relacionado del pipeline ni del dashboard.
