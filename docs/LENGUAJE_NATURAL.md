# Bin picking guiado por lenguaje natural (PLN)

Esta página documenta la *feature* de **lenguaje natural** del pipeline: la
capacidad de seleccionar el objeto a coger a partir de una instrucción escrita
("dame el cubo rojo de la izquierda") en lugar de un índice o una pose fija.

El subsistema vive en el paquete [`src/language/`](../src/language/) y se conecta
al pipeline end-to-end vía [`src/pipeline.py`](../src/pipeline.py).

---

## 1. Motivación

El bin picking competitivo (logística, reciclaje, manufactura) rara vez consiste
en "coger cualquier objeto": el operario o el MES indica *qué* coger ("la pieza
azul", "el paquete más a la izquierda", "primero los tornillos, luego las
tuercas"). Dar al pipeline una interfaz en lenguaje natural:

- **Acerca el sistema al paradigma VLA** (Vision-Language-Action) sin pagar su
  coste: los grandes VLA (RDT-1B, π0) requieren del orden de un millón de
  episodios reales. Aquí el *grounding* es determinista o se apoya en CLIP
  congelado, entrenable en un portátil M1 Pro.
- **Mantiene interpretabilidad**: el resultado del *grounding* es una tabla de
  *scores* por objeto, no una caja negra.
- **Es enchufable**: el parser por defecto no necesita red ni pesos; un backend
  LLM local (Ollama) o remoto se puede activar cuando se quiera más cobertura
  lingüística.

Esta feature **consolida** las exploraciones post-TFM **exp16–exp26** (CLIP
text-conditioning, robustez lingüística, multi-atributo, multi-objeto, tamaño,
secuencias, CLIP-image y razonamiento espacial). Los resultados cuantitativos de
cada una se resumen en la [sección 6](#6-resultados-consolidados-exp16exp26),
con cita al documento de exploración de origen.

---

## 2. Arquitectura

El flujo de selección por lenguaje es **Parser → Grounder → Pipeline**:

```
  texto: "dame el cubo rojo de la izquierda"
                 │
                 ▼
     ┌───────────────────────┐
     │  Parser                │   src/language/parser.py + backends/
     │  texto → Instruction   │   (deterministic | llm_local | llm_api)
     └───────────┬───────────┘
                 │  Instruction(target=TargetSpec(color, shape, size),
                 │              spatial=SpatialRelation, intent, steps)
                 ▼
     ┌───────────────────────┐      ObjectView[]  (obj_id, centroid 3D,
     │  Grounder              │ ◀───  attributes, bbox)  — construidos por
     │  Instruction + escena  │       el pipeline desde sus PoseResult
     │       → target         │
     └───────────┬───────────┘   src/language/grounding.py
                 │  GroundingResult(target_obj_id, scores, method,
                 │                  rejected, ambiguous)
                 ▼
     ┌───────────────────────┐
     │  BinPickingPipeline    │   src/pipeline.py
     │  filtra poses → grasp  │   select_target() → plan_grasps()
     └───────────────────────┘
```

Responsabilidades de cada unidad:

- **Parser** (`InstructionParser`): convierte texto crudo en una `Instruction`
  estructurada. **No conoce la escena** (es un mapeo texto → intención). El
  *factory* `make_parser(backend)` elige la implementación.
- **Grounder** (`Grounder`): asocia la `Instruction` a los objetos detectados.
  Puntúa cada objeto por coincidencia de atributos (color/forma/tamaño) y, en
  caso de empate, aplica la relación espacial sobre los centroides. Devuelve un
  `GroundingResult` con el `target_obj_id` y los *scores*.
- **Pipeline** (`BinPickingPipeline`): orquesta percepción → grounding → grasp.
  Construye los `ObjectView` a partir de sus `PoseResult` (centroide = traslación
  estimada, atributos de metadatos de simulación o de CLIP), llama al grounder y
  filtra las poses al *target* antes de planificar el agarre.

### Backends del parser

| Backend | Clase | Dependencias | Comportamiento |
|---|---|---|---|
| `deterministic` (**default**) | `DeterministicParser` | ninguna | Léxico controlado ES/EN (`src/language/vocab.py`). Reproducible, sin red. |
| `llm_local` | `LLMLocalParser` | Ollama (extra `language-llm`) | Prompt few-shot que pide un JSON con el esquema. **Fallback al determinista** si no hay servidor Ollama o el JSON no valida. |
| `llm_api` | `LLMApiParser` | proveedor remoto | **Stub / punto de extensión**: lee `LANGUAGE_API_KEY`/`LANGUAGE_API_PROVIDER`; sin clave (o hasta implementar un proveedor concreto) delega en el determinista. No contiene claves. |

Los tres backends devuelven la **misma** `Instruction`, de modo que el resto del
pipeline es agnóstico al parser usado.

### Métodos del grounder

| Método | Cómo obtiene los atributos |
|---|---|
| `attribute` (**default**) | Asume que cada `ObjectView` ya trae `attributes` (de metadatos de simulación o de una pasada previa). |
| `clip_image` | Si faltan atributos y hay `bbox`, los rellena con CLIP sobre el *crop* (`src/language/_clip.py`, lazy/opcional). |

---

## 3. API

### 3.1 Pipeline end-to-end

El lenguaje natural se activa con `PipelineConfig(language_enabled=True)` y se
pasa la instrucción a `run(...)`:

```python
from src.pipeline import BinPickingPipeline, PipelineConfig

config = PipelineConfig(
    language_enabled=True,          # activa el paso de grounding
    parser_backend="deterministic", # deterministic | llm_local | llm_api
    grounder_method="attribute",    # attribute | clip_image
    ambiguity_tolerant=True,        # si es ambiguo, conserva candidatos ordenados
)
pipeline = BinPickingPipeline(config)
pipeline.initialize()

result = pipeline.run(
    rgb, depth, K,
    masks=masks,
    instruction="dame el cubo rojo de la izquierda",
)
# result.grasps  -> trayectorias planificadas solo para el/los target(s)
# result.grounding -> GroundingResult (target_obj_id, scores, method, ...)
# result.instruction -> Instruction parseada
```

> Nota: la instrucción solo se aplica si `language_enabled=True`. Si el grounding
> es ambiguo y `ambiguity_tolerant=True`, `run()` conserva todas las poses
> candidatas ordenadas por *score* (en vez de descartarlas).

También se puede invocar el paso de selección de forma aislada sobre una lista de
`PoseResult` ya estimadas:

```python
poses_target, grounding, instr = pipeline.select_target(
    poses, "pick the blue sphere"
)
```

### 3.2 Parser y grounder por separado (sin sim)

```python
from src.language import make_parser, Grounder
from src.language.schema import ObjectView

parser = make_parser("deterministic")           # o "llm_local" / "llm_api"
instr = parser.parse("dame el cubo rojo de la izquierda")
# instr.target.color == "red", instr.target.shape == "cube"
# instr.spatial.relation == "left_of", instr.backend == "deterministic"

objs = [
    ObjectView(0, centroid=(-0.20, 0.0, 0.5), attributes={"color": "red", "shape": "cube"}),
    ObjectView(1, centroid=( 0.20, 0.0, 0.5), attributes={"color": "red", "shape": "cube"}),
]
grounder = Grounder(method="attribute")          # o method="clip_image"
res = grounder.ground(instr, objs)
# res.target_obj_id == 0  (desempate por left_of)
# res.method == "spatial", res.scores == {0: 1.0, 1: 1.0}
```

### 3.3 Modelo de datos (resumen)

Definido en [`src/language/schema.py`](../src/language/schema.py) como
*dataclasses* puras (sin numpy/torch):

- **`TargetSpec`**: `color` (`red|blue|green|yellow`), `shape`
  (`cube|sphere|cylinder|box`), `size` (`small|large`), `raw_noun`. Método
  `is_empty()` indica si no hay atributo discriminativo.
- **`SpatialRelation`**: `relation` (`left_of|right_of|nearest|farthest|on_top`)
  y `anchor` opcional.
- **`Instruction`**: `raw_text`, `target` (`TargetSpec`), `intent`
  (`pick|pick_then_place|sequence`), `spatial`, `steps` (sub-instrucciones para
  secuencias), `confidence`, `backend`.
- **`ObjectView`**: vista ligera de un objeto detectado — `obj_id`, `centroid`
  (x,y,z en metros), `attributes`, `bbox`. Desacopla el grounder de `PoseResult`.
- **`GroundingResult`**: `target_obj_id` (o `None`), `scores` (`{obj_id: score}`),
  `method` (`attribute|clip_image|spatial`), `rejected`, `ambiguous`.

---

## 4. Backends y dependencias

### Determinista (por defecto) — sin dependencias

No requiere instalar nada extra ni descargar pesos: usa el léxico controlado de
`src/language/vocab.py`. Es la opción reproducible para CI y para la defensa.

### LLM local (Ollama) — extra `language-llm`

```bash
pip install -e ".[language-llm]"      # o: uv sync --extra language-llm
ollama pull qwen2.5:3b-instruct       # modelo por defecto en llm_local.py
```

El modelo y el host por defecto están definidos en
[`src/language/backends/llm_local.py`](../src/language/backends/llm_local.py):
`qwen2.5:3b-instruct` en `http://localhost:11434`. Si el servidor Ollama no
responde o el JSON devuelto no valida, el parser **cae automáticamente** al
determinista (degradación elegante). Se puede cambiar el modelo:

```python
parser = make_parser("llm_local", model="qwen2.5:3b-instruct")
```

### LLM por API (stub) — extensión opcional

`llm_api` es un **punto de extensión** sin proveedor concreto implementado: lee
`LANGUAGE_API_PROVIDER` y `LANGUAGE_API_KEY` del entorno; si no hay clave (o
mientras no se implemente la llamada) delega en el determinista. **No contiene
claves.**

---

## 5. Demo y dashboard

### CLI — `experiments/run_pick_language.py`

Modo `--dry-run`: solo *parsing* + *grounding* sobre una escena sintética fija de
3 objetos (no requiere CoppeliaSim). Imprime y guarda el JSON resultante en
`experiments/results/language_pick/last_dry_run.json`:

```bash
python experiments/run_pick_language.py \
    --instruction "dame el cubo rojo de la izquierda" --dry-run
```

Opciones: `--parser-backend {deterministic,llm_local,llm_api}`,
`--scene {multi,clutter}`, `--render`. Sin `--dry-run`, intenta la ruta E2E con
CoppeliaSim (`run_pick_battery.run_language_pick`); si esa ruta aún no está
disponible en el entorno, lanza un `NotImplementedError` que sugiere usar
`--dry-run`.

### Dashboard (Streamlit)

`dashboard.py` incluye la sección **"🗣️ Lenguaje natural"**: un campo de texto
donde se escribe la instrucción y se visualiza el *parsing* (color/forma/tamaño/
relación) y la selección de objetivo con su tabla de *scores*. Usa la escena demo
de 3 objetos (cubo rojo izq., cubo azul centro, esfera roja der.).

```bash
pip install -e ".[dashboard]"   # o: uv sync --extra dashboard
streamlit run dashboard.py
```

---

## 6. Resultados consolidados (exp16–exp26)

La feature consolida 9 exploraciones post-TFM. Cada cifra proviene del documento
de exploración indicado (verificar siempre la fuente; no hay cifras inventadas).
La métrica común es **selection accuracy** (objeto correcto seleccionado).

| Exp | Capacidad | Resultado clave | n / setup | Fuente |
|----:|-----------|-----------------|-----------|--------|
| exp16 | CLIP text-conditioning (color) | **98.6 %** selection accuracy | n_val=800 | [04_vla_lite_clip.md](exploraciones/04_vla_lite_clip.md) |
| exp17 | Robustez lingüística (6 familias no vistas) | **100 %** (900/900 frases) | 150×6 familias | [05_vla_robustness.md](exploraciones/05_vla_robustness.md) |
| exp18 | Multi-atributo color+forma | **99.9 %** global (color 100 %, forma 99.8 %, combinado 100 %) | n=1200 val | [06_vla_shapes.md](exploraciones/06_vla_shapes.md) |
| exp20 | Multi-objeto N=2..5 | **100 %** global (N=2,3,4,5 = 100 %) | n=1500 val | [08_multi_object.md](exploraciones/08_multi_object.md) |
| exp22 | Atributo continuo TAMAÑO | **99.9 %** global (N=5: 99.7 %) | n=1500 val | [09_size_attribute.md](exploraciones/09_size_attribute.md) |
| exp23 | Instrucciones secuenciales multi-step | **8/8 secuencias**, **20/20 pasos** (100 %) | 8 escenas curadas | [10_sequential_instructions.md](exploraciones/10_sequential_instructions.md) |
| exp24 | CLIP-image grounding (sin atributos declarados) | **100 %** selection accuracy | n=500 val | [11_clip_image_grounding.md](exploraciones/11_clip_image_grounding.md) |
| exp25 | Robustez CLIP-image (domain randomization) | media **95.3 %**, **12/12 condiciones ≥ 75 %** (mín 77.7 % @ oclusión 60 %), CI 95 % | 300 escenas × 12 cond. | [12_robustness_domain_random.md](exploraciones/12_robustness_domain_random.md) |
| exp26 | Razonamiento espacial (leftmost/closest/topmost…) | **98.4 %** global (13 templates, mín 94.9 %) | n=1500 val | [13_spatial_reasoning.md](exploraciones/13_spatial_reasoning.md) |

> Las exploraciones exp16–exp26 evalúan el modelo *VLA-lite* (Diffusion Policy
> condicionado por CLIP + gate) sobre datos **sintéticos/controlados**. El
> subsistema `src/language/` consolida ese conocimiento en una API
> parser+grounder reutilizable; el grounder por atributos/espacial reproduce de
> forma determinista la lógica de selección validada en esas exploraciones. La
> evaluación de la ruta E2E con CoppeliaSim guiada por lenguaje no está aún
> cerrada (ver limitaciones).

---

## 7. Limitaciones

1. **Vocabulario controlado en el determinista**: el parser por defecto reconoce
   solo el léxico de `src/language/vocab.py` (4 colores, 4 formas, 2 tamaños, 5
   relaciones espaciales, en ES/EN). Frases fuera de ese léxico necesitan el
   backend `llm_local`. El reconocimiento es por límites de palabra; sinónimos no
   listados o errores tipográficos no se capturan.
2. **El grounding por atributos asume metadatos o CLIP**: el método `attribute`
   espera que los `ObjectView` ya traigan `attributes`. Para inferirlos de la
   imagen hay que usar `clip_image` (extra opcional con CLIP) o proveerlos desde
   la simulación.
3. **`llm_api` es un stub**: no implementa ningún proveedor concreto; sin
   `LANGUAGE_API_KEY` (o hasta implementar la llamada) delega en el determinista.
4. **Razonamiento espacial absoluto**: las relaciones operan sobre los centroides
   (extremos x/y/z, más cercano/lejano). No hay relaciones relativas a otro
   objeto ("a la derecha del cubo rojo") en el grounder determinista.
5. **Resultados sobre datos sintéticos/controlados**: las cifras de la sección 6
   provienen de escenas sintéticas (exp16–exp26). La integración E2E guiada por
   lenguaje en sim (`run_pick_battery.run_language_pick`) puede no estar
   disponible en todos los entornos; usar `--dry-run` para el parsing+grounding.

---

*Documento de la feature de lenguaje natural — consolida exp16–exp26. Ver también
[`COMPARATIVA_SOTA_LENGUAJE.md`](COMPARATIVA_SOTA_LENGUAJE.md) para el
posicionamiento frente al estado del arte.*
