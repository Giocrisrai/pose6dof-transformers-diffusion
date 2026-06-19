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
`--scene {multi,clutter}`, `--render`. **Sin `--dry-run` el CLI ejecuta la ruta
E2E real** sobre CoppeliaSim (`src.simulation.language_pick.run_language_pick`):
construye la escena, groundea la instrucción y realiza el pick sobre el objeto
elegido. Esa ruta requiere CoppeliaSim en `localhost:23000`. La descripción
completa de la ruta E2E está en la [sección 7](#7-ejecución-e2e-en-coppeliasim).

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

> **Nota de numeración**: las etiquetas `expNN` corresponden a los scripts de
> `experiments/` (p. ej. `exp16_vla_lite_clip.py`), mientras que los ficheros de
> `docs/exploraciones/` se numeran 04–13 (Exploración 4 = exp16, …, Exploración 13
> = exp26). La columna *Fuente* enlaza el documento correcto en cada fila.

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
> ruta E2E con CoppeliaSim guiada por lenguaje (escena multi-objeto → grounding →
> pick real) está implementada y documentada en la [sección 7](#7-ejecución-e2e-en-coppeliasim).

---

## 7. Ejecución E2E en CoppeliaSim

Las secciones anteriores describen el núcleo puro (parser + grounder) y la batería
de *selection accuracy*, que corren **sin simulador**. Esta sección documenta la
**ruta extremo a extremo en CoppeliaSim**: de una instrucción en lenguaje natural a
un *pick* físico sobre el objeto elegido dentro del simulador. El código vive en
[`src/simulation/language_pick.py`](../src/simulation/language_pick.py); toda esta
ruta requiere **CoppeliaSim escuchando en `localhost:23000`**.

### 7.1 `run_language_pick` — el pick guiado por lenguaje

`run_language_pick(instruction, scene="multi", parser_backend="deterministic",
render=False, n_objects=3, with_shapes=None, seed=42)` es la función orquestadora:

1. Planifica una escena multi-objeto (`plan_language_scene`): obj 0 es el *target*
   (cubo rojo por defecto) y el resto son distractores. Con `scene="clutter"` se
   activa `with_shapes=True` (formas variadas en los distractores).
2. Crea las primitivas en el simulador (`apply_scene`) vía `createPrimitiveShape`
   (cube/sphere/cylinder), las pinta según su color y aparca cualquier objeto
   preexistente para que no interfiera.
3. Groundea la instrucción contra los objetos de la escena (`select_sim_target`)
   y elige el *target*.
4. Ejecuta el **pick real** sobre el objeto elegido (`run_pick_sequence`).
5. Devuelve un *payload* JSON con:
   - `parsed` (color/forma/tamaño/relación espacial/backend del parser),
   - `grounding` (`target_obj_id`, `method`, `ambiguous`, `scores`),
   - `scene` (lista de objetos con color/forma/posición),
   - `selection_correct` (si se eligió el objetivo correcto, obj 0),
   - `pick` (métricas: `tip_grasp_proximity_m`, `object_moved_m`,
     `grasp_plausible`, `ik_converged`) o `None` si no hubo match,
   - `mp4_path` (vídeo del pick, solo si `render=True`).

Si CoppeliaSim no es accesible en `localhost:23000`, la llamada falla con un error
de conexión.

### 7.2 CLI E2E

```bash
# E2E real (requiere CoppeliaSim en :23000): escena clutter + render del MP4
.venv/bin/python experiments/run_pick_language.py \
    --instruction "dame el cubo rojo" --scene clutter --render

# Solo grounding, sin simulador (parsing + selección sobre escena fija):
.venv/bin/python experiments/run_pick_language.py \
    --instruction "dame el cubo rojo" --dry-run
```

Con `--dry-run` el CLI hace únicamente *parsing* + *grounding* (no toca el
simulador); sin `--dry-run` ejecuta la ruta E2E completa de `run_language_pick`.
Flags relevantes: `--scene {multi,clutter}`, `--parser-backend {deterministic,
llm_local,llm_api}`, `--render`.

### 7.3 Variedad de formas

La escena soporta tres primitivas mediante `createPrimitiveShape`:
**cube / sphere / cylinder**. Esto habilita instrucciones que discriminan por
forma además de por color, p. ej. `"the red sphere"` o `"pick the red cube"`. Con
`with_shapes=True` (automático en `scene="clutter"`) los distractores adoptan
formas variadas, de modo que el grounder debe atender a la forma para acertar.

### 7.4 Batería de *selection accuracy*

```bash
# Métrica pura (sin simulador): accuracy por dificultad
.venv/bin/python experiments/run_language_battery.py --n-scenes 30

# Además ejecuta picks reales en sim y añade sim_selection_accuracy
.venv/bin/python experiments/run_language_battery.py --n-scenes 10 --sim
```

La batería genera escenas alternando **tres dificultades** y mide si el grounding
selecciona el objeto correcto:

- **`color`**: el *target* se distingue solo por color (baseline trivial).
- **`shape`**: el *target* comparte color con ≥1 distractor pero tiene forma
  única → exige desambiguar por forma.
- **`spatial`**: todos los objetos comparten color y forma; solo la posición x
  (más a la izquierda) distingue al *target* → exige razonamiento espacial.

La parte de evaluación es **pura** (no necesita simulador). El grounder
determinista obtiene **1.0 en las tres dificultades puras** (verificado:
`--n-scenes 30` → color 10/10, shape 10/10, spatial 10/10). Esta cifra funciona
como **guard de regresión**: un grounder más débil bajaría en `shape` o `spatial`
(donde el color por sí solo no basta), por lo que un 1.0 sostenido en esas dos
dificultades confirma que la desambiguación por forma y por posición sigue activa.

Con `--sim` la batería ejecuta además un *pick* real por escena (requiere
CoppeliaSim) y añade `sim_selection_accuracy` al reporte. **Esta cifra depende de
la ejecución en vivo** (estado del simulador, físicas del pick) y se computa en
*runtime*; no es un valor fijo. El reporte completo (agregado + filas por escena)
se escribe en
[`experiments/results/language_battery/report.json`](../experiments/results/language_battery/report.json).

### 7.5 Reel de demo

```bash
.venv/bin/python experiments/make_language_reel.py   # requiere CoppeliaSim
```

Genera un *reel* en *crescendo*: por cada instrucción de la lista `CRESCENDO`
(de simple a difícil: color → forma → relación espacial) ejecuta un pick real,
superpone la instrucción y el *target* elegido con `reel_overlay`, y compila un
único MP4 en `experiments/results/language_reel/language_reel.mp4`. Sin
CoppeliaSim solo se puede inspeccionar la lista `CRESCENDO` (testeable).

### 7.6 Reproducibilidad: qué corre sin simulador y qué lo requiere

| Componente | Comando | ¿Requiere CoppeliaSim? |
|---|---|---|
| Núcleo puro (parser + grounder) | API `src.language` / `src.simulation.language_pick` (funciones puras) | **No** |
| Grounding-only del CLI | `run_pick_language.py … --dry-run` | **No** |
| Batería de *selection accuracy* | `run_language_battery.py --n-scenes 30` | **No** |
| Tests base | `pytest -m "not slow and not integration"` | **No** |
| Pick E2E | `run_pick_language.py …` (sin `--dry-run`) | **Sí** (:23000) |
| Batería con picks reales | `run_language_battery.py --sim` | **Sí** (:23000) |
| Reel de demo | `make_language_reel.py` | **Sí** (:23000) |

La parte pura (núcleo, batería sin `--sim`, `--dry-run`, tests) es determinista y
reproducible en CI o en la defensa sin levantar el simulador. La parte E2E
(pick real, `--sim`, reel) necesita CoppeliaSim activo en `localhost:23000`.

---

## 8. Limitaciones

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
   provienen de escenas sintéticas (exp16–exp26). La ruta E2E guiada por lenguaje
   en sim (`src.simulation.language_pick.run_language_pick`, ver
   [sección 7](#7-ejecución-e2e-en-coppeliasim)) está implementada pero requiere
   CoppeliaSim en `localhost:23000`; sin simulador, usar `--dry-run` para el
   *parsing*+*grounding* o la batería pura (`run_language_battery.py` sin `--sim`).
   La `sim_selection_accuracy` de la batería se computa en vivo y depende de la
   ejecución concreta del pick.

---

*Documento de la feature de lenguaje natural — consolida exp16–exp26. Ver también
[`COMPARATIVA_SOTA_LENGUAJE.md`](COMPARATIVA_SOTA_LENGUAJE.md) para el
posicionamiento frente al estado del arte.*
