# Guión de defensa — TFM bin-picking 6-DoF (FoundationPose + Diffusion Policy)

**Duración objetivo:** ~20 min · 19 slides · cierre balanceado (aportes + valor) · con demo
**Autor:** Giocrisrai Godoy · Máster en Ingeniería Matemática y Computación

> **Cómo usar este guión:** cada slide trae *Título · Contenido visual · Bullets · Guión hablado · Tiempo*. El "guión hablado" es lo que decís en voz alta (no se pone en la slide). Las cifras son las verificadas en la memoria — no las cambies sin cruzar con el documento.
> **Cifras canónicas:** AUC ADD-S 0.908 (YCB-V) / 0.957 (T-LESS); ciclo p95 6,29 s / 6,68 s; latencia DP 118 ms; K=2 modos; pick-and-place E2E 84 %; coste ~1 920 USD.

---

## Slide 1 — Portada (0:30)
**Visual:** Título del TFM, tu nombre, titulación, tutor, fecha. Logo UNIR.
**Bullets:**
- Estimación de pose 6-DoF mediante arquitecturas Transformer y modelos de difusión para bin-picking robótico
**Guión hablado:**
> "Buenos días. Mi nombre es Giocrisrai Godoy y voy a presentar mi Trabajo Fin de Máster: la integración de FoundationPose y Diffusion Policy para bin-picking robótico, un pipeline que estima la pose 6-DoF de un objeto y planifica su agarre, ejecutado en simulación y reproducible en hardware accesible."

---

## Slide 2 — El problema: bin-picking 6-DoF (1:00)
**Visual:** Imagen de un bin con piezas apiladas/desordenadas (foto industrial o render de la escena).
**Bullets:**
- Coger piezas desordenadas de un contenedor es un problema abierto en robótica industrial
- Requiere: estimar la pose 6-DoF (3 traslación + 3 rotación) y planificar una trayectoria de agarre
- Retos: objetos sin textura, simetrías, oclusiones, piezas que cambian
**Guión hablado:**
> "El bin-picking — coger piezas desordenadas de un contenedor — sigue siendo un problema abierto. El robot debe resolver dos cosas: primero, *dónde y cómo está* cada pieza, es decir su pose de seis grados de libertad; y segundo, *cómo agarrarla*. La dificultad está en objetos sin textura, simétricos, parcialmente ocultos, y que cambian de un lote a otro. Mi trabajo aborda la integración de estos dos sub-problemas en un único pipeline matemáticamente formalizado."

---

## Slide 3 — Objetivo e hipótesis (1:30)
**Visual:** Las tres hipótesis en tres cajas (H1/H2/H3), cada una con su criterio.
**Bullets:**
- **Objetivo:** integrar percepción (Transformers) y planificación (difusión) en un pipeline E2E validado
- **H1 — Precisión:** FoundationPose mejora Mean AR (BOP) ≥3 pp vs baseline GDR-Net++
- **H2 — Planificación multimodal:** trayectorias con score ≥0.95, muestreo <50 ms, ≥3 modos
- **H3 — Viabilidad:** ciclo end-to-end p95 <10 s sin GPU dedicada
**Guión hablado:**
> "El objetivo general es integrar ambos paradigmas en un pipeline end-to-end validado experimentalmente. Para contrastarlo planteé tres hipótesis con criterios cuantitativos estrictos: H1 sobre la precisión de la pose, H2 sobre la calidad de la planificación multimodal, y H3 sobre la viabilidad temporal en hardware accesible. Las tres se contrastan formalmente con intervalos de confianza al 95 %, y voy a ser transparente: una de ellas se acepta solo parcialmente, y explicaré por qué."

---

## Slide 4 — Estado del arte y brecha (1:30)
**Visual:** Línea temporal o mapa: métodos clásicos → Transformers (FoundationPose, GenFlow) → difusión (Diffusion Policy); marcar la brecha.
**Bullets:**
- Estimación de pose: de métodos clásicos a Transformers (FoundationPose, CVPR 2024); FreeZeV2 lidera objetos no vistos en BOP 2024
- Planificación: Diffusion Policy (modelos de difusión para acción multimodal)
- **Brecha:** no existía un pipeline que integrara matemáticamente percepción Transformer + planificación por difusión para bin-picking
**Guión hablado:**
> "El estado del arte avanza rápido. En estimación de pose, los Transformers como FoundationPose marcaron un hito en 2024, y métodos recientes como FreeZeV2 ya lideran en objetos no vistos. En planificación, las Diffusion Policies aportan acción multimodal. Pero revisando la literatura indexada hasta 2026, no encontré un trabajo que *integrara matemáticamente* ambos paradigmas para bin-picking. Esa es la brecha que cubre este TFM: no propongo un estimador nuevo, sino el acoplamiento formal de dos paradigmas existentes y su validación end-to-end."

---

## Slide 5 — Pipeline integrado (1:30)
**Visual:** `fig_pipeline_montaje.png` (diagrama del montaje del pipeline).
**Bullets:**
- Percepción (FoundationPose) → pose SE(3) → conditioning → Diffusion Policy → 16 waypoints → IK del robot → ejecución en CoppeliaSim
- Representación común: el grupo SE(3) conecta percepción y planificación
**Guión hablado:**
> "Este es el pipeline integrado. FoundationPose estima la pose 6-DoF a partir de RGB-D; esa pose, en el grupo SE(3), condiciona a la Diffusion Policy, que genera una trayectoria de 16 waypoints; cada waypoint se ejecuta mediante cinemática inversa en el simulador CoppeliaSim. La clave del acoplamiento es que ambos componentes operan sobre la misma representación matemática de la pose, SE(3), lo que permite conectarlos sin pérdida de información."

---

## Slide 6 — Fundamento matemático (2:00)
**Visual:** Ecuaciones clave: SO(3)/representación 6D continua, SE(3) como grupo de Lie, score matching + Langevin para la difusión.
**Bullets:**
- Rotaciones: SO(3), representación 6D continua (evita gimbal lock)
- Pose: SE(3) como grupo de Lie
- Atención: scaled dot-product / multi-head (percepción)
- Difusión: score matching y dinámica de Langevin para el muestreo inverso (planificación)
**Guión hablado:**
> "El núcleo matemático del trabajo. Las rotaciones se representan con la parametrización 6D continua sobre SO(3), que evita las discontinuidades del gimbal lock. La pose completa vive en SE(3), el grupo de Lie de los movimientos rígidos. La percepción usa mecanismos de atención —scaled dot-product y multi-head. Y la planificación se formula como un proceso de difusión: aprendemos el *score* de la distribución de trayectorias y muestreamos mediante dinámica de Langevin inversa. Esta formalización unificada es la primera contribución del trabajo."

---

## Slide 7 — Metodología experimental (1:30)
**Visual:** `fig_flujo_experimental.png` (diagrama de flujo experimental).
**Bullets:**
- Datasets BOP: YCB-Video (1098 inst.) y T-LESS (1012 inst.) — piezas industriales sin textura
- Métricas: AUC ADD-S, Mean AR (VSD/MSSD/MSPD) — tratan simetrías correctamente
- Protocolo: splits oficiales BOP, 5 semillas, IC 95 % por bootstrap (B=1000)
- Ablations + análisis de amenazas a la validez
**Guión hablado:**
> "El rigor metodológico es un eje del trabajo. Evalúo sobre dos datasets estándar del benchmark BOP: YCB-Video y T-LESS, este último de piezas industriales sin textura, justo el caso difícil del bin-picking. Uso las métricas oficiales que tratan correctamente los objetos simétricos, los splits oficiales, cinco semillas independientes, e intervalos de confianza al 95 % por bootstrap. Además incluyo estudios de ablación y un análisis explícito de amenazas a la validez. Es un protocolo más detallado que el que reportan muchas publicaciones del área."

---

## Slide 8 — H1: Precisión de pose (1:30)
**Visual:** Tabla/figura comparativa FoundationPose vs GDR-Net++ (AUC ADD-S por dataset, con IC).
**Bullets:**
- AUC ADD-S: **0.908** [IC 0.901–0.916] YCB-V · **0.957** [0.954–0.959] T-LESS
- Recall @10 mm: 95.8 % / 99.7 %
- Mejora vs GDR-Net++: +3.0 pp (YCB-V), +3.6 pp (T-LESS) → supera el umbral
- **Veredicto: H1 ACEPTADA**
**Guión hablado:**
> "Primer resultado, H1. FoundationPose alcanza un AUC ADD-S de 0.908 en YCB-Video y 0.957 en T-LESS, con intervalos de confianza estrechos. La mejora sobre el baseline GDR-Net++ es de 3.0 y 3.6 puntos porcentuales, por encima del umbral de 3 puntos que fijé. H1 se acepta: el pipeline reproduce y mejora el estado del arte en precisión de pose."

---

## Slide 9 — H2: Planificación multimodal (1:30)
**Visual:** Esquema de la Diffusion Policy generando trayectorias; marcar los dos criterios no cumplidos.
**Bullets:**
- Latencia de muestreo DDIM-25: **118 ms** > umbral 50 ms
- Modos diferenciables: **K=2** < objetivo ≥3
- Score medio: 0.96 ≥ 0.95 (sí se cumple)
- **Veredicto: H2 PARCIALMENTE ACEPTADA** (honestidad)
**Guión hablado:**
> "H2 es donde soy más transparente. La Diffusion Policy genera trayectorias de buena calidad —score 0.96, por encima del umbral. Pero dos criterios estrictos no se cumplen: la latencia de muestreo es de 118 milisegundos, por encima de los 50 que fijé; y detecté dos modos diferenciables en lugar de los tres exigidos. Por eso H2 se acepta solo *parcialmente*. La razón es que la política se entrenó sobre trayectorias heurísticas deterministas, así que aprende a imitar al planificador más que a explorar modos alternativos. Lo reporto honestamente porque es un hallazgo, no un fallo a esconder."

---

## Slide 10 — H3: Viabilidad sin GPU dedicada (1:30)
**Visual:** Gráfico de profiling (FP 80 % / sim 17 % / difusión 2 %) + tiempos de ciclo.
**Bullets:**
- Ciclo E2E p95: **6,29 s** (YCB-V) / **6,68 s** (T-LESS) < umbral 10 s (margen >3,3 s)
- PBVS converge en 100 % de las muestras (50/50)
- Cuello de botella: FoundationPose (80,2 % del ciclo)
- Hardware: Apple M1 Pro + Colab T4 (~1 920 USD)
- **Veredicto: H3 ACEPTADA**
**Guión hablado:**
> "H3, la viabilidad. El ciclo completo end-to-end tiene un percentil 95 de 6,3 segundos en YCB-Video y 6,7 en T-LESS, holgadamente por debajo del umbral de 10 segundos. El profiling muestra que el cuello de botella es la percepción —FoundationPose es el 80 % del ciclo—, mientras que la planificación por difusión es apenas el 2 %. Y todo esto corre en hardware accesible: un MacBook M1 Pro más una cuota de Colab. H3 se acepta: el pipeline es viable sin GPU dedicada de alto coste."

---

## Slide 11 — Progresión de la planificación: Iter 5 → 7c (2:00)
**Visual:** **`fig_iter_progression.png`** (la figura nueva: grasp/deposit/IK/pick-and-place a lo largo de las iteraciones).
**Bullets:**
- Aprendizaje supervisado (Iter 5): 60 % pick-and-place E2E
- Refuerzo (DPPO, Iter 6): rompe el agarre (olvido catastrófico) → motivó el currículo
- Currículo + best-of-N + fix de IK (Iter 7c): **84 % E2E, IK 100 %** — supera al baseline
**Guión hablado:**
> "Más allá del contraste de hipótesis, exploré cómo mejorar la planificación de agarre. Esta figura cuenta la historia. Partiendo del aprendizaje supervisado con 60 % de éxito, apliqué ajuste fino por refuerzo, que paradójicamente *rompió* el agarre por olvido catastrófico. Eso motivó un currículo en dos fases, una selección best-of-N por percepción, y la corrección de un problema de convergencia de cinemática inversa. El resultado: 84 % de éxito end-to-end y convergencia de IK del 100 %, superando al baseline supervisado. Es un ejemplo de cómo la evidencia experimental reorientó las decisiones de diseño."

---

## Slide 12 — DEMO: el pipeline en acción (2:00)
**Visual:** **`reel_showcase.mp4`** (reel cinematográfico ~90 s — se puede recortar a 30-45 s para la demo).
**Bullets:**
- Pick-and-place completo: aproximación → agarre → elevación → depósito
- Cámara cinematográfica dedicada (no interfiere con la percepción)
**Guión hablado:**
> "Veámoslo funcionar. [Reproducir reel] Esto es el pipeline ejecutando un ciclo completo de pick-and-place en CoppeliaSim: la cámara estima la pose, la política genera la trayectoria, y el robot aproxima, agarra, eleva y deposita la pieza. La toma usa una cámara cinematográfica dedicada que no interfiere con la percepción del sistema."
**Plan B:** si la corrida en vivo falla, el reel grabado es el respaldo.

---

## Slide 13 — Bin picking guiado por lenguaje natural (1:00) · Capa de lenguaje natural
**Visual:** `fig_language_pick.png` (la instrucción en texto + la pieza seleccionada/agarrada por el robot).
**Bullets:**
- Le dices «dame el cubo rojo de la izquierda» y el robot selecciona y agarra el objeto descrito
- Parser determinista ES/EN + modelo de lenguaje local enchufable (con fallback) — anclaje por atributos (color/forma/tamaño) y relación espacial
- Open-license, corre en portátil; capa **opt-in** sobre el pipeline base (no lo altera)
- Validado E2E en CoppeliaSim: **selección 100 %** (banco controlado: pura n=90 / sim n=9), agarre cinemático 4 mm, IK convergente
**Guión hablado:**
> "Como capa adicional de este trabajo, el pipeline ahora entiende lenguaje natural. Le digo, por ejemplo, «dame el cubo rojo de la izquierda», y el sistema ancla esa instrucción a un objeto concreto por sus atributos —color, forma, tamaño— y su relación espacial, y luego lo agarra con la misma cadena de percepción y planificación de antes. Por defecto usa un parser determinista en español e inglés, reproducible y sin red; opcionalmente se puede enchufar un modelo de lenguaje local. Soy transparente con la validación: la *selección* del objeto correcto acierta el 100 % en banco controlado —noventa instrucciones puras más nueve en simulación— y la robustez ante distractores está cubierta en los experimentos 16 a 26; el agarre sigue siendo cinemático, validado por proximidad pinza-objeto de 4 milímetros y convergencia de la cinemática inversa. Es una capa opt-in: no altera el pipeline base, lo extiende."

---

## Slide 14 — Honestidad y limitaciones (1:00)
**Visual:** Lista de limitaciones declaradas (L1–L5), destacando snap+attach y sim-only.
**Bullets:**
- El agarre usa **snap+attach** (cinemático, estándar en sims comerciales): valida la cadena percepción→planificación→IK, NO la mecánica física de fricción
- Validación **solo en simulación** (CoppeliaSim), sin robot real
- La Diffusion Policy imita una heurística (no la supera por diseño en este escenario)
**Guión hablado:**
> "Es importante declarar los límites con honestidad. El agarre se modela con la técnica de snap+attach, estándar en simuladores comerciales: valida toda la cadena percepción-planificación-cinemática-ejecución, pero no la mecánica física del agarre por fricción. La validación es en simulación, sin robot real. Y la política imita a una heurística por diseño. Reconocer esto no debilita el trabajo: lo hace verificable y honesto, y define con precisión qué se demostró y qué queda por demostrar."

---

## Slide 15 — Aportes 1 y 2 (1:00)
**Visual:** Dos aportes en cajas.
**Bullets:**
- **Aporte 1:** integración matemática unificada de dos paradigmas (Transformers + difusión) sobre SE(3)/SDEs
- **Aporte 2:** reproducibilidad cuantitativa del estado del arte sin GPU dedicada (~1 920 USD)
**Guión hablado:**
> "Las contribuciones. La primera es metodológica: la integración matemática formal de percepción por Transformers y planificación por difusión, conectadas en SE(3) y mediante ecuaciones diferenciales estocásticas. La segunda es de reproducibilidad: demostrar que se puede reproducir el estado del arte en hardware de menos de dos mil dólares, no en un clúster."

---

## Slide 16 — Aportes 3 y 4 (1:00)
**Visual:** Dos aportes en cajas.
**Bullets:**
- **Aporte 3:** rigor metodológico superior al estándar (IC, bootstrap, 5 semillas, ablations, amenazas a la validez)
- **Aporte 4:** validación end-to-end con evidencia visual reproducible (video + repositorio público)
**Guión hablado:**
> "La tercera contribución es el rigor metodológico: intervalos de confianza, bootstrap, múltiples semillas, ablaciones y análisis de amenazas a la validez, un nivel de detalle por encima del estándar del área. Y la cuarta es la validación end-to-end con evidencia visual reproducible: un video y un repositorio público que permiten a cualquiera reproducir los resultados."

---

## Slide 17 — Valor y potencialidad (1:30)
**Visual:** Comparativa de coste (~1 920 USD vs 15-150k) + contexto de mercado.
**Bullets:**
- Democratización: ~1 920 USD vs 15 000–150 000 USD de un setup industrial (1-2 órdenes de magnitud)
- Mercado bin-picking: ~2 000 M USD (2025) → 7 300 M (2036), CAGR 12,5 % (FactMR)
- Nicho: prueba de concepto E2E de bajo coste; la autonomía total aún recurre a human-in-the-loop
- Aplicaciones: logística/e-commerce, manufactura, clasificación
**Guión hablado:**
> "¿Por qué importa esto más allá de lo académico? El mercado de bin-picking crece de dos mil a más de siete mil millones de dólares hacia 2036. Las soluciones comerciales cuestan entre quince y ciento cincuenta mil dólares por estación; este pipeline corre en menos de dos mil. Eso lo sitúa como prueba de concepto reproducible de bajo coste, justo en el nicho donde la autonomía total todavía no es viable y se recurre al human-in-the-loop. Las aplicaciones directas están en logística, manufactura y clasificación."
**Nota:** las cifras de mercado son de market-research (estimaciones, no peer-reviewed) — presentarlas como contexto, no como medición propia.

---

## Slide 18 — Trabajo futuro (1:00)
**Visual:** Líneas de trabajo futuro en lista.
**Bullets:**
- Validación del agarre físico real (fricción, ventosa de vacío — Soraruf 2025, Kim 2022)
- Aceleración por flow-matching (FlowPolicy, ManiFlow — inferencia en 1-2 pasos)
- Transferencia a robot real (cámaras RealSense); pose category-level; edge (Jetson)
- Adopción de SOTA reciente (FreeZeV2, BOP-Industrial 2025)
- **Portabilidad a otros manipuladores** (Franka, KUKA): la planificación en espacio de tarea + IK genérico lo permiten por diseño; falta validación empírica
**Guión hablado:**
> "El trabajo futuro tiene líneas concretas, ancladas en literatura reciente. La más importante: validar el agarre físico real, reemplazando snap+attach por modelos de contacto con verificación de sellado o predicción de robustez. Acelerar la planificación con flow-matching, que reduce la inferencia a uno o dos pasos. Transferir a un robot real con cámaras RealSense. Y adoptar los avances del benchmark BOP 2025, ahora con datasets industriales específicos de bin-picking."

---

## Slide 19 — Cierre (0:30)
**Visual:** Mensaje de cierre + datos de contacto / repositorio.
**Bullets:**
- Pipeline E2E integrado, validado (H1 y H3 aceptadas, H2 parcial), reproducible en hardware accesible
- 84 % pick-and-place E2E · repositorio público
- Gracias — preguntas
**Guión hablado:**
> "En resumen: presenté un pipeline end-to-end que integra matemáticamente percepción y planificación, validado con rigor —H1 y H3 aceptadas, H2 parcialmente—, reproducible en hardware de menos de dos mil dólares, con un 84 % de éxito en pick-and-place. Todo el código y la evidencia están en un repositorio público. Muchas gracias; quedo a disposición para sus preguntas."

---

# Anexo — Preguntas anticipadas (Q&A)

**P: ¿El robot agarra de verdad o es un truco?**
> Honestamente: el agarre es snap+attach, cinemático. Valida la cadena percepción→planificación→IK→ejecución, que es lo que el trabajo demuestra. La mecánica física de fricción es trabajo futuro explícito, con métodos ya identificados (Soraruf 2025, Kim 2022).

**P: Si H2 no se cumple del todo, ¿el trabajo falla?**
> No. H2 expone un límite real y diagnosticado: la política imita trayectorias deterministas, de ahí los 2 modos y la latencia. Reportarlo con honestidad es parte del rigor. Además, la planificación es solo el 2 % del ciclo, así que los 118 ms no comprometen la viabilidad (H3).

**P: ¿Por qué simulación y no robot real?**
> Por alcance y coste: el TFM valida la integración matemática y la viabilidad temporal end-to-end, que no requieren hardware físico. La transferencia a robot real es la primera línea de trabajo futuro.

**P: FoundationPose ya existía, ¿qué aportás?**
> No propongo un estimador nuevo. El aporte es la *integración matemática formal* de percepción Transformer + planificación por difusión, su validación E2E con rigor, y la reproducibilidad en hardware accesible. Es una brecha que no estaba cubierta en la literatura indexada.

**P: ¿Por qué la Diffusion Policy si imita una heurística?**
> En este escenario controlado sirve para validar el pipeline E2E. Su valor está en escenarios donde la heurística no escala (clutter, oclusión, multi-objeto). El trabajo de iteraciones (best-of-N, currículo) muestra que la política puede superar al baseline supervisado: 84 % vs 60 %.

**P: ¿Esto solo sirve para el robot que usaste, o es exportable a otros?**
> Es exportable por diseño. La Diffusion Policy planifica en espacio de tarea —genera poses 6-DoF del efector, no ángulos articulares—, y el IK es genérico (damped least squares sobre base→efector). El robot está en una configuración desacoplada (ya contempla Franka/UR). Portar a otro brazo serie es reconfigurar + reentrenar sobre el nuevo workspace, no reescribir el método. Honestamente: es una propiedad arquitectónica, no la validé con otro robot físico — eso es trabajo futuro.

**P: El 84 % de pick-and-place, ¿sobre qué se mide?**
> Sobre 50 picks en CoppeliaSim, semilla fija 2026, mismo protocolo que las iteraciones anteriores. Combina grasp plausible + deposit plausible + IK convergido. Es el mejor resultado de la línea de planificación.

---

> **Checklist pre-defensa:** (1) abrir el .docx en Word + F9 para índices; (2) probar el reel en el proyector; (3) tener el reel showcase descargado localmente como Plan B; (4) ensayar tiempos (apuntar a 18-19 min para dejar margen de Q&A).
>
> **Descarga de los reels (respaldo):** assets del release [`v0.1.0`](https://github.com/Giocrisrai/pose6dof-transformers-diffusion/releases/tag/v0.1.0) — [`reel_showcase.mp4`](https://github.com/Giocrisrai/pose6dof-transformers-diffusion/releases/download/v0.1.0/reel_showcase.mp4), [`reel_resumen.mp4`](https://github.com/Giocrisrai/pose6dof-transformers-diffusion/releases/download/v0.1.0/reel_resumen.mp4), [`demo_v7c.mp4`](https://github.com/Giocrisrai/pose6dof-transformers-diffusion/releases/download/v0.1.0/demo_v7c.mp4).
