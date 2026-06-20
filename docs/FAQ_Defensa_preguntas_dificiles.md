# Banco de preguntas difíciles del tribunal — defensa TFM
Respuestas modelo (~30–40 s). Principio: **reconocer → fundamentar → reencuadrar como decisión consciente.** Nunca a la defensiva.

## Bloque A — Validez y alcance (lo más probable)

**1. "Tu agarre es snap+attach, no simulas la física de fricción. ¿Qué validez tiene entonces?"**
> Es una limitación que declaro explícitamente (L3). El *snap+attach* aísla a propósito lo que SÍ quería validar: la cadena percepción → planificación → cinemática inversa, es decir, que la pose estimada condiciona una trayectoria ejecutable y alcanzable. La mecánica de contacto es un problema ortogonal, con su propia literatura (IPC-GraspSim, modelos de sellado). Validar las dos cosas a la vez habría confundido las fuentes de error. Por eso lo separo, lo declaro, y lo dejo como primera línea de trabajo futuro.

**2. "Todo es simulación, en CoppeliaSim. ¿Cómo sé que funciona en un robot real?"**
> No lo sé, y no lo afirmo: lo declaro como límite (L3) y como el salto pendiente número uno. Lo que sí garantizo es que la arquitectura es transferible por diseño: la política planifica en espacio de tarea, no en ángulos articulares, y la cinemática inversa es genérica. El sim-to-real es real, pero mi aporte está en el marco de integración y su reproducibilidad, no en un despliegue industrial, que sería otro TFM.

**3. "FoundationPose tiene licencia no comercial. ¿Qué aporte real hay si no se puede usar en industria?"**
> Correcto, es la limitación L5 y la digo en la defensa. Mi aporte es académico y metodológico: el puente matemático entre percepción y planificación, que es independiente del estimador concreto. FoundationPose es intercambiable por uno de licencia permisiva —GDR-Net++ es Apache 2.0— sin tocar el resto del pipeline, justamente porque el acoplamiento es por la pose en SE(3), no por la arquitectura interna.

## Bloque B — La hipótesis parcial (te van a apretar)

**4. "H2 no se cumple: 118 ms y 2 modos. ¿No invalida eso tu tesis?"**
> Al contrario, la matiza con honestidad. H2 tiene tres criterios: la calidad (score 0.96) se cumple; la latencia (118 ms > 50) y la multimodalidad (2 < 3) no. Pero el umbral de 50 ms resultó más estricto que el requisito real: la planificación es solo el 2,3 % del ciclo, despreciable frente a la percepción. Y diagnostiqué la causa de los pocos modos: la política imita un planificador determinista, no demostraciones reales. Aceptar H2 "solo en parte" y explicar por qué es más valioso que forzar un sí.

**5. "Si la Diffusion Policy imita al heurístico y es más lenta, ¿para qué usarla?"**
> En este montaje, con datos sintéticos heurísticos, efectivamente imita: lo cuantifico y lo digo. El valor de la difusión es el marco —captura distribuciones multimodales de forma matemáticamente fundamentada— y su potencial con demostraciones reales o *flow-matching*, donde el muestreo baja a uno o dos pasos. Mi contribución no es "la difusión gana hoy", sino formalizar el acoplamiento SE(3)→SDE que permite explotarla cuando los datos lo habiliten.

## Bloque C — Contribución y novedad

**6. "No propones un método nuevo, solo integras dos existentes. ¿Dónde está la contribución de máster?"**
> La contribución es precisamente el puente, y es matemática, no de ingeniería. Derivo cómo la pose en SE(3) condiciona la SDE inversa que genera la trayectoria, conectando geometría de grupos de Lie con procesos estocásticos. Ningún trabajo previo lo había hecho para bin-picking. Integrar no es trivial cuando la integración exige un formalismo común; ahí está la originalidad, más la reproducibilidad cuantitativa del estado del arte en hardware accesible.

**7. "Comparas con la métrica oficial Mean AR pero recomputas AUC ADD-S local. ¿Es justa la comparación con GDR-Net++?"**
> Lo separo con cuidado. La comparación de +3,0 y +3,6 puntos es sobre el Mean AR del *leaderboard* oficial, métrica idéntica para ambos. El AUC ADD-S es una validación local complementaria que recomputo con bootstrap para tener intervalos de confianza propios; la declaro como limitación de constructo (L1) porque es métrica relacionada, no idéntica. Nunca mezclo ambas para sostener H1.

## Bloque D — Rigor y matemática

**8. "¿Por qué representación 6D continua y no cuaterniones? Justifícalo."**
> Por un resultado de Stuelpnagel (1964): no existe parametrización continua de SO(3) con menos de cinco dimensiones. Cuaterniones (4D) y ángulos de Euler (3D) tienen discontinuidades —doble cobertura, gimbal lock— que degradan la convergencia de la red. La representación 6D de Zhou et al. (2019) es continua y por eso estabiliza el entrenamiento. Es una decisión fundamentada formalmente, no por conveniencia.

**9. "El refuerzo empeoró el resultado (60→28 %). ¿No indica fragilidad de tu enfoque?"**
> Indica olvido catastrófico, un fenómeno conocido que afecta a los mejores laboratorios, no fragilidad del pipeline. Lo relevante es la respuesta: lo diagnostiqué, apliqué un currículo y selección best-of-N, y remonté a 84 %, por encima del sistema original. Lo incluyo en la defensa a propósito: muestra método científico —diagnosticar y corregir— en lugar de presentar solo el resultado pulido.

**10. "¿Son estadísticamente robustos tus resultados?"**
> Sí: intervalos de confianza al 95 % por bootstrap no paramétrico con 1000 *resamples*, varias semillas, *splits* oficiales, ablaciones y un análisis explícito de amenazas a la validez. Los intervalos son estrechos. Reporto incertidumbre en cada número; no hay cifras sueltas.

## Bloque E — Aplicabilidad

**11. "Comparar 1.920 USD con una celda de 150.000, ¿no es injusto? No incluyes robot ni integración."**
> Es una comparación de coste de I+D y reproducción, no de despliegue, y lo enmarco así. El mensaje no es "esto reemplaza una celda industrial", sino que investigar y reproducir el estado del arte ya no exige infraestructura de millones. Para un despliegue real habría que sumar robot, integración y seguridad; eso es ingeniería de producto, fuera del alcance de un TFM de validación.

**12. "¿Qué pasa con objetos fuera de la distribución de entrenamiento?"**
> La percepción (FoundationPose) generaliza zero-shot a objetos no vistos con su CAD: ese es su fuerte. La planificación, al entrenarse con un repertorio limitado, sí se degrada fuera de distribución —lo medí, una esfera rueda donde un cubo no— y lo resolví con randomización de dominio (forma y color), que recuperó la precisión. Lo tengo cuantificado.

## Bloque F — Preguntas estándar y esperables (prepáralas, caen casi seguro)

**13. "En una frase, ¿cuál es su aportación original?"**
> El puente matemático que conecta la percepción —la pose en SE(3) de un Transformer— con la planificación —una SDE de difusión—, integrando dos paradigmas que nunca se habían combinado para bin-picking, y demostrándolo de forma reproducible en hardware accesible.

**14. "¿Por qué eligió este tema?"**
> Porque el bin-picking concentra lo que más me interesa del máster: geometría de grupos de Lie, procesos estocásticos y deep learning, aplicados a un problema industrial sin resolver. Y porque me permitía unir matemática formal con un sistema que funciona de principio a fin, no solo teoría.

**15. "Es un TFM en grupo. ¿Qué hizo usted exactamente?"**
> Me responsabilicé de la estimación de pose con FoundationPose y su evaluación en BOP, y de toda la formalización matemática: SE(3)/SO(3), atención como operador geométrico, y la derivación del acoplamiento con las SDEs. La implementación de Diffusion Policy y el entorno CoppeliaSim fueron de mi compañero; la integración y la evaluación E2E, conjuntas. Cada parte es individualmente un TFM completo.

**16. "Su ciclo tarda ~6 s, pero la industria pide menos de 500 ms. ¿Sirve de verdad?"**
> Hoy no para línea de alta cadencia, y lo digo. Pero el cuello de botella está localizado: el 80 % es la percepción, optimizable con TensorRT o destilación; la planificación es el 2 %. El objetivo del TFM era viabilidad sin GPU dedicada (<10 s), no latencia industrial; esa optimización es ingeniería posterior bien acotada.

**17. "¿Por qué CoppeliaSim y no Isaac Sim o Gazebo?"**
> Por reproducibilidad y coste: CoppeliaSim Edu es gratuito, corre en una laptop sin GPU dedicada y tiene puente ZMQ y ROS 2. Isaac Sim exige GPU NVIDIA potente, lo que contradice mi tesis de accesibilidad. La arquitectura no depende del simulador.

**18. "¿Por qué YCB-Video y T-LESS?"**
> Son complementarios y estándar en BOP. T-LESS son piezas industriales sin textura, con simetrías y oclusiones: el caso difícil y el más relevante para bin-picking. YCB-Video aporta objetos con textura para medir generalización. Juntos cubren robustez sin textura y discriminación visual.

**19. "¿Por qué difusión y no un VAE o un GMM para la planificación?"**
> Porque la difusión modela distribuciones multimodales complejas con más fidelidad y estabilidad de entrenamiento que VAE o GMM, sin colapso de modos, y tiene un fundamento limpio como SDE inversa. Eso permite el acoplamiento formal con la geometría de la pose, que es mi aporte.

**20. "¿Qué fue lo más difícil?"**
> Diagnosticar por qué el refuerzo rompió el agarre. Tuve que aislar que el problema no era el RL en sí, sino un condicionamiento con poca señal discriminativa. Ese diagnóstico, no la primera solución, fue lo que destrabó la remontada al 84 %.

**21. "Si empezara de nuevo, ¿qué haría distinto?"**
> Entrenaría la política con demostraciones más diversas desde el principio, no solo trayectorias heurísticas, para atacar antes el límite de multimodalidad de H2. Y reservaría tiempo para una primera prueba sim-to-real, aunque fuera mínima.

**22. "¿Cómo garantiza que sus números se reproducen?"**
> Todo es trazable: cada cifra apunta a un commit y un RUN_CARD, con semillas fijas, lockfile de dependencias, imagen Docker, tests con integración continua y los JSON de resultados commiteados. Incluso publiqué el cálculo de intervalos de confianza como paquete en PyPI. Cualquiera puede recomputarlo.

**23. "¿Y piezas transparentes, muy reflectantes o deformables?"**
> Fuera de mi alcance declarado: trabajo con objetos rígidos y CAD conocido. Transparentes y reflectantes degradan la profundidad RGB-D (problema abierto en todo el área); deformables exigen otra representación. Lo dejo explícito como límite, no como algo resuelto.

## Bloque G — Curvas y trampas

**24. "¿No es esto solo juntar dos repos de GitHub?"**
> Si fuera solo ejecutar dos modelos en serie, sí. Pero el aporte es el acoplamiento formal: derivar cómo la pose en SE(3) condiciona la SDE, reconciliar las representaciones, y validarlo con protocolo BOP y honestidad estadística. La dificultad no está en correr cada modelo, sino en unirlos con un formalismo común y medir si la unión aporta.

**25. "Usted mismo dice que la difusión 'imita al heurístico'. ¿No se contradice con proponerla?"**
> No, porque distingo el resultado actual del marco. Con los datos que usé, imita: lo mido y lo reconozco. Lo que propongo es la formulación que permite ir más allá cuando haya demostraciones reales o flow-matching. Presento evidencia, no promesas: por eso H2 es parcial.

**26. "¿Cuál es el impacto real de su trabajo más allá de lo académico?"**
> Bajar la barrera de entrada. Demuestro que reproducir y experimentar con el estado del arte en manipulación ya no exige infraestructura de seis cifras, sino menos de dos mil dólares. Eso habilita a universidades y pymes de la región a investigar robótica avanzada, que antes era exclusiva de grandes laboratorios.

---
### Si te bloqueas en cualquiera
Frase puente honesta: *"Es una observación pertinente; lo abordo como limitación declarada en el capítulo de alcance, y mi decisión de diseño fue…"* → y reencuadras. Reconocer un límite con seguridad puntúa más que defender lo indefendible.
