# Guion de defensa — narrativa (TFM bin-picking 6-DoF)
**Duración:** ~20 min · **Autor:** Giocrisrai Godoy · Máster en Ingeniería Matemática y Computación

## La columna narrativa (memorízala: es tu hilo)
> "Lo que cualquier niño hace sin pensar —recoger una pieza del suelo— sigue siendo un problema abierto para un robot. **Aposté por unir dos avances de IA que nadie había combinado** (la familia de ChatGPT para *ver* y la de los generadores de imágenes para *moverse*), lo validé con rigor y honestidad —incluido un fracaso del que aprendí— y demostré que se puede hacer en una laptop, no en un laboratorio de millones."

Todo lo demás cuelga de esa frase. Cuando te pierdas, vuelve a ella.

**Las 3 emociones que quieres provocar:** *curiosidad* (qué difícil es lo fácil) → *interés* (la apuesta original) → *confianza* (lo hice con rigor y lo digo todo, hasta lo que falló).

---

## Apartado 1 — El problema y la apuesta

### S1 · Portada (0:30)
> "Buenos días. Soy Giocrisrai Godoy y defiendo mi TFM: la integración de FoundationPose y Diffusion Policy para *bin-picking* robótico. En los próximos veinte minutos quiero convencerlos de una idea: que unir dos paradigmas de IA que vivían separados resuelve un problema que la industria lleva décadas persiguiendo."

*(Entrega: respira. Mira al tribunal, no a la slide. Esta frase fija el rumbo.)*

### S2 · El problema (1:00)
> "El *bin-picking* es elegir una pieza de un montón desordenado, agarrarla y depositarla. Suena trivial, pero esconde la **paradoja de Moravec**: lo que la evolución tardó millones de años en pulir —ver y manipular— es lo más difícil de replicar. Ganarle al campeón mundial de ajedrez se resolvió en 1997; recoger una pieza del suelo **sigue siendo investigación**. Y no es un problema de nicho: cada paquete de e-commerce pasó por unas manos haciendo exactamente esto."

*(Transición:)* "Para que un robot lo haga, primero tiene que responder dos preguntas."

### S3 · Objetivo e hipótesis (1:30)
> "Dónde está la pieza y cómo agarrarla. Mi objetivo fue integrar ambas respuestas en un único *pipeline*, validado de principio a fin. Y me comprometí con tres hipótesis **falsables**, con números, no con adjetivos: **H1**, mejorar la precisión de pose en al menos 3 puntos sobre el referente; **H2**, generar agarres multimodales rápidos; y **H3**, que todo corra sin GPU dedicada, por debajo de 10 segundos por ciclo. Les adelanto: dos se cumplen y una se cumple solo en parte. Y esa parte es, quizá, lo más interesante de contar."

*(Entrega: el "se cumple solo en parte" dícelo con calma, no como disculpa. Es madurez científica.)*

### S4 · Estado del arte y la brecha (1:30)
> "La percepción de pose pasó de métodos clásicos a Transformers, con FoundationPose marcando el estado del arte en 2024. La planificación de agarre encontró en los modelos de difusión una forma natural de capturar que *hay muchas maneras válidas de agarrar*. Dos líneas potentes… que nadie había unido con un marco matemático común para *bin-picking*. **Esa es mi brecha, y mi contribución.** No propongo una pieza nueva: propongo el puente entre dos mundos."

---

## Apartado 2 — Qué construí

### S5 · El pipeline integrado (1:30)
> "Así funciona, de izquierda a derecha: la cámara RGB-D entra a FoundationPose, que estima la pose en SE(3); esa pose **condiciona** a la Diffusion Policy, que genera dieciséis pasos de trayectoria; y cada paso se ejecuta por cinemática inversa en el simulador. Tres familias de técnicas, un solo flujo."

### S6 · El fundamento matemático (2:00)
> "Y aquí está el corazón formal, el que justifica que esto es un trabajo de Ingeniería Matemática. Las rotaciones viven en SO(3), con la representación 6D continua que evita el *gimbal lock*. La pose completa es un elemento de SE(3), un grupo de Lie. La percepción es atención *scaled dot-product*. Y la planificación es *score matching* más dinámica de Langevin: una ecuación diferencial estocástica que, partiendo de ruido, recupera una trayectoria válida. **El puente entre percepción y acción es geométrico: la pose en SE(3) condiciona la SDE que genera el movimiento.** Esa conexión es lo original."

*(Entrega: este es TU terreno. Habla despacio, con seguridad. Si el tribunal te va a apretar, será aquí; respóndeles con gusto.)*

---

## Apartado 3 — ¿Funcionó? (resultados con honestidad)

### S7 · Metodología (1:30)
> "Antes de los números, cómo los obtuve, porque el rigor es parte del aporte. Evalué sobre YCB-Video y T-LESS —piezas sin textura, el caso difícil— con las métricas oficiales que tratan bien las simetrías, *splits* oficiales, varias semillas, e intervalos de confianza al 95 % por *bootstrap*. Nada de un número suelto: todo con su incertidumbre."

### S8 · H1 — Precisión (1:30)
> "FoundationPose alcanza un AUC ADD-S de 0.908 en YCB-Video y 0.957 en T-LESS, con intervalos estrechos, y mejora al referente GDR-Net++ en 3.0 y 3.6 puntos: por encima del umbral que me fijé. **H1 se acepta.** Reproduje el estado del arte en mi propio hardware, que ya es un resultado."

### S9 · H2 — Planificación multimodal (1:30) · *el momento honesto*
> "Y aquí soy transparente. La calidad de las trayectorias cumple, pero el muestreo tarda 118 milisegundos, por encima de los 50 que pedí, y detecto dos modos en vez de tres. Por eso **H2 se acepta solo en parte**. ¿La causa? La política aprendió imitando un planificador, no demostraciones reales, así que copia bien pero explora poco. Lo importante: lo diagnostiqué, no lo escondí. Y al sustituir el condicionamiento por un encoder visual, el agarre subió del 36 % al 78 %. La ciencia honesta no es la que no falla; es la que entiende por qué."

*(Entrega: este slide te hace ganar el tribunal. Mira a los ojos. No te disculpes: explica.)*

### S10 · H3 — Viabilidad sin GPU dedicada (1:30)
> "El ciclo completo corre en 6.3 y 6.7 segundos en el percentil 95, holgadamente bajo los 10. El cuello de botella es la percepción —el 80 % del tiempo—; la difusión es apenas el 2 %. Y todo sobre una laptop más una GPU de alquiler. **H3 se acepta.**"

### S11 · La historia del fracaso y la remontada (2:00) · *el clímax*
> "Esta figura es la que más me gusta, porque cuenta un fracaso. Mi primer sistema completo lograba un 60 % de ciclos exitosos. Quise mejorarlo con aprendizaje por refuerzo… y bajó al 28 %: optimizó depositar y **olvidó cómo agarrar**. Se llama olvido catastrófico y le pasa a los mejores laboratorios. No me rendí: diagnostiqué, apliqué un currículo —primero agarrar, luego depositar—, generé ocho opciones y ejecuté la mejor, y arreglé un error de geometría. Resultado: **84 %**, por encima del sistema original. En IA real no se avanza por arte de magia; se avanza diagnosticando."

### S12 · Demo (1:00)
> "Veámoslo. *(Reproducir reel.)* Un ciclo completo en simulación: estima la pose, genera la trayectoria, aproxima, agarra, eleva y deposita."

*(Entrega: no hables encima del video; deja que respire. Solo señala las fases.)*

---

## Apartado 4 — Qué significa y qué sigue

### S13 · Límites (1:00) · *credibilidad*
> "Y digo qué NO hice. El agarre es *snap+attach*: valida la cadena percepción-planificación-ejecución, no la física de fricción. Es simulación, sin brazo real. Y FoundationPose tiene licencia no comercial. Lo declaro porque conocer los límites es parte del resultado."

### S14–S15 · Aportes (2:00)
> "Cuatro aportes. **Uno**, metodológico: la integración matemática formal de percepción Transformer y planificación por difusión sobre SE(3). **Dos**, reproducibilidad: el estado del arte recomputado en hardware accesible, con tests, integración continua y un paquete publicado en PyPI. **Tres**, rigor por encima del estándar del área —intervalos de confianza, ablaciones, amenazas a la validez—. **Cuatro**, validación de principio a fin con evidencia visual y repositorio público."

### S16 · Coste y democratización (1:00)
> "El mercado de *bin-picking* crece hacia los 7 mil millones; las soluciones cuestan entre 15 y 150 mil dólares. Esto corre en menos de 2 mil. No compite con la industria: demuestra que investigar robótica avanzada ya está al alcance de una universidad o una pyme."

### S17 · Trabajo futuro (1:00)
> "Tres líneas claras: validar el agarre físico real; acelerar la difusión con *flow-matching* a uno o dos pasos; y transferir a un robot real. Es la misma receta —percepción Transformer más acción por difusión— que mueve a los humanoides de hoy."

### S18 · Cierre (0:30)
> "En síntesis: un *pipeline* que une matemáticamente percepción y planificación, validado con rigor y honestidad —dos hipótesis aceptadas, una en parte—, reproducible por menos de lo que cuesta una moto, y todo público. Gracias. Quedo atento a sus preguntas."

*(Entrega: la última frase, más lenta. Sonríe. Has contado una historia completa.)*

---

## Cómo manejarlo con confianza (lo que te costó)
- **No memorices palabra por palabra.** Memoriza la *columna narrativa* y la *idea de cada slide* (una frase). El resto sale solo.
- **Transiciones = pegamento.** Cada slide termina apuntando al siguiente ("para eso, primero…", "¿y funcionó?…"). Eso da sensación de relato, no de lista.
- **El fracaso (S11) y la honestidad (S9, S13) son tus mejores cartas**, no debilidades. Dícelas mirando al tribunal, sin disculparte.
- **Pausas.** Tras cada número fuerte (0.908, 84 %), calla un segundo. Deja que pese.
- **Tu terreno fuerte es S6 (matemática).** Si te preguntan, alégrate: es donde más sólido estás.
- **Cronómetro:** ~20 min. Si vas justo, recorta S14-S15 (aportes) y S17 (futuro), nunca S6 ni S9/S11.
