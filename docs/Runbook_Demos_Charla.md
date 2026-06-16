# Runbook — Demos en vivo para la charla de Robótica e IA

Presentación: `Presentacion_Robotica_IA.pptx` (19 slides, ~45-47 min + Q&A).
Los 3 videos están **embebidos** en el pptx (slides 2, 13 y 19) — funcionan sin internet y sin archivos externos.
Respaldo descargable (por si hay que reponer un MP4): release [`v0.1.0`](https://github.com/Giocrisrai/pose6dof-transformers-diffusion/releases/tag/v0.1.0) → `reel_showcase.mp4`, `reel_resumen.mp4`, `demo_v7c.mp4`.

---

## La noche anterior

- [ ] Cargar la laptop (las demos usan MPS/CPU intensivo).
- [ ] Probar el pptx completo en PowerPoint: los 3 videos reproducen con clic.
- [ ] Ensayar UNA vez la demo de CoppeliaSim de punta a punta (paso a paso abajo).
- [ ] Verificar que el repo está en `main` actualizado: `cd ~/Documents/MATLAB/TFM/repo_tfm && git pull`.

## 1 hora antes (en la sala)

- [ ] Conectar proyector y verificar que se ve la slide 1 y que el AUDIO no hace falta (los videos no dependen de audio).
- [ ] Activar **modo No Molestar** (Centro de Control → Concentración) — evita notificaciones en pantalla.
- [ ] Abrir TODO lo de las demos ANTES de empezar (ver abajo) y dejarlo minimizado.
- [ ] Plan B verificado: si nada funciona, la charla completa se sostiene solo con el pptx.

---

## DEMO 1 — Trayectorias en vivo (slide 10, ~8 min)

**Qué es:** interfaz web minimalista (`scripts/demo_charla.py`, hecha para proyector: letras grandes, fondo oscuro); el público propone la posición de la pieza y el modelo de difusión genera N trayectorias en vivo. Carga un ejemplo sola al abrir — nunca se ve en blanco.

> El demo Gradio anterior (`gradio_demo.py`) queda deprecado para charlas: demasiadas secciones y tipografía pequeña.

**Arrancar ANTES de la charla** (~30 s; el modelo carga al inicio):

```bash
cd ~/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python scripts/demo_charla.py
# → abrir http://127.0.0.1:7860 y dejar la pestaña lista (ya muestra un ejemplo)
```

**Guión de la demo:**
1. Pedir a 2-3 personas del público: "díganme dónde está la pieza" (sliders x, y, z) — o un clic en los presets (Centro / Izquierda / Esquina).
2. «✨ Generar trayectorias» → ~4 s (2 s de inferencia + 2 s de preparación del visor).
3. Pestaña **🌀 Los caminos posibles**: estrella amarilla = pieza; cada línea cian = un camino DISTINTO; los puntos verdes (finales) coinciden en ~1-2 cm — "diversidad con propósito".
4. Pestaña **🎬 El robot en acción (3D interactivo)**: ciclo didáctico completo — el robot VE la pieza (anillo dorado), IMAGINA N caminos (nube de difusión dibujándose), ELIGE el mejor (best-of-N), lo EJECUTA y DEPOSITA la pieza en la bandeja **con caída por gravedad y rebote**. Cámaras conmutables (general / perfil / cenital / seguir pinza) y órbita libre con el mouse.
5. **El momento WOW — perturbación**: durante la ejecución se enciende el botón rojo **🫳 perturbar la pieza**. Pide a alguien del público que lo pulse: la pieza se desliza por la mesa, el sistema muestra "⚡ re-planificando" con una nube NUEVA de difusión (generada de verdad por el modelo hacia la nueva posición) y el brazo la agarra igual. Frase: "esto es lo que diferencia a una solución robusta: el entorno cambia y el sistema se adapta".
6. Repetir con otra posición del público: cada clic da caminos y animación nuevos.

**Si falla:** decir "les muestro el resultado grabado" → la slide 9 (nube de 50 trayectorias) cuenta lo mismo. Seguir sin drama.

---

## DEMO 2 — CoppeliaSim en vivo (slide 12, ~8 min)

**Qué es:** un ciclo completo de pick-and-place (Iter 7c: best-of-8 + fix IK) corriendo en el simulador con cámara cinematográfica.

**Preparar ANTES de la charla:**

```bash
# 1. Abrir CoppeliaSim (la app) y dejarla corriendo en el puerto ZMQ por defecto (23000).
#    No hace falta cargar escena a mano: se carga sola (data/scenes/bin_base.ttt).
# 2. Acomodar la vista de la cámara en CoppeliaSim (zoom al robot y la mesa).
```

**Opción A (recomendada) — desde el mismo demo web:** abrir la pestaña
**"🤖 CoppeliaSim — ejecución real"** (muestra 🟢/🔴 si el simulador está detectado),
pedir al público la posición de la pieza sobre la mesa (2 sliders + rotación 0/45/90°)
y pulsar «Ejecutar en el simulador». Si quedó una simulación corriendo de un intento
anterior, el demo la detiene solo (no hace falta tocar CoppeliaSim).

**Selector de forma/color + política (🔬 el ANTES/DESPUÉS — el mejor momento de la demo):**
- **Cubo rojo + política original** = el caso confiable de siempre. Úsalo para el "wow" inicial.
- **El guión antes/después** (si hay tiempo, ~2 min extra, validado 12-jun):
  1. Esfera verde + política **original (Iter 7c)** → se degrada (agarre 2-6 cm, la pieza
     sale empujada). El demo lo explica solo: "la IA es tan buena como su distribución".
  2. Cambiar SOLO el selector de política a **robusta (Iter 8)** y repetir → la agarra
     a **0.8 cm**. Frase preparada: "la reentrenamos anoche con formas y colores
     aleatorios — 5 minutos de datos y 46 segundos de entrenamiento".
- Validado 12-jun: original+esfera 🔬 2.1 cm/pieza empujada · robusta+esfera ✅ agarre
  0.8 cm (la esfera puede rodar AL DEPOSITAR — el demo explica que es física, no percepción)
  · robusta+cubo rojo ✅ 28 s (1.8 cm). Con la robusta el cubo rojo es AÚN mejor que con
  la original (eval pareada: E2E 92 % vs 68 %).
- Si quieres éxito garantizado de punta a punta con piezas raras: usa **cubo azul/amarillo
  + robusta** (los cubos no ruedan al depositar; esferas/cilindros a veces sí).

**Selector de piezas distractoras 🎯 (el cierre del crescendo — validado 12-jun):**
- Pon **distractoras = 2** y política **clutter (Iter 9)**: aparecen 2 piezas extra de
  forma/color aleatorios y el robot va SOLO por la pieza indicada. Frase preparada:
  "ahora la mesa tiene tres piezas — le pedimos LA esfera verde, y tiene que elegirla".
- Caso garantizado E2E: **cubo rojo + clutter + 2 distractoras** (validado: 1.7 cm,
  depósito 3.5 cm, 44 s). Con esfera/cilindro el agarre es ~1-2 cm pero puede rodar al depositar.
- El antes/después del clutter (opcional): misma escena con política **robusta (Iter 8)**
  se degrada (eval: grasp 84 %, 2.8 cm) — la de clutter lo recupera (100 %, 2.1 cm).
- Crescendo completo de la demo si hay tiempo: ① cubo rojo + original → wow básico;
  ② esfera verde + original → se degrada; ③ misma esfera + robusta → la agarra;
  ④ 2 distractoras + clutter → elige la indicada entre tres. Cuatro clics, una historia:
  "la IA aprende a ignorar lo que no importa".

**🛡️ "Verificar antes de actuar" (el momento más sofisticado — validado 14-jun):**
- Aparece SOLO cuando hay distractoras (≥1): el demo, en el resultado, dice cuántos de
  los 8 caminos imaginados **descartó por pasar muy cerca de otra pieza**. Es la prueba
  de que el robot no choca a ciegas.
- **El gancho conceptual** (decirlo mientras corre): "fíjense que NO mueve el brazo para
  pensar — imagina 8 trayectorias como números dentro de la red, descarta las que rozarían
  otra pieza, y solo entonces ejecuta la mejor de las seguras. Verifica antes de actuar,
  como un humano que mira antes de meter la mano". Frase de cierre: "esto es lo que separa
  una demo de un sistema industrial: no basta agarrar, hay que no romper el resto".
- Pregunta del público que esto responde ("¿y si choca las otras piezas?"): medido —
  sin verificación el brazo empuja un cubo en 18-24 % de los intentos; con verificación
  baja a 0 % (política robusta) / 18 % (clutter), y el peor empujón cae de 13,6 cm a 0,2 cm.
- **Honestidad preparada** (si preguntan por los límites): "verifica el camino de la PUNTA,
  no del cuerpo entero del brazo, y las piezas redondas ruedan al tocarse — la seguridad
  total necesita planificación de colisiones de cuerpo entero (RRT/OMPL). Esta es la versión
  barata que ya recupera la mayoría". Nunca vender más de lo medido.
- Validado en vivo 14-jun: clutter + cubo rojo + 2 distractoras → descartó 6 de 8 caminos,
  agarró la pieza indicada entre 3, depósito 5,3 cm, E2E ✅.
La página informa el progreso y al final las métricas. **Validado: pick completo
en ~32 s** (agarre 2.2 cm, IK ✓). Interactivo de verdad: la pieza va donde el público diga.

**Opción B — desde terminal** (clip con cámara cinematográfica, más vistoso pero pose fija):

```bash
cd ~/Documents/MATLAB/TFM/repo_tfm
.venv/bin/python experiments/make_reel_clip_iter7c.py --pose-index 49 --torch-seed 3
```

**Durante la demo:**
- Dar Enter y narrar las fases mientras corre: "está viendo la escena → generó 8 trayectorias y eligió la mejor → ejecuta → deposita".
- **Timing real validado (9-jun)**: ~1,5 min de punta a punta (conexión + carga de escena + animación); la animación visible dura ~1 min. Resultado del run de validación: grasp 4,1 cm plausible, depósito 2,4 cm, IK convergió. ✅
- La pose 49 con seed 3 es la corrida "hero" verificada. Si quieres variar, `--pose-index 30` es la otra corrida buena conocida.
- IMPORTANTE: mostrar la ventana de CoppeliaSim en el proyector (no la terminal).
- Nota: el script regenera `demo_v7c.mp4` al terminar (inofensivo; es su modo de guardar evidencia).

**Si falla** (CoppeliaSim no conecta, lag, etc.): pasar a la **slide 13** — es exactamente este sistema grabado con telemetría. Frase preparada: "la demo en vivo se puso tímida; les muestro la grabación de esta misma demo".

---

## Timing de referencia (47 min)

| Bloque | Slides | Tiempo |
|---|---|---|
| Gancho + video | 1-2 | 0:00–3:00 |
| Por qué es difícil | 3-4 | 3:00–9:00 |
| Cómo ve (pose 6-DoF, FoundationPose) | 5-6 | 9:00–17:00 |
| Cómo decide (difusión, multimodal) | 7-9 | 17:00–23:00 |
| **DEMO 1 Gradio** | 10 | 23:00–31:00 |
| Pipeline completo | 11 | 31:00–33:00 |
| **DEMO 2 CoppeliaSim** (+plan B) | 12-13 | 33:00–40:00 |
| Historia honesta (60→28→84 %) | 14-15 | 40:00–44:00 |
| Costo + futuro + cierre | 16-18 | 44:00–47:00 |
| Q&A (reel en loop) | 19 | abierto |

Si vas atrasado: recortar la slide 11 (el pipeline ya se entendió con las demos) y la 17 (futuro) a una frase cada una.

---

## Q&A — respuestas preparadas

- **"¿Funciona con un robot real?"** — Todo está validado en simulación; la transferencia a robot físico (cámara RealSense, fricción real) es el siguiente paso natural y el pipeline está diseñado para eso.
- **"¿Por qué simulación y no robot real?"** — Costo y rigor: en simulación se pueden correr 50 evaluaciones idénticas y reproducibles por iteración; un robot físico cuesta 15.000+ USD.
- **"¿El robot 'agarra' de verdad?"** — En el simulador el agarre es cinemático (snap+attach, estándar en sims); lo que se valida es la cadena completa ver→decidir→mover, no la fricción.
- **"¿Cuánto tardó?"** — ~5 meses de trabajo de tesis, 7 iteraciones documentadas del sistema de planificación.
- **"¿Qué es lo más difícil?"** — Que mejorar una parte no rompa otra (el RL que subió el depósito y destruyó el agarre). Diagnóstico sistemático > intuición.
