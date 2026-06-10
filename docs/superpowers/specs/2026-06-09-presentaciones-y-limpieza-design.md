# Diseño: presentaciones (defensa + divulgativa) y limpieza pre-defensa

Fecha: 2026-06-09 · Estado: aprobado por el autor y ejecutado.

## Contexto

Revisión end-to-end pre-defensa del TFM (headline Iter 7c: P&P E2E 84 %, IK 100 %).
El pptx de defensa era correcto pero muy textual (3 imágenes en 18 slides); faltaba
una presentación para una charla divulgativa de robótica e IA (~45 min, audiencia
diversa); y el repo público mezclaba artefactos activos con legacy.

## Decisiones

1. **Defensa** (`Presentacion_Defensa_TFM.pptx`): mejorar in-place, sin tocar la
   estructura de 18 slides ni las cifras del guión ensayado. Figuras añadidas
   (todas pre-existentes y verificadas, o generadas desde cifras canónicas):
   - S2 frame de escena · S5 diagrama de arquitectura (hero fig, regenerada con
     «255 tests») · S8 comparativa oficial FP vs GDR-Net++ · S9 trayectorias 3D
     multimodales (exp9) · S10 chart latencia/cuello de botella · S12 video
     `reel_showcase.mp4` **embebido** + montaje 4 paneles · S16 chart de costo.
2. **Charla divulgativa** (`Presentacion_Robotica_IA.pptx`, 19 slides, generada
   por script): narrativa gancho → Moravec → cómo ve (pose 6-DoF) → cómo decide
   (difusión = misma IA que genera imágenes) → 2 demos en vivo (Gradio,
   CoppeliaSim) con plan B en video → historia honesta 60→28→84 % → costo →
   futuro. Videos embebidos: reel_showcase (S2), demo_v2 (S13), reel_resumen (S19).
   Notas del orador con timing en cada slide. Runbook de demos aparte
   (`Runbook_Demos_Charla.md` en el directorio TFM).
3. **Limpieza del repo** (PR #31): archivar (no borrar) a `archive/` los
   notebooks legacy y el video E2E v1, con manifest. Los scripts exp* y el
   notebook Colab NO se mueven (evidencia referenciada de las 13 exploraciones).
   `dashboard.py` y `gradio_demo.py` se conservan: el Gradio se usa en la charla;
   ninguno se muestra en la defensa (el video es la evidencia ante el tribunal).

## Generadores (reproducibilidad)

En `~/Documents/MATLAB/TFM/presentacion_assets/`:
`make_charts_defensa.py`, `make_fig_difusion_concepto.py`,
`make_pptx_divulgativa.py`, `patch_pptx_defensa.py` (+ frames extraídos con ffmpeg).
Backup del pptx original en `old_versions/Presentacion_Defensa_TFM.BACKUP_pre_visual_*.pptx`.
