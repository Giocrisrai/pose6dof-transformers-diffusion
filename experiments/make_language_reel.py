#!/usr/bin/env python3
"""Reel de demo del pick guiado por lenguaje (crescendo de instrucciones).

Por cada instrucción ejecuta run_language_pick (requiere CoppeliaSim), toma
los frames del pick, superpone la instrucción y el target elegido con
reel_overlay, y compila un MP4 único. Sin CoppeliaSim solo expone la lista
CRESCENDO (testeable).

Uso (CoppeliaSim en :23000):
    .venv/bin/python experiments/make_language_reel.py

Notas de adaptación a reel_overlay:
- normalize_frame(frame) devuelve un frame NUEVO (1280x720); no muta el original.
- draw_title_bar(frame, number, title) y draw_metrics(frame, lines) mutan
  el frame in-place Y lo devuelven. Se usa el valor de retorno por claridad.
- draw_metrics recibe lines = list[(texto: str, pasa_threshold: bool)]; el
  texto en verde si pasa_threshold=True, blanco si False.
- compile_mp4 se importa desde src.simulation.pick_sequence (mismo módulo que
  usa run_language_pick internamente), no desde run_pick_with_diffusion.
- Los frames de pick se nombran con 6 dígitos (%06d.png) según pick_sequence.
  El reel concatena todos los picks con índice global de 6 dígitos.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

# Crescendo: de simple a difícil (color → forma → relación espacial)
# Cada instrucción es un escalón en complejidad semántica para el demo.
CRESCENDO = [
    "pick the red cube",
    "dame el cubo azul",
    "the red sphere",
    "dame el cubo rojo de la izquierda",
    "pick the nearest red object",
]


def main() -> int:
    """Ejecuta el reel completo (requiere CoppeliaSim activo en :23000).

    Por cada instrucción del CRESCENDO:
      1. Llama run_language_pick para obtener el payload (grounding + pick).
      2. Lee los frames PNG generados en experiments/results/language_pick/frames/.
      3. Normaliza cada frame a 1280x720 con normalize_frame.
      4. Superpone la barra de título (número + instrucción) con draw_title_bar.
      5. Superpone el panel de métricas (target id + selección correcta) con draw_metrics.
      6. Escribe el frame procesado en experiments/results/language_reel/frames/.
    Finalmente compila todos los frames en un único language_reel.mp4.
    """
    import cv2

    from src.language import make_parser
    from src.simulation import reel_overlay
    from src.simulation.language_pick import run_language_pick
    from src.simulation.pick_sequence import compile_mp4

    out_dir = REPO / "experiments" / "results" / "language_reel"
    all_frames_dir = out_dir / "frames"
    all_frames_dir.mkdir(parents=True, exist_ok=True)
    pick_frames_dir = REPO / "experiments" / "results" / "language_pick" / "frames"

    idx = 0
    for n, instruction in enumerate(CRESCENDO, 1):
        # Parsear la instrucción para derivar color/forma del target y construir
        # una escena que CONTENGA el objeto descrito (demo honesto).
        instr_parsed = make_parser("deterministic").parse(instruction)
        tcolor = instr_parsed.target.color or "red"
        tshape = instr_parsed.target.shape or "cube"
        out = run_language_pick(
            instruction,
            scene="clutter",
            parser_backend="deterministic",
            render=False,
            n_objects=3,
            with_shapes=True,
            seed=n,
            target_color=tcolor,
            target_shape=tshape,
        )
        tgt = out["grounding"]["target_obj_id"]
        ok = tgt is not None and bool(out.get("selection_correct"))

        # Si no hubo match en el grounding, no reutilizar frames del pick anterior.
        if tgt is None:
            # Sin match: omitir frames para este paso del crescendo.
            continue

        # Etiqueta del target para el panel de métricas
        tgt_label = f"target #{tgt}" if tgt is not None else "sin match"

        for fp in sorted(pick_frames_dir.glob("*.png")):
            frame = cv2.imread(str(fp))
            if frame is None:
                continue
            # normalize_frame devuelve un frame NUEVO escalado a 1280x720
            frame = reel_overlay.normalize_frame(frame)
            # draw_title_bar: number (str), title (str truncado a 48 chars internamente)
            reel_overlay.draw_title_bar(
                frame,
                str(n),
                reel_overlay.truncate(instruction, 48),
            )
            # draw_metrics: lines = list[(texto, pasa_threshold: bool)]
            reel_overlay.draw_metrics(frame, [(tgt_label, ok)])
            # Nombre del frame: 6 dígitos (formato esperado por compile_mp4)
            cv2.imwrite(str(all_frames_dir / f"{idx:06d}.png"), frame)
            idx += 1

    mp4 = compile_mp4(all_frames_dir, out_dir / "language_reel.mp4", fps=25)
    print(f"reel: {mp4}")
    print(f"total frames compilados: {idx}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
