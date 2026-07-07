"""Superpone un HUD sobre los frames cinematográficos del pick y compila MP4+GIF.
Interfaz visual de nuestra simulación: estado (pieza, pose, fase) sobre la escena."""
import subprocess, json
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

EXP = Path(__file__).resolve().parent
SRC = EXP/"cine_frames"; HUD = EXP/"hud_frames"; HUD.mkdir(exist_ok=True)
FIGS = EXP/"figs"
FONT = "/System/Library/Fonts/Supplemental/Arial.ttf"
FB = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"

# métricas reales del experimento
rep = json.loads((EXP/"e2e_report.json").read_text())
POSE = f"{rep['pose']['t_err_mm']} mm · {rep['pose']['R_err_deg']}°"

PHASES = [(0.06,"Preparación"),(0.28,"Aproximación (IK)"),(0.42,"Descenso"),
          (0.50,"Agarre · snap+attach"),(0.68,"Elevación"),(0.90,"Depósito"),(1.01,"Liberación")]
def phase(f):
    for thr,name in PHASES:
        if f < thr: return name
    return "Fin"

def font(sz, bold=False):
    try: return ImageFont.truetype(FB if bold else FONT, sz)
    except Exception: return ImageFont.load_default()

frames = sorted(SRC.glob("*.png")); N = len(frames)
f_title=font(30,True); f_lbl=font(24); f_ph=font(28,True)
for i,fp in enumerate(frames):
    im = Image.open(fp).convert("RGBA"); W,H = im.size
    ov = Image.new("RGBA",(W,H),(0,0,0,0)); dr = ImageDraw.Draw(ov)
    # barra superior
    dr.rectangle([0,0,W,58],fill=(15,23,42,190))
    # panel inferior izquierdo
    dr.rounded_rectangle([24,H-150,470,H-30],radius=14,fill=(15,23,42,175))
    # barra de progreso
    dr.rectangle([0,H-6,W,H],fill=(30,41,59,160))
    dr.rectangle([0,H-6,int(W*(i+1)/N),H],fill=(45,133,90,235))
    base = Image.alpha_composite(im,ov); dr = ImageDraw.Draw(base)
    dr.text((26,14),"TFM · Bin picking guiado por text-to-CAD",font=f_title,fill=(255,255,255,255))
    dr.text((W-360,20),"CoppeliaSim · UR5e · Apple M1",font=f_lbl,fill=(148,163,184,255))
    dr.text((40,H-140),"Pieza generada:  escuadra en L (60×40×45 mm)",font=f_lbl,fill=(226,232,240,255))
    dr.text((40,H-108),f"Pose 6-DoF (depth real):  {POSE}",font=f_lbl,fill=(226,232,240,255))
    dr.text((40,H-72),f"Fase:  {phase(i/N)}",font=f_ph,fill=(96,205,138,255))
    base.convert("RGB").save(HUD/f"{i:04d}.png")

# compilar MP4 (60 fps) y GIF (subsampleado)
mp4 = FIGS/"cine_pick_hud.mp4"
subprocess.run(["ffmpeg","-y","-framerate","60","-i",str(HUD/"%04d.png"),
    "-vf","scale=1280:-2","-c:v","libx264","-pix_fmt","yuv420p","-crf","23",str(mp4)],
    check=True, capture_output=True)
gif = FIGS/"cine_pick_hud.gif"
subprocess.run(["ffmpeg","-y","-framerate","60","-i",str(HUD/"%04d.png"),
    "-vf","fps=20,scale=720:-1:flags=lanczos","-loop","0",str(gif)],
    check=True, capture_output=True)
import shutil; shutil.rmtree(HUD)
print("MP4:", mp4.name, round(mp4.stat().st_size/1e6,1),"MB")
print("GIF:", gif.name, round(gif.stat().st_size/1e6,1),"MB")
