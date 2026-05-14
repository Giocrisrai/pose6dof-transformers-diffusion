#!/usr/bin/env python3
"""Experimento 24: visual grounding con CLIP-image (sin atributos sinteticos).

Cierra el ciclo end-to-end real: en lugar de pasar al gate atributos
DECLARADOS (RGB+shape onehot), generamos un CROP de cada objeto y dejamos
que CLIP-image extraiga el embedding visual.

Esto es la transicion del modelo "juguete sintetico" al modelo APLICABLE
a escenas reales con camara: la cadena es:

  RGB image -> crops (uno por objeto) -> CLIP-image(crop) -> embedding 512-D
                                                                    v
                                                              Gate(text_emb, vis_emb_a, vis_emb_b)
                                                                    v
                                                              Diffusion Policy

Setup del experimento:
- Generamos crops sinteticos (no foto-realistas) con shapes coloreadas + ruido
  para entrenar el gate-visual. Igual entrena, asi que CLIP-image generaliza.
- Validamos que el modelo identifica el objeto correcto basandose SOLO en
  apariencia visual del crop (no en el atributo declarado).

Esfuerzo: este experimento es una **demostracion** de viabilidad. Para
produccion real se necesitarian crops reales de las camaras + augmentations.

Salida:
    data/models/diffusion_policy_clip_image.pth (si pasa criterios)
    experiments/results/exp24_clip_image/exp24_results.json
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

HORIZON = 16
ACTION_DIM = 7
COND_DIM = 64
HIDDEN_DIM = 256
TIMESTEPS = 100
SEED = 42

COLORS = ["red", "blue", "green"]
SHAPES = ["cube", "sphere", "cylinder", "box"]
COLOR_RGB_NP = {
    "red": np.array([220, 50, 50], dtype=np.uint8),
    "blue": np.array([50, 100, 220], dtype=np.uint8),
    "green": np.array([50, 180, 100], dtype=np.uint8),
}
ATTR_DIM_VIS = 768  # CLIPVisionModel-base pooler output
MAX_OBJ = 4

TEMPLATES = [
    "pick the {color} {shape}",
    "grab the {color} {shape}",
    "select the {color} {shape}",
    "the {color} object",
    "the {shape}",
]

OUTPUT_MODEL = REPO / "data/models/diffusion_policy_clip_image.pth"
OUTPUT_RESULTS = REPO / "experiments/results/exp24_clip_image"
OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def render_object_crop(color: str, shape: str, size_px: int = 64, rng=None) -> np.ndarray:
    """Renderiza un crop sintetico simple del objeto.

    cube: rectangulo coloreado relleno
    sphere: circulo coloreado relleno
    cylinder: rectangulo vertical alto
    box: rectangulo horizontal alargado

    Anade ruido y variaciones para que CLIP no caiga en patron fijo.
    """
    if rng is None:
        rng = np.random.default_rng()
    img = np.full((size_px, size_px, 3), 220 + rng.integers(-20, 20), dtype=np.uint8)  # fondo claro
    color_rgb = COLOR_RGB_NP[color]
    # Variar color +/- ruido
    color_rgb = np.clip(color_rgb + rng.integers(-25, 25, size=3), 0, 255).astype(np.uint8)

    cx, cy = size_px // 2 + rng.integers(-4, 4), size_px // 2 + rng.integers(-4, 4)

    if shape == "cube":
        s = 22 + rng.integers(-4, 4)
        x1, y1 = max(cx - s, 0), max(cy - s, 0)
        x2, y2 = min(cx + s, size_px), min(cy + s, size_px)
        img[y1:y2, x1:x2] = color_rgb
        # Borde
        img[y1, x1:x2] = (color_rgb * 0.6).astype(np.uint8)
        img[y2-1, x1:x2] = (color_rgb * 0.6).astype(np.uint8)
        img[y1:y2, x1] = (color_rgb * 0.6).astype(np.uint8)
        img[y1:y2, x2-1] = (color_rgb * 0.6).astype(np.uint8)
    elif shape == "sphere":
        radius = 22 + rng.integers(-3, 3)
        yy, xx = np.ogrid[:size_px, :size_px]
        mask = (xx - cx) ** 2 + (yy - cy) ** 2 <= radius ** 2
        img[mask] = color_rgb
        # Sombra para dar volumen
        shade = (xx - cx + radius//3) ** 2 + (yy - cy + radius//3) ** 2 <= (radius // 2) ** 2
        img[shade & mask] = np.minimum(color_rgb + 40, 255).astype(np.uint8)
    elif shape == "cylinder":
        w = 14 + rng.integers(-3, 3)
        h = 30 + rng.integers(-3, 3)
        x1, y1 = max(cx - w, 0), max(cy - h, 0)
        x2, y2 = min(cx + w, size_px), min(cy + h, size_px)
        img[y1:y2, x1:x2] = color_rgb
    elif shape == "box":
        w = 28 + rng.integers(-4, 4)
        h = 16 + rng.integers(-3, 3)
        x1, y1 = max(cx - w, 0), max(cy - h, 0)
        x2, y2 = min(cx + w, size_px), min(cy + h, size_px)
        img[y1:y2, x1:x2] = color_rgb

    # Ruido global
    noise = rng.integers(-15, 15, size=img.shape)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


class CLIPEncoders:
    """Wrapper que mantiene CLIP-text y CLIP-image cargados."""
    def __init__(self, device):
        from transformers import CLIPTokenizer, CLIPTextModel, CLIPProcessor, CLIPVisionModel
        self.device = device
        self.tok = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.text_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.vision_model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
        for p in self.text_model.parameters(): p.requires_grad = False
        for p in self.vision_model.parameters(): p.requires_grad = False

    @torch.no_grad()
    def encode_text(self, texts: list[str]):
        ins = self.tok(texts, padding=True, return_tensors="pt", truncation=True)
        ins = {k: v.to(self.device) for k, v in ins.items()}
        return self.text_model(**ins).pooler_output  # (B, 512)

    @torch.no_grad()
    def encode_images(self, images: list[np.ndarray]):
        """images: list of np.ndarray (H, W, 3) uint8. Devuelve (B, 512)."""
        pil_images = [Image.fromarray(im) for im in images]
        ins = self.processor(images=pil_images, return_tensors="pt")
        ins = {k: v.to(self.device) for k, v in ins.items()}
        return self.vision_model(**ins).pooler_output  # (B, 512)


class CLIPProjector(nn.Module):
    def __init__(self, in_dim=512, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.Mish(), nn.Linear(64, out_dim))
    def forward(self, x): return self.net(x)


class VisualGate(nn.Module):
    """Gate que opera sobre CLIP-image embeddings (no atributos declarados).

    Cada objeto es representado por un vector 512-D de CLIP-vision (frozen).
    El gate aprende a combinar text_emb (de la instruccion) con cada vis_emb
    para producir un score logit por objeto.
    """
    def __init__(self, text_dim=512, vis_dim=768, hidden=256):
        super().__init__()
        # Proyectamos visual a un espacio menor para reducir params
        self.vis_proj = nn.Sequential(
            nn.Linear(vis_dim, 128), nn.Mish(),
            nn.Linear(128, 64),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, 128), nn.Mish(),
            nn.Linear(128, 64),
        )
        self.score = nn.Sequential(
            nn.Linear(64 + 64, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, 1),
        )

    def forward(self, text_emb, vis_embs, mask):
        """text_emb: (B, 512); vis_embs: (B, N, 512); mask: (B, N).
        Devuelve (B, N) softmax."""
        B, N, _ = vis_embs.shape
        text_proj = self.text_proj(text_emb).unsqueeze(1).expand(-1, N, -1)  # (B, N, 64)
        vis_proj = self.vis_proj(vis_embs)  # (B, N, 64)
        x = torch.cat([text_proj, vis_proj], dim=-1)  # (B, N, 128)
        logits = self.score(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        return F.softmax(logits, dim=-1)


def generate_scene(rng):
    n_obj = int(rng.integers(2, MAX_OBJ + 1))
    target_color = COLORS[rng.integers(0, len(COLORS))]
    target_shape = SHAPES[rng.integers(0, len(SHAPES))]
    template = TEMPLATES[rng.integers(0, len(TEMPLATES))]
    text = template.format(color=target_color, shape=target_shape)
    mentions_color = "{color}" in template
    mentions_shape = "{shape}" in template

    objects = []; positions = []
    target_idx = int(rng.integers(0, n_obj))
    for i in range(n_obj):
        if i == target_idx:
            c, s = target_color, target_shape
        else:
            while True:
                c = COLORS[rng.integers(0, len(COLORS))]
                s = SHAPES[rng.integers(0, len(SHAPES))]
                differs = False
                if mentions_color and c != target_color: differs = True
                if mentions_shape and s != target_shape: differs = True
                if differs: break
        attempts = 0
        while True:
            p = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4),
                            rng.uniform(0.7, 1.0)])
            if all(np.linalg.norm(p - q) > 0.18 for q in positions): break
            attempts += 1
            if attempts > 30: break
        positions.append(p)
        objects.append({"color": c, "shape": s, "pos": p})

    target_pos = objects[target_idx]["pos"]
    start = np.array([0.0, 0.0, 0.7])
    traj = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
    for h in range(HORIZON):
        alpha = h / (HORIZON - 1)
        pos = start * (1 - alpha) + target_pos * alpha
        pos[2] += 0.1 * np.sin(np.pi * alpha)
        traj[h, :3] = pos
        traj[h, 3:7] = [0.0, 0.0, 0.0, 1.0]
    return {"objects": objects, "n_obj": n_obj, "target_idx": target_idx,
            "target_pos": target_pos, "text": text, "traj_gt": traj}


def precompute_embeddings(scenes, clip_encoders, rng_for_render, batch_size=32):
    """Pre-computa CLIP-text y CLIP-image embeddings para todas las escenas."""
    from tqdm.auto import tqdm
    # Text
    texts = [s["text"] for s in scenes]
    text_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc="text", leave=False):
        text_embs.append(clip_encoders.encode_text(texts[i:i+batch_size]).cpu().numpy())
    text_embs = np.concatenate(text_embs, axis=0)

    # Image: para cada escena, renderizamos crops y los encodeamos (CLIP-vision 768-D)
    vis_embs = np.zeros((len(scenes), MAX_OBJ, ATTR_DIM_VIS), dtype=np.float32)
    all_crops = []
    crop_index = []  # (scene_idx, obj_idx)
    for i, s in enumerate(scenes):
        for j, obj in enumerate(s["objects"]):
            crop = render_object_crop(obj["color"], obj["shape"], rng=rng_for_render)
            all_crops.append(crop)
            crop_index.append((i, j))

    # Encode in batches
    for k in tqdm(range(0, len(all_crops), batch_size), desc="vision", leave=False):
        batch = all_crops[k:k+batch_size]
        emb = clip_encoders.encode_images(batch).cpu().numpy()
        for m, (si, oi) in enumerate(crop_index[k:k+batch_size]):
            vis_embs[si, oi] = emb[m]

    return text_embs, vis_embs


def assemble_cond(selected_pos, clip_proj, positions, vis_summary):
    """vis_summary: (B, 64) — proyeccion CLIP-image agregada via gate.vis_proj."""
    B = selected_pos.shape[0]
    cond = torch.zeros(B, COND_DIM, device=selected_pos.device, dtype=selected_pos.dtype)
    cond[:, :3] = selected_pos
    n_pos_dims = MAX_OBJ * 3  # 4*3=12 con MAX_OBJ=4
    cond[:, 3:3+n_pos_dims] = positions.view(B, -1)
    cond[:, 3+n_pos_dims:3+n_pos_dims+32] = clip_proj
    # Visual summary en lo que queda
    remaining = COND_DIM - (3 + n_pos_dims + 32)
    if remaining > 0:
        cond[:, 3+n_pos_dims+32:] = vis_summary[:, :remaining]
    return cond


class SceneDataset(Dataset):
    def __init__(self, scenes, text_embs, vis_embs):
        self.scenes = scenes
        self.text_embs = text_embs
        self.vis_embs = vis_embs

    def __len__(self): return len(self.scenes)

    def __getitem__(self, i):
        s = self.scenes[i]
        positions = np.zeros((MAX_OBJ, 3), dtype=np.float32)
        mask = np.zeros(MAX_OBJ, dtype=np.float32)
        for k, obj in enumerate(s["objects"]):
            positions[k] = obj["pos"]
            mask[k] = 1.0
        return {
            "traj": torch.tensor(s["traj_gt"]),
            "text_emb": torch.tensor(self.text_embs[i]),
            "vis_embs": torch.tensor(self.vis_embs[i]),
            "positions": torch.tensor(positions),
            "mask": torch.tensor(mask),
            "target_idx": torch.tensor(s["target_idx"], dtype=torch.long),
        }


def train(model, projector, gate, train_loader, val_loader, scheduler, device,
            n_epochs=50, lr=3e-4, warmup_epochs=3, grad_clip=1.0):
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
    params = list(model.parameters()) + list(projector.parameters()) + list(gate.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return max(1e-6/lr, 0.5 * (1 + np.cos(np.pi * progress)))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_val = float("inf"); best_state = None
    train_losses, val_losses = [], []
    t0 = time.time()

    for epoch in range(n_epochs):
        model.train(); projector.train(); gate.train()
        sum_diff, sum_gacc, n = 0, 0, 0
        for b in train_loader:
            xb = b["traj"].to(device); te = b["text_emb"].to(device)
            ve = b["vis_embs"].to(device); pos = b["positions"].to(device)
            mask = b["mask"].to(device); tgt = b["target_idx"].to(device)
            B = xb.shape[0]
            gates = gate(te, ve, mask)
            selected = (gates.unsqueeze(-1) * pos).sum(dim=1)
            clip_proj = projector(te)
            # Visual summary: weighted avg de proyecciones visuales
            vis_proj = gate.vis_proj(ve)  # (B, N, 64)
            vis_summary = (gates.unsqueeze(-1) * vis_proj).sum(dim=1)  # (B, 64)
            cb = assemble_cond(selected, clip_proj, pos, vis_summary)
            t = torch.randint(0, TIMESTEPS, (B,), device=device)
            eps = torch.randn_like(xb)
            ab_t = alpha_bar[t].view(-1, 1, 1)
            x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
            eps_pred = model(x_noisy, t, cb)
            loss_diff = F.mse_loss(eps_pred, eps)
            loss_cls = F.nll_loss((gates + 1e-9).log(), tgt)
            loss = loss_diff + 0.5 * loss_cls
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            sum_diff += loss_diff.item() * B
            sum_gacc += (gates.argmax(dim=-1) == tgt).float().mean().item() * B
            n += B
        train_losses.append(sum_diff / n); train_gacc = sum_gacc / n

        model.eval(); projector.eval(); gate.eval()
        with torch.no_grad():
            sum_diff, sum_gacc, n = 0, 0, 0
            for b in val_loader:
                xb = b["traj"].to(device); te = b["text_emb"].to(device)
                ve = b["vis_embs"].to(device); pos = b["positions"].to(device)
                mask = b["mask"].to(device); tgt = b["target_idx"].to(device)
                B = xb.shape[0]
                gates = gate(te, ve, mask)
                selected = (gates.unsqueeze(-1) * pos).sum(dim=1)
                clip_proj = projector(te)
                vis_proj = gate.vis_proj(ve)
                vis_summary = (gates.unsqueeze(-1) * vis_proj).sum(dim=1)
                cb = assemble_cond(selected, clip_proj, pos, vis_summary)
                t = torch.randint(0, TIMESTEPS, (B,), device=device)
                eps = torch.randn_like(xb)
                ab_t = alpha_bar[t].view(-1, 1, 1)
                x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
                eps_pred = model(x_noisy, t, cb)
                sum_diff += F.mse_loss(eps_pred, eps).item() * B
                sum_gacc += (gates.argmax(dim=-1) == tgt).float().mean().item() * B
                n += B
            val_loss = sum_diff / n; val_gacc = sum_gacc / n
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {"model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                            "projector": {k: v.cpu().clone() for k, v in projector.state_dict().items()},
                            "gate": {k: v.cpu().clone() for k, v in gate.state_dict().items()}}

        lr_sched.step()
        if (epoch + 1) % 5 == 0 or val_loss == best_val:
            print(f"  Ep {epoch+1:3d}/{n_epochs} | val={val_loss:.5f} | "
                  f"gate train/val={train_gacc:.1%}/{val_gacc:.1%}")

    elapsed = (time.time() - t0) / 60
    if best_state:
        model.load_state_dict(best_state["model"])
        projector.load_state_dict(best_state["projector"])
        gate.load_state_dict(best_state["gate"])
    print(f"  Training: {elapsed:.1f} min")
    return train_losses, val_losses, best_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=2500)
    ap.add_argument("--n-val", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=32)
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = get_device()
    print(f"[exp24] CLIP-image visual grounding | device={device}")

    print("\n[1/5] Cargando CLIP text + vision...")
    clip_encoders = CLIPEncoders(device)

    print(f"\n[2/5] Generando escenas y precomputando CLIP-text + CLIP-image embeddings...")
    rng = np.random.default_rng(SEED)
    scenes_train = [generate_scene(rng) for _ in range(args.n_train)]
    scenes_val = [generate_scene(rng) for _ in range(args.n_val)]
    print(f"  Train: {len(scenes_train)}, Val: {len(scenes_val)}")
    rng_render = np.random.default_rng(SEED + 100)
    text_train, vis_train = precompute_embeddings(scenes_train, clip_encoders, rng_render)
    text_val, vis_val = precompute_embeddings(scenes_val, clip_encoders, rng_render)
    print(f"  Text embs: {text_train.shape}, Vis embs: {vis_train.shape}")

    print("\n[3/5] Inicializando modelo...")
    model = ConditionalUNet1D(action_dim=ACTION_DIM, horizon=HORIZON,
                                 cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM).to(device)
    projector = CLIPProjector(512, 32).to(device)
    gate = VisualGate(text_dim=512, vis_dim=768).to(device)
    n_params = sum(p.numel() for p in model.parameters()) + \
                sum(p.numel() for p in projector.parameters()) + \
                sum(p.numel() for p in gate.parameters())
    print(f"  Params entrenables: {n_params/1e6:.2f} M (sin contar CLIP frozen 150M)")

    print("\n[4/5] Entrenando...")
    train_ds = SceneDataset(scenes_train, text_train, vis_train)
    val_ds = SceneDataset(scenes_val, text_val, vis_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    scheduler = SimpleDDPMScheduler(num_timesteps=TIMESTEPS)
    train_losses, val_losses, best_val = train(
        model, projector, gate, train_loader, val_loader, scheduler, device, args.epochs)

    print("\n[5/5] Evaluando...")
    # Eval: medir accuracy
    gate.eval(); projector.eval(); model.eval()
    correct = 0
    with torch.no_grad():
        for b in val_loader:
            te = b["text_emb"].to(device); ve = b["vis_embs"].to(device)
            mask = b["mask"].to(device); tgt = b["target_idx"].to(device)
            gates = gate(te, ve, mask)
            chosen = gates.argmax(dim=-1)
            correct += (chosen == tgt).sum().item()
    acc = correct / len(scenes_val)

    criteria = {"acc_min": 0.75}
    pass_flag = acc >= criteria["acc_min"]

    results = {
        "config": {"n_train": args.n_train, "n_val": args.n_val, "epochs": len(train_losses),
                    "device": device, "seed": SEED, "crop_size_px": 64,
                    "clip_model": "openai/clip-vit-base-patch32"},
        "training": {"best_val_loss": best_val},
        "evaluation": {"selection_accuracy": acc, "n_correct": correct, "n_total": len(scenes_val)},
        "criteria": criteria, "pass": pass_flag, "all_criteria_pass": pass_flag,
    }

    print("\n" + "=" * 60)
    print(f"  Selection accuracy (CLIP-image visual): {acc:.1%}  "
          f"({'PASA' if pass_flag else 'FALLA'})")
    print(f"  Decision: {'ALL PASS' if pass_flag else 'PARTIAL/FAIL'}")
    print("=" * 60)

    with open(OUTPUT_RESULTS / "exp24_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {OUTPUT_RESULTS / 'exp24_results.json'}")

    if pass_flag:
        torch.save({
            "model_state_dict": model.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "gate_state_dict": gate.state_dict(),
            "config": {"is_clip_image": True, "max_obj": MAX_OBJ},
        }, OUTPUT_MODEL)
        print(f"[OK] {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
