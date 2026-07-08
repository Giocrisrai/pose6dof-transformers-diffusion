#!/usr/bin/env python3
"""Experimento 26: razonamiento ESPACIAL en VLA-lite.

Frases tipo "the leftmost red", "the closest sphere", "the highest object",
"the one on the left", "the rightmost blue cylinder".

Approach: el gate ahora recibe ademas de los atributos visuales/categoricos,
las POSICIONES NORMALIZADAS de cada objeto (relativas al frame de la escena).
CLIP-text ya entiende palabras espaciales ("left", "right", "top", "bottom",
"closest", "farthest"); el gate aprende a alinear esas palabras con las
coordenadas espaciales de los objetos.

Salida:
    data/models/diffusion_policy_clip_spatial.pth
    experiments/results/exp26_spatial/exp26_results.json
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
COLOR_RGB = {"red": [1.0, 0.0, 0.0], "blue": [0.0, 0.0, 1.0], "green": [0.0, 1.0, 0.0]}
SHAPE_ENC = {s: [1.0 if s == x else 0.0 for x in SHAPES] for s in SHAPES}
ATTR_DIM = 3 + 4 + 3  # RGB + shape onehot + position xyz normalizada
MAX_OBJ = 4

# Templates espaciales (referencias a posicion sin mencionar todos los atributos)
TEMPLATES_SPATIAL = [
    "pick the leftmost {color} object",
    "grab the rightmost {shape}",
    "select the leftmost one",
    "the rightmost object",
    "pick the closest {color} {shape}",
    "the closest one",
    "the farthest object",
    "the highest one",
    "the lowest one",
    "the {color} on the left",
    "the {color} on the right",
    "pick the one closest to me",
    "the topmost {shape}",
]


OUTPUT_MODEL = REPO / "data/models/diffusion_policy_clip_spatial.pth"
OUTPUT_RESULTS = REPO / "experiments/results/exp26_spatial"
OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CLIPTextEncoder:
    def __init__(self, device):
        from transformers import CLIPTextModel, CLIPTokenizer
        self.device = device
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, texts):
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return self.model(**inputs).pooler_output


class CLIPProjector(nn.Module):
    def __init__(self, in_dim=512, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, 64), nn.Mish(), nn.Linear(64, out_dim))
    def forward(self, x): return self.net(x)


class SpatialGate(nn.Module):
    """Gate con razonamiento espacial: attrs = RGB + shape onehot + pos normalizada."""
    def __init__(self, clip_dim=512, attr_dim=ATTR_DIM, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(attr_dim + clip_dim, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, 1),
        )

    def forward(self, clip_emb, attrs, mask):
        B, N, _ = attrs.shape
        x = torch.cat([attrs, clip_emb.unsqueeze(1).expand(-1, N, -1)], -1)
        logits = self.score(x).squeeze(-1).masked_fill(mask == 0, -1e9)
        return F.softmax(logits, dim=-1)


def resolve_target(template: str, objects: list, target_color: str, target_shape: str) -> int:
    """Resuelve el indice target segun el template y los objetos.

    Si menciona color: filtra por color y elige segun el modificador espacial.
    Si menciona shape: igual.
    Si no menciona ninguno: aplica el modificador a TODOS los objetos.
    """
    indices_pool = list(range(len(objects)))
    if "{color}" in template:
        indices_pool = [i for i in indices_pool if objects[i]["color"] == target_color]
    if "{shape}" in template:
        indices_pool = [i for i in indices_pool if objects[i]["shape"] == target_shape]
    if not indices_pool:
        return -1

    if "leftmost" in template or "on the left" in template:
        return min(indices_pool, key=lambda i: objects[i]["pos"][0])
    if "rightmost" in template or "on the right" in template:
        return max(indices_pool, key=lambda i: objects[i]["pos"][0])
    if "closest" in template:
        # Y mas grande = mas cerca del robot (asumiendo origen en frente)
        return max(indices_pool, key=lambda i: objects[i]["pos"][1])
    if "farthest" in template:
        return min(indices_pool, key=lambda i: objects[i]["pos"][1])
    if "highest" in template or "topmost" in template:
        return max(indices_pool, key=lambda i: objects[i]["pos"][2])
    if "lowest" in template:
        return min(indices_pool, key=lambda i: objects[i]["pos"][2])
    # Fallback
    return indices_pool[0]


def generate_scene(rng):
    n_obj = int(rng.integers(3, MAX_OBJ + 1))  # min 3 para que las refs espaciales tengan sentido
    template = TEMPLATES_SPATIAL[rng.integers(0, len(TEMPLATES_SPATIAL))]

    # Color/shape target placeholders
    target_color = COLORS[rng.integers(0, len(COLORS))]
    target_shape = SHAPES[rng.integers(0, len(SHAPES))]

    # Generar objetos: garantizar que el template se puede resolver
    for attempt in range(50):
        objects = []
        positions = []
        for i in range(n_obj):
            c = COLORS[rng.integers(0, len(COLORS))]
            s = SHAPES[rng.integers(0, len(SHAPES))]
            attempts = 0
            while True:
                p = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4),
                                rng.uniform(0.7, 1.0)])
                if all(np.linalg.norm(p - q) > 0.18 for q in positions): break
                attempts += 1
                if attempts > 30: break
            positions.append(p)
            objects.append({"color": c, "shape": s, "pos": p})

        # Si el template menciona color, asegurar al menos 2 del color target
        if "{color}" in template:
            color_count = sum(1 for o in objects if o["color"] == target_color)
            if color_count < 2:
                # Forzar 2 objetos del color
                for j in range(min(2, n_obj)):
                    objects[j]["color"] = target_color
        if "{shape}" in template:
            shape_count = sum(1 for o in objects if o["shape"] == target_shape)
            if shape_count < 2:
                for j in range(min(2, n_obj)):
                    objects[j]["shape"] = target_shape

        target_idx = resolve_target(template, objects, target_color, target_shape)
        if target_idx >= 0:
            break

    text = template.format(color=target_color, shape=target_shape)
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
            "target_pos": target_pos, "text": text, "traj_gt": traj,
            "template": template}


def scene_to_tensors(scene, max_obj=MAX_OBJ):
    attrs = np.zeros((max_obj, ATTR_DIM), dtype=np.float32)
    positions = np.zeros((max_obj, 3), dtype=np.float32)
    mask = np.zeros(max_obj, dtype=np.float32)
    for i, obj in enumerate(scene["objects"]):
        attrs[i, :3] = COLOR_RGB[obj["color"]]
        attrs[i, 3:7] = SHAPE_ENC[obj["shape"]]
        # Position normalizada a [-1, 1] (rango aprox del workspace)
        attrs[i, 7:10] = obj["pos"] / np.array([0.4, 0.4, 1.0])
        positions[i] = obj["pos"]
        mask[i] = 1.0
    return attrs, positions, mask, scene["target_idx"]


def assemble_cond(selected_pos, clip_proj, positions, attrs):
    B = selected_pos.shape[0]
    cond = torch.zeros(B, COND_DIM, device=selected_pos.device, dtype=selected_pos.dtype)
    cond[:, :3] = selected_pos
    cond[:, 3:15] = positions.view(B, -1)[:, :12]  # 4 obj x 3
    cond[:, 15:47] = clip_proj
    # Atributos resumen: usar shape de objeto target (selected via softmax sobre attrs)
    cond[:, 47:57] = attrs.max(dim=1).values
    return cond


class SceneDataset(Dataset):
    def __init__(self, scenes, clip_embs):
        self.scenes = scenes
        self.clip_embs = clip_embs

    def __len__(self): return len(self.scenes)

    def __getitem__(self, i):
        s = self.scenes[i]
        attrs, positions, mask, target = scene_to_tensors(s)
        return {
            "traj": torch.tensor(s["traj_gt"]),
            "attrs": torch.tensor(attrs),
            "positions": torch.tensor(positions),
            "mask": torch.tensor(mask),
            "target_idx": torch.tensor(target, dtype=torch.long),
            "clip_emb": torch.tensor(self.clip_embs[i]),
        }


def make_dataset(n, seed, clip_encoder, batch_clip=64):
    from tqdm.auto import tqdm
    rng = np.random.default_rng(seed)
    scenes = [generate_scene(rng) for _ in range(n)]
    texts = [s["text"] for s in scenes]
    clip_embs = []
    for i in tqdm(range(0, n, batch_clip), desc="clip", leave=False):
        clip_embs.append(clip_encoder.encode(texts[i:i+batch_clip]).cpu().numpy())
    clip_embs = np.concatenate(clip_embs, axis=0)
    return scenes, clip_embs


def train(model, projector, gate, train_loader, val_loader, scheduler, device,
            n_epochs=60, lr=3e-4, warmup_epochs=3):
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
            xb = b["traj"].to(device); attrs = b["attrs"].to(device)
            pos = b["positions"].to(device); mask = b["mask"].to(device)
            tgt = b["target_idx"].to(device); ce = b["clip_emb"].to(device)
            B = xb.shape[0]
            gates = gate(ce, attrs, mask)
            selected = (gates.unsqueeze(-1) * pos).sum(dim=1)
            clip_proj = projector(ce)
            cb = assemble_cond(selected, clip_proj, pos, attrs)
            t = torch.randint(0, TIMESTEPS, (B,), device=device)
            eps = torch.randn_like(xb)
            ab_t = alpha_bar[t].view(-1, 1, 1)
            x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
            eps_pred = model(x_noisy, t, cb)
            loss_diff = F.mse_loss(eps_pred, eps)
            loss_cls = F.nll_loss((gates + 1e-9).log(), tgt)
            loss = loss_diff + 0.5 * loss_cls
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimizer.step()
            sum_diff += loss_diff.item() * B
            sum_gacc += (gates.argmax(dim=-1) == tgt).float().mean().item() * B
            n += B
        train_losses.append(sum_diff / n); train_gacc = sum_gacc / n

        model.eval(); projector.eval(); gate.eval()
        with torch.no_grad():
            sum_diff, sum_gacc, n = 0, 0, 0
            for b in val_loader:
                xb = b["traj"].to(device); attrs = b["attrs"].to(device)
                pos = b["positions"].to(device); mask = b["mask"].to(device)
                tgt = b["target_idx"].to(device); ce = b["clip_emb"].to(device)
                B = xb.shape[0]
                gates = gate(ce, attrs, mask)
                selected = (gates.unsqueeze(-1) * pos).sum(dim=1)
                clip_proj = projector(ce)
                cb = assemble_cond(selected, clip_proj, pos, attrs)
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


def evaluate(model, projector, gate, scheduler, scenes, clip_embs, device, batch_size=64):
    n = len(scenes)
    chosen_indices = np.empty(n, dtype=np.int64)
    model.eval(); projector.eval(); gate.eval()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch = scenes[i:i+batch_size]
            attrs = torch.tensor(np.stack([scene_to_tensors(s)[0] for s in batch]), device=device)
            mask = torch.tensor(np.stack([scene_to_tensors(s)[2] for s in batch]), device=device)
            ce = torch.tensor(clip_embs[i:i+batch_size], device=device)
            gates = gate(ce, attrs, mask)
            chosen_indices[i:i+batch_size] = gates.argmax(dim=-1).cpu().numpy()
    correct = 0
    by_template = {}
    for i, s in enumerate(scenes):
        ok = chosen_indices[i] == s["target_idx"]
        if ok: correct += 1
        tpl = s["template"]
        if tpl not in by_template:
            by_template[tpl] = {"correct": 0, "total": 0}
        by_template[tpl]["correct"] += int(ok); by_template[tpl]["total"] += 1
    return {
        "selection_accuracy_global": correct / n,
        "accuracy_by_template": {tpl: r["correct"] / r["total"] for tpl, r in by_template.items()},
        "n_by_template": {tpl: r["total"] for tpl, r in by_template.items()},
        "n_evaluated": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=8000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(SEED); np.random.seed(SEED)
    device = get_device()
    print(f"[exp26] Razonamiento espacial | device={device} | attr_dim={ATTR_DIM} (RGB+shape+pos)")

    print("\n[1/4] CLIP text encoder...")
    clip_encoder = CLIPTextEncoder(device)

    print("\n[2/4] Generando escenas con templates espaciales...")
    scenes_train, ce_train = make_dataset(args.n_train, SEED, clip_encoder)
    scenes_val, ce_val = make_dataset(args.n_val, SEED + 1, clip_encoder)
    print(f"  Train: {len(scenes_train)}, Val: {len(scenes_val)}")
    print("  Ejemplos:")
    for s in scenes_train[:4]:
        print(f"    '{s['text']}' | n_obj={s['n_obj']} | target={s['target_idx']}")

    print("\n[3/4] Entrenando...")
    model = ConditionalUNet1D(action_dim=ACTION_DIM, horizon=HORIZON,
                                 cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM).to(device)
    projector = CLIPProjector(512, 32).to(device)
    gate = SpatialGate(clip_dim=512, attr_dim=ATTR_DIM).to(device)

    train_ds = SceneDataset(scenes_train, ce_train)
    val_ds = SceneDataset(scenes_val, ce_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    scheduler = SimpleDDPMScheduler(num_timesteps=TIMESTEPS)
    train_losses, val_losses, best_val = train(
        model, projector, gate, train_loader, val_loader, scheduler, device, args.epochs)

    print("\n[4/4] Evaluando...")
    metrics = evaluate(model, projector, gate, scheduler, scenes_val, ce_val, device, args.batch_size)

    criteria = {"acc_min_global": 0.70, "acc_min_per_template": 0.55}
    pass_flags = {
        "global_above": metrics["selection_accuracy_global"] >= criteria["acc_min_global"],
        "all_templates_above": all(a >= criteria["acc_min_per_template"]
                                       for a in metrics["accuracy_by_template"].values()),
    }
    all_pass = all(pass_flags.values())

    results = {
        "config": {"n_train": args.n_train, "n_val": args.n_val, "epochs": len(train_losses),
                    "attr_dim": ATTR_DIM, "templates": TEMPLATES_SPATIAL,
                    "device": device, "seed": SEED},
        "training": {"best_val_loss": best_val},
        "evaluation": metrics, "criteria": criteria, "pass": pass_flags,
        "all_criteria_pass": all_pass,
    }
    print("\n" + "=" * 60)
    print(f"  GLOBAL accuracy: {metrics['selection_accuracy_global']:.1%}  "
          f"({'PASA' if pass_flags['global_above'] else 'FALLA'})")
    print("  Por template:")
    for tpl, acc in sorted(metrics["accuracy_by_template"].items(), key=lambda x: -x[1]):
        n = metrics["n_by_template"][tpl]
        flag = "✓" if acc >= criteria["acc_min_per_template"] else "✗"
        print(f"    {flag} '{tpl}' -> {acc:.1%} (n={n})")
    print(f"  Decision: {'ALL PASS' if all_pass else 'PARTIAL/FAIL'}")
    print("=" * 60)

    with open(OUTPUT_RESULTS / "exp26_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {OUTPUT_RESULTS / 'exp26_results.json'}")
    if all_pass:
        torch.save({
            "model_state_dict": model.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "gate_state_dict": gate.state_dict(),
            "config": {"attr_dim": ATTR_DIM, "is_spatial": True},
        }, OUTPUT_MODEL)
        print(f"[OK] {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
