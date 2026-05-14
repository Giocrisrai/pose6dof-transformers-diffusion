#!/usr/bin/env python3
"""Experimento 18: extension VLA-lite a FORMAS (no solo colores).

Anade la dimension shape al sistema:
- 4 formas: cube, sphere, cylinder, box
- 3 colores (heredados del exp16): red, blue, green
- Templates: "pick the red sphere", "grab the cylinder", "select the cube", etc.

El modelo aprende a:
1. Si la frase menciona forma: discriminar por shape
2. Si menciona color: discriminar por color
3. Si menciona ambos: discriminar por la combinacion exacta

Salida:
    data/models/diffusion_policy_clip_shapes.pth
    experiments/results/exp18_vla_shapes/exp18_results.json
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
from torch.utils.data import DataLoader, TensorDataset

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
# One-hot 4-D para la forma
SHAPE_ENC = {s: [1.0 if s == x else 0.0 for x in SHAPES] for s in SHAPES}

# 3 familias de templates segun el atributo descrito
TEMPLATES_COLOR = [
    "pick the {color} object",
    "grab the {color} item",
    "select the {color} one",
]
TEMPLATES_SHAPE = [
    "pick the {shape}",
    "grab the {shape}",
    "select the {shape}",
    "take the {shape}",
]
TEMPLATES_BOTH = [
    "pick the {color} {shape}",
    "grab the {color} {shape}",
    "select the {color} {shape}",
    "take the {color} {shape}",
]


OUTPUT_MODEL = REPO / "data/models/diffusion_policy_clip_shapes.pth"
OUTPUT_RESULTS = REPO / "experiments/results/exp18_vla_shapes"
OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CLIPTextEncoder:
    def __init__(self, device):
        from transformers import CLIPTokenizer, CLIPTextModel
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
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64), nn.Mish(), nn.Linear(64, out_dim))

    def forward(self, x):
        return self.net(x)


class MultiAttributeGate(nn.Module):
    """Generaliza TextGroundedGate del exp16 a multiples atributos.

    Cada objeto se describe por:
    - RGB (3-D) = color
    - shape_onehot (4-D) = forma

    El gate combina CLIP + atributos para producir gates softmax.
    """
    def __init__(self, clip_dim=512, attr_dim=3+4, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(attr_dim + clip_dim, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, 1),
        )

    def forward(self, clip_emb, attr_a, attr_b):
        """clip_emb: (B, 512); attr_*: (B, 7) = [RGB; shape_onehot]."""
        s_a = self.score(torch.cat([attr_a, clip_emb], -1)).squeeze(-1)
        s_b = self.score(torch.cat([attr_b, clip_emb], -1)).squeeze(-1)
        gates = F.softmax(torch.stack([s_a, s_b], -1), dim=-1)
        return gates[:, 0], gates[:, 1]


def generate_scene(rng):
    """Genera escena con 2 objetos.

    Politica de generacion:
    - 33 % escenas color-distinguible (mismo shape, distinto color) -> usar TEMPLATES_COLOR
    - 33 % escenas shape-distinguible (mismo color, distinto shape) -> usar TEMPLATES_SHAPE
    - 33 % escenas mixtas (distintos ambos) -> usar TEMPLATES_BOTH
    """
    mode = rng.integers(0, 3)  # 0=color, 1=shape, 2=both

    if mode == 0:
        # Mismo shape, colores distintos
        shape = SHAPES[rng.integers(0, len(SHAPES))]
        c_a, c_b = COLORS[0], COLORS[0]
        while c_a == c_b:
            c_idx = rng.choice(len(COLORS), size=2, replace=False)
            c_a, c_b = COLORS[c_idx[0]], COLORS[c_idx[1]]
        s_a = s_b = shape
        target_idx = rng.integers(0, 2)
        target_attr_value = c_a if target_idx == 0 else c_b
        template = TEMPLATES_COLOR[rng.integers(0, len(TEMPLATES_COLOR))]
        text = template.format(color=target_attr_value)

    elif mode == 1:
        # Mismo color, shapes distintos
        color = COLORS[rng.integers(0, len(COLORS))]
        s_idx = rng.choice(len(SHAPES), size=2, replace=False)
        s_a, s_b = SHAPES[s_idx[0]], SHAPES[s_idx[1]]
        c_a = c_b = color
        target_idx = rng.integers(0, 2)
        target_attr_value = s_a if target_idx == 0 else s_b
        template = TEMPLATES_SHAPE[rng.integers(0, len(TEMPLATES_SHAPE))]
        text = template.format(shape=target_attr_value)

    else:
        # Distintos ambos
        c_idx = rng.choice(len(COLORS), size=2, replace=False)
        s_idx = rng.choice(len(SHAPES), size=2, replace=False)
        c_a, c_b = COLORS[c_idx[0]], COLORS[c_idx[1]]
        s_a, s_b = SHAPES[s_idx[0]], SHAPES[s_idx[1]]
        target_idx = rng.integers(0, 2)
        tgt_color = c_a if target_idx == 0 else c_b
        tgt_shape = s_a if target_idx == 0 else s_b
        template = TEMPLATES_BOTH[rng.integers(0, len(TEMPLATES_BOTH))]
        text = template.format(color=tgt_color, shape=tgt_shape)

    while True:
        p_a = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.7, 1.0)])
        p_b = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.7, 1.0)])
        if np.linalg.norm(p_a - p_b) > 0.20:
            break

    target_pos = p_a if target_idx == 0 else p_b
    start = np.array([0.0, 0.0, 0.7])
    traj = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
    for h in range(HORIZON):
        alpha = h / (HORIZON - 1)
        pos = start * (1 - alpha) + target_pos * alpha
        pos[2] += 0.1 * np.sin(np.pi * alpha)
        traj[h, :3] = pos
        traj[h, 3:7] = [0.0, 0.0, 0.0, 1.0]

    return {
        "p_a": p_a, "p_b": p_b, "c_a": c_a, "c_b": c_b,
        "s_a": s_a, "s_b": s_b, "target_idx": int(target_idx),
        "target_pos": target_pos, "distractor_pos": p_b if target_idx == 0 else p_a,
        "text": text, "traj_gt": traj, "mode": int(mode),
    }


def build_static_cond(scene):
    """Cond fija: pos_a, pos_b, atributos (RGB+shape) para A y B."""
    cond = np.zeros(COND_DIM, dtype=np.float32)
    cond[4:7] = scene["p_a"]
    cond[7:10] = scene["p_b"]
    # Atributo A: [RGB(3), shape_onehot(4)] -> dims 10..17
    cond[10:13] = COLOR_RGB[scene["c_a"]]
    cond[13:17] = SHAPE_ENC[scene["s_a"]]
    # Atributo B: [RGB(3), shape_onehot(4)] -> dims 17..24
    cond[17:20] = COLOR_RGB[scene["c_b"]]
    cond[20:24] = SHAPE_ENC[scene["s_b"]]
    return cond


def assemble_cond(static_cond, selected_pos, clip_proj):
    out = static_cond.clone()
    out[:, :3] = selected_pos
    out[:, 24:56] = clip_proj  # 32-D CLIP proj
    return out


def make_dataset(n, seed, clip_encoder, device, batch_clip=64):
    from tqdm.auto import tqdm
    rng = np.random.default_rng(seed)
    scenes = [generate_scene(rng) for _ in range(n)]
    texts = [s["text"] for s in scenes]
    clip_embs = []
    for i in tqdm(range(0, n, batch_clip), desc="clip", leave=False):
        clip_embs.append(clip_encoder.encode(texts[i:i+batch_clip]).cpu().numpy())
    clip_embs = np.concatenate(clip_embs, axis=0)
    static_conds = np.array([build_static_cond(s) for s in scenes], dtype=np.float32)
    trajs = np.array([s["traj_gt"] for s in scenes], dtype=np.float32)
    return scenes, static_conds, trajs, clip_embs


def extract_attrs(static_cond):
    """Extrae attr_a (RGB+shape) y attr_b de static_cond. Shapes: (B, 7) cada uno."""
    attr_a = torch.cat([static_cond[:, 10:13], static_cond[:, 13:17]], dim=-1)
    attr_b = torch.cat([static_cond[:, 17:20], static_cond[:, 20:24]], dim=-1)
    return attr_a, attr_b


def ddim_sample(model, scheduler, cond, device, n_steps=25):
    B = cond.shape[0]
    x = torch.randn(B, HORIZON, ACTION_DIM, device=device, dtype=cond.dtype)
    si = np.linspace(0, scheduler.num_timesteps - 1, n_steps).astype(int)[::-1]
    ab = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, step in enumerate(si):
            t = torch.full((B,), int(step), dtype=torch.long, device=device)
            e = model(x, t, cond)
            ab_t = ab[step]
            x0 = (x - torch.sqrt(1 - ab_t) * e) / torch.sqrt(ab_t)
            if i < len(si) - 1:
                ab_n = ab[si[i + 1]]
                x = torch.sqrt(ab_n) * x0 + torch.sqrt(1 - ab_n) * e
            else:
                x = x0
    return x.cpu().numpy()


def train_full(model, projector, gate, train_loader, val_loader, scheduler, device,
                  n_epochs=40, lr=3e-4, warmup_epochs=3, grad_clip=1.0):
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
        for xb, sc, ce, tgt in train_loader:
            xb, sc, ce, tgt = xb.to(device), sc.to(device), ce.to(device), tgt.to(device)
            B = xb.shape[0]
            attr_a, attr_b = extract_attrs(sc)
            p_a = sc[:, 4:7]; p_b = sc[:, 7:10]
            g_a, g_b = gate(ce, attr_a, attr_b)
            selected = g_a.unsqueeze(-1) * p_a + g_b.unsqueeze(-1) * p_b
            clip_proj = projector(ce)
            cb = assemble_cond(sc, selected, clip_proj)
            t = torch.randint(0, TIMESTEPS, (B,), device=device)
            eps = torch.randn_like(xb)
            ab_t = alpha_bar[t].view(-1, 1, 1)
            x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
            eps_pred = model(x_noisy, t, cb)
            loss_diff = F.mse_loss(eps_pred, eps)
            logits = torch.stack([g_a, g_b], -1).log()
            loss_cls = F.nll_loss(logits, tgt)
            loss = loss_diff + 0.5 * loss_cls
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            sum_diff += loss_diff.item() * B
            sum_gacc += ((g_a > g_b).long() == (tgt == 0).long()).float().mean().item() * B
            n += B
        train_losses.append(sum_diff / n)
        gate_train_acc = sum_gacc / n

        model.eval(); projector.eval(); gate.eval()
        with torch.no_grad():
            sum_diff, sum_gacc, n = 0, 0, 0
            for xb, sc, ce, tgt in val_loader:
                xb, sc, ce, tgt = xb.to(device), sc.to(device), ce.to(device), tgt.to(device)
                B = xb.shape[0]
                attr_a, attr_b = extract_attrs(sc)
                p_a = sc[:, 4:7]; p_b = sc[:, 7:10]
                g_a, g_b = gate(ce, attr_a, attr_b)
                selected = g_a.unsqueeze(-1) * p_a + g_b.unsqueeze(-1) * p_b
                clip_proj = projector(ce)
                cb = assemble_cond(sc, selected, clip_proj)
                t = torch.randint(0, TIMESTEPS, (B,), device=device)
                eps = torch.randn_like(xb)
                ab_t = alpha_bar[t].view(-1, 1, 1)
                x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
                eps_pred = model(x_noisy, t, cb)
                sum_diff += F.mse_loss(eps_pred, eps).item() * B
                sum_gacc += ((g_a > g_b).long() == (tgt == 0).long()).float().mean().item() * B
                n += B
            val_loss = sum_diff / n
            gate_val_acc = sum_gacc / n
        val_losses.append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                "model": {k: v.cpu().clone() for k, v in model.state_dict().items()},
                "projector": {k: v.cpu().clone() for k, v in projector.state_dict().items()},
                "gate": {k: v.cpu().clone() for k, v in gate.state_dict().items()},
            }

        lr_sched.step()
        if (epoch + 1) % 5 == 0 or val_loss == best_val:
            print(f"  Ep {epoch+1:3d}/{n_epochs} | diff={train_losses[-1]:.5f} | "
                  f"val={val_loss:.5f} | gate train/val={gate_train_acc:.1%}/{gate_val_acc:.1%}")

    elapsed = (time.time() - t0) / 60
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        projector.load_state_dict(best_state["projector"])
        gate.load_state_dict(best_state["gate"])
    print(f"  Training: {elapsed:.1f} min")
    return train_losses, val_losses, best_val


def evaluate(model, projector, gate, scheduler, scenes, static_conds, clip_embs,
              device, batch_size=64):
    """Selection accuracy global + por modo (color / shape / both)."""
    n = len(static_conds)
    trajs_pred = np.empty((n, HORIZON, ACTION_DIM), dtype=np.float32)
    model.eval(); projector.eval(); gate.eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sc = torch.tensor(static_conds[i:i+batch_size], device=device)
            ce = torch.tensor(clip_embs[i:i+batch_size], device=device)
            attr_a, attr_b = extract_attrs(sc)
            p_a = sc[:, 4:7]; p_b = sc[:, 7:10]
            g_a, g_b = gate(ce, attr_a, attr_b)
            selected = g_a.unsqueeze(-1) * p_a + g_b.unsqueeze(-1) * p_b
            clip_proj = projector(ce)
            cb = assemble_cond(sc, selected, clip_proj)
            x = ddim_sample(model, scheduler, cb, device, 25)
            trajs_pred[i:i+batch_size] = x
    latency = (time.time() - t0) * 1000 / n

    # Accuracy global + por modo
    correct = {"global": 0, 0: 0, 1: 0, 2: 0}
    total = {"global": 0, 0: 0, 1: 0, 2: 0}
    for i, scene in enumerate(scenes):
        endpoint = trajs_pred[i, -1, :3]
        d_target = np.linalg.norm(endpoint - scene["target_pos"])
        d_distractor = np.linalg.norm(endpoint - scene["distractor_pos"])
        ok = d_target < d_distractor
        correct["global"] += int(ok); total["global"] += 1
        correct[scene["mode"]] += int(ok); total[scene["mode"]] += 1

    mode_names = {0: "color", 1: "shape", 2: "both"}
    return {
        "selection_accuracy_global": correct["global"] / total["global"],
        "accuracy_by_mode": {mode_names[k]: correct[k] / max(total[k], 1) for k in [0, 1, 2]},
        "n_by_mode": {mode_names[k]: total[k] for k in [0, 1, 2]},
        "latency_ms_per_traj_ddim25": latency,
        "n_evaluated": total["global"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1200)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print(f"[exp18] device={device} n_train={args.n_train} n_val={args.n_val} epochs={args.epochs}")

    print("\n[1/4] Cargando CLIP text encoder...")
    clip_encoder = CLIPTextEncoder(device)

    print("\n[2/4] Generando escenas multi-atributo (color/shape/both)...")
    scenes_train, sc_train, traj_train, ce_train = make_dataset(
        args.n_train, SEED, clip_encoder, device)
    scenes_val, sc_val, traj_val, ce_val = make_dataset(
        args.n_val, SEED + 1, clip_encoder, device)
    modes_train = np.bincount([s["mode"] for s in scenes_train], minlength=3)
    print(f"  Train modes: color={modes_train[0]} shape={modes_train[1]} both={modes_train[2]}")
    print(f"  Ejemplos:")
    for s in scenes_train[:3]:
        print(f"    text='{s['text']}' | A=({s['c_a']},{s['s_a']}) B=({s['c_b']},{s['s_b']}) tgt={s['target_idx']}")

    print("\n[3/4] Entrenando modelo VLA-lite con multi-atributo...")
    model = ConditionalUNet1D(action_dim=ACTION_DIM, horizon=HORIZON,
                                 cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM).to(device)
    projector = CLIPProjector(512, 32).to(device)
    gate = MultiAttributeGate(clip_dim=512, attr_dim=7, hidden=128).to(device)

    target_train = np.array([s["target_idx"] for s in scenes_train], dtype=np.int64)
    target_val = np.array([s["target_idx"] for s in scenes_val], dtype=np.int64)
    train_ds = TensorDataset(torch.tensor(traj_train), torch.tensor(sc_train),
                                torch.tensor(ce_train), torch.tensor(target_train))
    val_ds = TensorDataset(torch.tensor(traj_val), torch.tensor(sc_val),
                              torch.tensor(ce_val), torch.tensor(target_val))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    scheduler = SimpleDDPMScheduler(num_timesteps=TIMESTEPS)
    train_losses, val_losses, best_val = train_full(
        model, projector, gate, train_loader, val_loader, scheduler, device,
        n_epochs=args.epochs)

    print("\n[4/4] Evaluando selection accuracy global + por modo...")
    metrics = evaluate(model, projector, gate, scheduler, scenes_val, sc_val, ce_val,
                        device, batch_size=args.batch_size)

    criteria = {"acc_min_global": 0.75, "acc_min_per_mode": 0.65}
    pass_flags = {
        "global_above_target": metrics["selection_accuracy_global"] >= criteria["acc_min_global"],
        "color_above": metrics["accuracy_by_mode"]["color"] >= criteria["acc_min_per_mode"],
        "shape_above": metrics["accuracy_by_mode"]["shape"] >= criteria["acc_min_per_mode"],
        "both_above": metrics["accuracy_by_mode"]["both"] >= criteria["acc_min_per_mode"],
    }
    all_pass = all(pass_flags.values())

    results = {
        "config": {"n_train": args.n_train, "n_val": args.n_val,
                    "epochs": len(train_losses), "batch_size": args.batch_size,
                    "device": device, "seed": SEED,
                    "colors": COLORS, "shapes": SHAPES},
        "training": {"best_val_loss": best_val,
                      "train_losses_last5": train_losses[-5:],
                      "val_losses_last5": val_losses[-5:]},
        "evaluation": metrics,
        "criteria": criteria,
        "pass": pass_flags,
        "all_criteria_pass": all_pass,
        "sample_scenes": [{"text": s["text"], "A": f"{s['c_a']}_{s['s_a']}",
                            "B": f"{s['c_b']}_{s['s_b']}", "target": s["target_idx"]}
                           for s in scenes_val[:10]],
    }

    print("\n" + "=" * 60)
    print(f"  GLOBAL selection acc: {metrics['selection_accuracy_global']:.1%}  "
          f"({'PASA' if pass_flags['global_above_target'] else 'FALLA'})")
    print(f"  Por modo:")
    for mode, acc in metrics["accuracy_by_mode"].items():
        n = metrics["n_by_mode"][mode]
        flag = "✓" if acc >= criteria["acc_min_per_mode"] else "✗"
        print(f"    {flag} {mode:6s}: {acc:.1%} (n={n})")
    print(f"  Latencia DDIM-25: {metrics['latency_ms_per_traj_ddim25']:.2f} ms/traj")
    print(f"  Decision: {'ALL PASS' if all_pass else 'PARTIAL/FAIL'}")
    print("=" * 60)

    with open(OUTPUT_RESULTS / "exp18_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {OUTPUT_RESULTS / 'exp18_results.json'}")

    if all_pass:
        torch.save({
            "model_state_dict": model.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "gate_state_dict": gate.state_dict(),
            "config": {"horizon": HORIZON, "action_dim": ACTION_DIM,
                        "cond_dim": COND_DIM, "hidden_dim": HIDDEN_DIM,
                        "clip_model": "openai/clip-vit-base-patch32",
                        "colors": COLORS, "shapes": SHAPES,
                        "is_vla_lite_shapes": True},
            "metrics": metrics,
        }, OUTPUT_MODEL)
        print(f"[OK] {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
