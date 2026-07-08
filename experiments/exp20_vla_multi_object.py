#!/usr/bin/env python3
"""Experimento 20: VLA-lite con N>2 objetos en la escena.

Extension natural del exp 18: el MultiAttributeGate se generaliza a N
objetos (N en {2, 3, 4, 5}) usando softmax sobre todos los candidatos.
El UNet recibe la posicion del objeto seleccionado (top-1 o softmax-mix).

Casos de uso desbloqueados:
- Cintas de logistica con varios paquetes simultaneos
- Bins de reciclaje con multiples categorias visibles
- PCBs con varios componentes a la vez

Esta exploracion responde al roadmap de docs/EXTRAPOLACION_INDUSTRIAL.md
seccion "Multi-objeto en una sola instruccion" (esfuerzo planificado 2-3 dias).

Salida:
    data/models/diffusion_policy_clip_multi.pth
    experiments/results/exp20_vla_multi_object/exp20_results.json
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
ATTR_DIM = 3 + 4  # RGB + shape onehot
MAX_OBJ = 5

TEMPLATES = [
    "pick the {color} {shape}",
    "grab the {color} {shape}",
    "select the {color} {shape}",
    "take the {color} {shape}",
    "fetch the {color} {shape}",
]

OUTPUT_MODEL = REPO / "data/models/diffusion_policy_clip_multi.pth"
OUTPUT_RESULTS = REPO / "experiments/results/exp20_vla_multi_object"
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

    def forward(self, x):
        return self.net(x)


class MultiObjectGate(nn.Module):
    """Gate generalizado a N objetos (N variable hasta MAX_OBJ).

    Aplica el mismo scorer (shared weights) a cada objeto independientemente
    y produce logits-N que pasan por softmax. Los slots "vacios" (escena
    con menos de MAX_OBJ objetos) se marcan con un mask que se aplica antes
    del softmax (logits enmascarados a -inf).
    """
    def __init__(self, clip_dim=512, attr_dim=ATTR_DIM, hidden=128):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(attr_dim + clip_dim, hidden), nn.Mish(),
            nn.Linear(hidden, hidden), nn.Mish(),
            nn.Linear(hidden, 1),
        )

    def forward(self, clip_emb, attrs, mask):
        """clip_emb: (B, 512); attrs: (B, MAX_OBJ, attr_dim);
        mask: (B, MAX_OBJ) (1 = valid, 0 = padding). Devuelve (B, MAX_OBJ) softmax."""
        B, N, A = attrs.shape
        clip_expand = clip_emb.unsqueeze(1).expand(-1, N, -1)
        x = torch.cat([attrs, clip_expand], dim=-1)        # (B, N, A+512)
        logits = self.score(x).squeeze(-1)                  # (B, N)
        # Mask padding -> -inf
        logits = logits.masked_fill(mask == 0, -1e9)
        return F.softmax(logits, dim=-1)


def generate_scene(rng):
    """Genera escena con N objetos en {2,3,4,5}.

    Garantiza al menos uno con la combinacion (color, shape) buscada.
    """
    n_obj = int(rng.integers(2, MAX_OBJ + 1))

    # Elegir target color+shape
    target_color = COLORS[rng.integers(0, len(COLORS))]
    target_shape = SHAPES[rng.integers(0, len(SHAPES))]
    text = TEMPLATES[rng.integers(0, len(TEMPLATES))].format(
        color=target_color, shape=target_shape)

    # Generar objetos: el primero es el target, el resto distractores distintos
    objects = []
    positions = []
    target_idx = int(rng.integers(0, n_obj))

    for i in range(n_obj):
        if i == target_idx:
            c, s = target_color, target_shape
        else:
            # Distractor: cambiar al menos uno de color/shape
            while True:
                c = COLORS[rng.integers(0, len(COLORS))]
                s = SHAPES[rng.integers(0, len(SHAPES))]
                if (c, s) != (target_color, target_shape):
                    break

        # Posicion no demasiado cerca de las anteriores
        attempts = 0
        while True:
            p = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4),
                            rng.uniform(0.7, 1.0)])
            if all(np.linalg.norm(p - q) > 0.18 for q in positions):
                break
            attempts += 1
            if attempts > 30:
                break  # acepta lo que haya
        positions.append(p)
        objects.append({"color": c, "shape": s, "pos": p})

    target_pos = objects[target_idx]["pos"]
    # Trayectoria GT: arc desde origen al target con bump
    start = np.array([0.0, 0.0, 0.7])
    traj = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
    for h in range(HORIZON):
        alpha = h / (HORIZON - 1)
        pos = start * (1 - alpha) + target_pos * alpha
        pos[2] += 0.1 * np.sin(np.pi * alpha)
        traj[h, :3] = pos
        traj[h, 3:7] = [0.0, 0.0, 0.0, 1.0]

    return {
        "objects": objects, "n_obj": n_obj, "target_idx": target_idx,
        "target_pos": target_pos, "text": text, "traj_gt": traj,
    }


def scene_to_tensors(scene, max_obj=MAX_OBJ):
    """Convierte escena a (attrs_padded, mask, target_pos). attrs_padded: (max_obj, ATTR_DIM)."""
    attrs = np.zeros((max_obj, ATTR_DIM), dtype=np.float32)
    positions = np.zeros((max_obj, 3), dtype=np.float32)
    mask = np.zeros(max_obj, dtype=np.float32)
    for i, obj in enumerate(scene["objects"]):
        attrs[i, :3] = COLOR_RGB[obj["color"]]
        attrs[i, 3:7] = SHAPE_ENC[obj["shape"]]
        positions[i] = obj["pos"]
        mask[i] = 1.0
    return attrs, positions, mask, scene["target_idx"]


def assemble_cond(selected_pos, clip_proj, positions, attrs_flat):
    """Construye el cond del UNet:
    [0..3]    selected_pos
    [3..18]   positions of all 5 slots (5x3=15)
    [18..50]  clip projection (32)
    [50..64]  padding (incluye summary attrs)
    """
    B = selected_pos.shape[0]
    cond = torch.zeros(B, COND_DIM, device=selected_pos.device, dtype=selected_pos.dtype)
    cond[:, :3] = selected_pos
    cond[:, 3:18] = positions.view(B, -1)[:, :15]
    cond[:, 18:50] = clip_proj
    # 50..64: agregamos un attr summary del target (max-attr over slots)
    summary = (attrs_flat * 1.0).max(dim=1).values  # (B, ATTR_DIM)
    cond[:, 50:50+ATTR_DIM] = summary
    return cond


class SceneDataset(Dataset):
    def __init__(self, scenes, clip_embs):
        self.scenes = scenes
        self.clip_embs = clip_embs

    def __len__(self):
        return len(self.scenes)

    def __getitem__(self, i):
        s = self.scenes[i]
        attrs, positions, mask, target_idx = scene_to_tensors(s)
        return {
            "traj": torch.tensor(s["traj_gt"]),
            "attrs": torch.tensor(attrs),
            "positions": torch.tensor(positions),
            "mask": torch.tensor(mask),
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
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
            xb = b["traj"].to(device)
            attrs = b["attrs"].to(device)
            pos = b["positions"].to(device)
            mask = b["mask"].to(device)
            tgt = b["target_idx"].to(device)
            ce = b["clip_emb"].to(device)
            B = xb.shape[0]

            gates = gate(ce, attrs, mask)                          # (B, MAX_OBJ)
            selected_pos = (gates.unsqueeze(-1) * pos).sum(dim=1)  # (B, 3)
            clip_proj = projector(ce)
            cb = assemble_cond(selected_pos, clip_proj, pos, attrs)

            t = torch.randint(0, TIMESTEPS, (B,), device=device)
            eps = torch.randn_like(xb)
            ab_t = alpha_bar[t].view(-1, 1, 1)
            x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
            eps_pred = model(x_noisy, t, cb)
            loss_diff = F.mse_loss(eps_pred, eps)
            # Aux: cross-entropy sobre el gate (masked)
            log_gates = (gates + 1e-9).log()
            loss_cls = F.nll_loss(log_gates, tgt)
            loss = loss_diff + 0.5 * loss_cls

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            sum_diff += loss_diff.item() * B
            chosen = gates.argmax(dim=-1)
            sum_gacc += (chosen == tgt).float().mean().item() * B
            n += B
        train_losses.append(sum_diff / n)
        gate_train = sum_gacc / n

        model.eval(); projector.eval(); gate.eval()
        with torch.no_grad():
            sum_diff, sum_gacc, n = 0, 0, 0
            for b in val_loader:
                xb = b["traj"].to(device); attrs = b["attrs"].to(device)
                pos = b["positions"].to(device); mask = b["mask"].to(device)
                tgt = b["target_idx"].to(device); ce = b["clip_emb"].to(device)
                B = xb.shape[0]
                gates = gate(ce, attrs, mask)
                selected_pos = (gates.unsqueeze(-1) * pos).sum(dim=1)
                clip_proj = projector(ce)
                cb = assemble_cond(selected_pos, clip_proj, pos, attrs)
                t = torch.randint(0, TIMESTEPS, (B,), device=device)
                eps = torch.randn_like(xb)
                ab_t = alpha_bar[t].view(-1, 1, 1)
                x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
                eps_pred = model(x_noisy, t, cb)
                sum_diff += F.mse_loss(eps_pred, eps).item() * B
                chosen = gates.argmax(dim=-1)
                sum_gacc += (chosen == tgt).float().mean().item() * B
                n += B
            val_loss = sum_diff / n
            gate_val = sum_gacc / n
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
                  f"val={val_loss:.5f} | gate train/val={gate_train:.1%}/{gate_val:.1%}")

    elapsed = (time.time() - t0) / 60
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        projector.load_state_dict(best_state["projector"])
        gate.load_state_dict(best_state["gate"])
    print(f"  Training: {elapsed:.1f} min")
    return train_losses, val_losses, best_val


def evaluate(model, projector, gate, scheduler, scenes, clip_embs, device, batch_size=64):
    """Accuracy global + por N de objetos."""
    n = len(scenes)
    trajs_pred = np.empty((n, HORIZON, ACTION_DIM), dtype=np.float32)
    chosen_indices = np.empty(n, dtype=np.int64)
    model.eval(); projector.eval(); gate.eval()
    t0 = time.time()
    with torch.no_grad():
        for i in range(0, n, batch_size):
            batch_scenes = scenes[i:i+batch_size]
            attrs = torch.tensor(np.stack([scene_to_tensors(s)[0] for s in batch_scenes]),
                                    device=device)
            pos = torch.tensor(np.stack([scene_to_tensors(s)[1] for s in batch_scenes]),
                                  device=device)
            mask = torch.tensor(np.stack([scene_to_tensors(s)[2] for s in batch_scenes]),
                                   device=device)
            ce = torch.tensor(clip_embs[i:i+batch_size], device=device)
            gates = gate(ce, attrs, mask)
            selected_pos = (gates.unsqueeze(-1) * pos).sum(dim=1)
            clip_proj = projector(ce)
            cb = assemble_cond(selected_pos, clip_proj, pos, attrs)
            x = ddim_sample(model, scheduler, cb, device, 25)
            trajs_pred[i:i+batch_size] = x
            chosen_indices[i:i+batch_size] = gates.argmax(dim=-1).cpu().numpy()
    latency = (time.time() - t0) * 1000 / n

    # Stats por N de objetos
    by_n = {}
    correct_global = 0
    for i, s in enumerate(scenes):
        N = s["n_obj"]
        ok = chosen_indices[i] == s["target_idx"]
        if N not in by_n:
            by_n[N] = {"correct": 0, "total": 0}
        by_n[N]["correct"] += int(ok)
        by_n[N]["total"] += 1
        if ok:
            correct_global += 1

    return {
        "selection_accuracy_global": correct_global / n,
        "accuracy_by_n_objects": {str(N): r["correct"] / r["total"]
                                     for N, r in sorted(by_n.items())},
        "n_by_n_objects": {str(N): r["total"] for N, r in sorted(by_n.items())},
        "latency_ms_per_traj_ddim25": latency,
        "n_evaluated": n,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=6000)
    ap.add_argument("--n-val", type=int, default=1500)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print(f"[exp20] device={device} n_train={args.n_train} n_val={args.n_val} "
          f"epochs={args.epochs}, max_obj={MAX_OBJ}")

    print("\n[1/4] CLIP text encoder...")
    clip_encoder = CLIPTextEncoder(device)

    print(f"\n[2/4] Generando escenas con N={2}..{MAX_OBJ} objetos...")
    scenes_train, ce_train = make_dataset(args.n_train, SEED, clip_encoder)
    scenes_val, ce_val = make_dataset(args.n_val, SEED + 1, clip_encoder)
    n_dist_train = np.bincount([s["n_obj"] for s in scenes_train], minlength=MAX_OBJ + 1)
    print(f"  Train N distribution: {dict(enumerate(n_dist_train))}")
    print("  Ejemplos:")
    for s in scenes_train[:3]:
        print(f"    text='{s['text']}' | n_obj={s['n_obj']} | target={s['target_idx']}")

    print("\n[3/4] Entrenando modelo multi-objeto...")
    model = ConditionalUNet1D(action_dim=ACTION_DIM, horizon=HORIZON,
                                 cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM).to(device)
    projector = CLIPProjector(512, 32).to(device)
    gate = MultiObjectGate(clip_dim=512, attr_dim=ATTR_DIM, hidden=128).to(device)

    train_ds = SceneDataset(scenes_train, ce_train)
    val_ds = SceneDataset(scenes_val, ce_val)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    scheduler = SimpleDDPMScheduler(num_timesteps=TIMESTEPS)
    train_losses, val_losses, best_val = train_full(
        model, projector, gate, train_loader, val_loader, scheduler, device,
        n_epochs=args.epochs)

    print("\n[4/4] Evaluando accuracy global + por N...")
    metrics = evaluate(model, projector, gate, scheduler, scenes_val, ce_val, device,
                        batch_size=args.batch_size)

    criteria = {"acc_min_global": 0.75, "acc_min_per_n": 0.65}
    pass_flags = {
        "global_above": metrics["selection_accuracy_global"] >= criteria["acc_min_global"],
        "all_n_above": all(a >= criteria["acc_min_per_n"]
                              for a in metrics["accuracy_by_n_objects"].values()),
    }
    all_pass = all(pass_flags.values())

    results = {
        "config": {"n_train": args.n_train, "n_val": args.n_val,
                    "epochs": len(train_losses), "batch_size": args.batch_size,
                    "device": device, "seed": SEED, "max_obj": MAX_OBJ,
                    "colors": COLORS, "shapes": SHAPES},
        "training": {"best_val_loss": best_val,
                      "train_losses_last5": train_losses[-5:],
                      "val_losses_last5": val_losses[-5:]},
        "evaluation": metrics,
        "criteria": criteria,
        "pass": pass_flags,
        "all_criteria_pass": all_pass,
        "sample_scenes": [
            {"text": s["text"], "n_obj": s["n_obj"], "target_idx": s["target_idx"],
              "objects": [f"{o['color']}_{o['shape']}" for o in s["objects"]]}
            for s in scenes_val[:8]],
    }

    print("\n" + "=" * 60)
    print(f"  GLOBAL selection acc: {metrics['selection_accuracy_global']:.1%}  "
          f"({'PASA' if pass_flags['global_above'] else 'FALLA'})")
    print("  Por N de objetos:")
    for N, acc in sorted(metrics["accuracy_by_n_objects"].items()):
        n = metrics["n_by_n_objects"][N]
        flag = "✓" if acc >= criteria["acc_min_per_n"] else "✗"
        print(f"    {flag} N={N}: {acc:.1%} (n={n})")
    print(f"  Latencia DDIM-25: {metrics['latency_ms_per_traj_ddim25']:.2f} ms/traj")
    print(f"  Decision: {'ALL PASS' if all_pass else 'PARTIAL/FAIL'}")
    print("=" * 60)

    with open(OUTPUT_RESULTS / "exp20_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {OUTPUT_RESULTS / 'exp20_results.json'}")

    if all_pass:
        torch.save({
            "model_state_dict": model.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "gate_state_dict": gate.state_dict(),
            "config": {"horizon": HORIZON, "action_dim": ACTION_DIM,
                        "cond_dim": COND_DIM, "hidden_dim": HIDDEN_DIM,
                        "clip_model": "openai/clip-vit-base-patch32",
                        "max_obj": MAX_OBJ, "colors": COLORS, "shapes": SHAPES,
                        "is_vla_lite_multi_object": True},
            "metrics": metrics,
        }, OUTPUT_MODEL)
        print(f"[OK] {OUTPUT_MODEL}")


if __name__ == "__main__":
    main()
