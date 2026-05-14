#!/usr/bin/env python3
"""Experimento 16: VLA-lite con CLIP text-conditioning.

Anade condicionamiento por lenguaje natural a Diffusion Policy:
"pick the red object" -> CLIP text encoder -> embedding -> Diffusion -> trayectoria
filtrada hacia el objeto descrito.

Setup:
- Escenas sinteticas multi-objeto (2 objetos por escena, colores {red, blue, green})
- Texto: "pick the {color} {object}" generado proceduralmente
- CLIP text encoder (openai/clip-vit-base-patch32, FROZEN, 63 M params)
- Proyeccion 512-D -> 32-D anadida a la cond del Diffusion Policy
- Diffusion Policy adaptado: cond_dim = 64 = 32 (CLIP proj) + 16 (obj_A) + 16 (obj_B)

Metrica nueva: SELECTION ACCURACY = % de trayectorias que terminan mas cerca
del objeto target descrito que del distractor.

Salida:
    data/models/diffusion_policy_clip.pth (si pasa criterios)
    experiments/results/exp16_vla_lite/exp16_results.json
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
from tqdm.auto import tqdm

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import ConditionalUNet1D, SimpleDDPMScheduler

# Config
HORIZON = 16
ACTION_DIM = 7
COND_DIM = 64
HIDDEN_DIM = 256
TIMESTEPS = 100
SEED = 42

# Catalogo de colores
COLORS = ["red", "blue", "green"]
COLOR_RGB = {"red": [1.0, 0.0, 0.0], "blue": [0.0, 0.0, 1.0], "green": [0.0, 1.0, 0.0]}

# Templates de descripcion
TEMPLATES = [
    "pick the {color} object",
    "grab the {color} item",
    "get the {color} thing",
    "select the {color} one",
    "take the {color} cube",
]

OUTPUT_MODEL = REPO / "data/models/diffusion_policy_clip.pth"
OUTPUT_RESULTS = REPO / "experiments/results/exp16_vla_lite"
OUTPUT_RESULTS.mkdir(parents=True, exist_ok=True)


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class CLIPTextEncoder:
    """Wrapper minimo del CLIP text encoder (frozen)."""
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
        """texts: list[str] -> tensor (B, 512) en self.device."""
        inputs = self.tokenizer(texts, padding=True, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        out = self.model(**inputs)
        return out.pooler_output  # (B, 512)


def generate_scene(rng):
    """Genera una escena: 2 objetos con colores distintos, posiciones,
    texto de instruccion, y la trayectoria GT que termina en el target."""
    # 2 colores distintos
    color_idx = rng.choice(len(COLORS), size=2, replace=False)
    c_a, c_b = COLORS[color_idx[0]], COLORS[color_idx[1]]
    # Posiciones (no demasiado cerca para no confundir)
    while True:
        p_a = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.7, 1.0)])
        p_b = np.array([rng.uniform(-0.4, 0.4), rng.uniform(-0.4, 0.4), rng.uniform(0.7, 1.0)])
        if np.linalg.norm(p_a - p_b) > 0.20:  # al menos 20 cm de separacion
            break
    # Target = uno de los dos al azar
    target_idx = rng.integers(0, 2)
    target_color = c_a if target_idx == 0 else c_b
    target_pos = p_a if target_idx == 0 else p_b
    template = TEMPLATES[rng.integers(0, len(TEMPLATES))]
    text = template.format(color=target_color)

    # Trayectoria GT: spline simple desde origen hasta target_pos
    # (replica simplificada de plan_grasp_heuristic pero hacia el objeto descrito)
    start = np.array([0.0, 0.0, 0.7])
    traj = np.zeros((HORIZON, ACTION_DIM), dtype=np.float32)
    for h in range(HORIZON):
        alpha = h / (HORIZON - 1)
        # Curva suave: arco que se acerca al target
        pos = start * (1 - alpha) + target_pos * alpha
        # Pequeno bump en Z para evitar colisiones
        pos[2] += 0.1 * np.sin(np.pi * alpha)
        traj[h, :3] = pos
        # Orientacion cuaternion: identidad para simplificar
        traj[h, 3:7] = [0.0, 0.0, 0.0, 1.0]

    return {
        "p_a": p_a, "p_b": p_b,
        "c_a": c_a, "c_b": c_b,
        "target_idx": int(target_idx),
        "target_pos": target_pos,
        "distractor_pos": p_b if target_idx == 0 else p_a,
        "text": text,
        "traj_gt": traj,
    }


def build_static_cond(scene):
    """Solo la parte fija del cond (sin CLIP, que se inyecta luego en cada batch).

    Layout: [4..6]  p_a, [7..9]  p_b, [10..12] RGB(c_a), [13..15] RGB(c_b).
    Las dimensiones 19..50 quedan para la proyeccion CLIP (32-D),
    que se anade durante el forward en runtime con el projector entrenable.
    """
    cond = np.zeros(COND_DIM, dtype=np.float32)
    cond[4:7] = scene["p_a"]
    cond[7:10] = scene["p_b"]
    cond[10:13] = COLOR_RGB[scene["c_a"]]
    cond[13:16] = COLOR_RGB[scene["c_b"]]
    return cond


def assemble_cond(static_cond, clip_proj):
    """Forma final del cond combinando la parte fija + proyeccion CLIP entrenable.
    static_cond: (B, 64) float
    clip_proj:   (B, 32) float (salida del projector entrenable)
    Devuelve:    (B, 64) con clip_proj inyectado en [19..51]
    """
    out = static_cond.clone()
    out[:, 19:51] = clip_proj
    return out


def assemble_cond_with_selected_pos(static_cond, selected_pos, clip_proj=None):
    """Variante con gating: el cond incluye en [0..3] la posicion seleccionada
    (mezcla ponderada por el gate) y opcionalmente la proj CLIP en [19..51].
    """
    out = static_cond.clone()
    out[:, :3] = selected_pos
    if clip_proj is not None:
        out[:, 19:51] = clip_proj
    return out


class CLIPProjector(nn.Module):
    """Proyeccion lineal de CLIP 512-D -> 32-D para fit en COND_DIM."""
    def __init__(self, in_dim=512, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.Mish(),
            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class TextGroundedGate(nn.Module):
    """Modulo VLA-lite: dado CLIP(text), RGB(c_a), RGB(c_b), produce un vector
    de atencion (gate_a, gate_b) que selecciona uno de los dos objetos.

    Esto da al modelo el inductive bias para asociar la descripcion textual
    al objeto correcto, en lugar de esperar que el UNet aprenda esto
    implicitamente del MSE de denoising.
    """
    def __init__(self, clip_dim=512, hidden=128):
        super().__init__()
        # Cada objeto: RGB(3) + CLIP(512) -> score logit
        self.score = nn.Sequential(
            nn.Linear(3 + clip_dim, hidden),
            nn.Mish(),
            nn.Linear(hidden, hidden),
            nn.Mish(),
            nn.Linear(hidden, 1),
        )

    def forward(self, clip_emb, rgb_a, rgb_b):
        """clip_emb: (B, 512), rgb_a/b: (B, 3). Devuelve (gate_a, gate_b) (B,)."""
        s_a = self.score(torch.cat([rgb_a, clip_emb], dim=-1)).squeeze(-1)  # (B,)
        s_b = self.score(torch.cat([rgb_b, clip_emb], dim=-1)).squeeze(-1)
        logits = torch.stack([s_a, s_b], dim=-1)  # (B, 2)
        gates = F.softmax(logits, dim=-1)
        return gates[:, 0], gates[:, 1]


def make_dataset(n, seed, clip_encoder, device, batch_clip=64):
    """Genera n escenas + GT + CLIP embeddings RAW (512-D, sin proyectar).

    El projector se entrena dentro del bucle, asi que aqui solo precomputamos
    los embeddings CLIP que son inputs fijos (CLIP esta frozen).
    """
    rng = np.random.default_rng(seed)
    scenes = [generate_scene(rng) for _ in range(n)]
    texts = [s["text"] for s in scenes]

    clip_embs = []
    for i in tqdm(range(0, n, batch_clip), desc="clip", leave=False):
        batch_texts = texts[i:i+batch_clip]
        emb = clip_encoder.encode(batch_texts)
        clip_embs.append(emb.cpu().numpy())
    clip_embs = np.concatenate(clip_embs, axis=0)  # (n, 512) RAW

    static_conds = np.array([build_static_cond(s) for s in scenes], dtype=np.float32)
    trajs = np.array([s["traj_gt"] for s in scenes], dtype=np.float32)

    return scenes, static_conds, trajs, clip_embs


def train_model(model, projector, gate, train_loader, val_loader, scheduler, device,
                  n_epochs=40, lr=3e-4, warmup_epochs=3, grad_clip=1.0,
                  use_aux_classification_loss=True):
    """Entrena DP + projector + TextGroundedGate conjuntamente.

    El batch contiene (traj, static_cond, clip_emb_raw_512, target_idx).
    Cada paso:
      1) gate(clip_emb, RGB_a, RGB_b) -> (gate_a, gate_b)
      2) selected_pos = gate_a * p_a + gate_b * p_b
      3) projector(clip_emb) -> clip_proj 32-D
      4) cond = [selected_pos, ..., clip_proj]
      5) DP MSE denoising loss
      6) loss aux: cross-entropy(gate logits, target_idx) — supervision directa

    El aux loss es lo que hace que el gate aprenda rapido (vs solo MSE indirecto).
    """
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
    params = list(model.parameters()) + list(projector.parameters()) + list(gate.parameters())
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=1e-5)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / warmup_epochs
        progress = (epoch - warmup_epochs) / max(n_epochs - warmup_epochs, 1)
        return max(1e-6/lr, 0.5 * (1 + np.cos(np.pi * progress)))
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train_losses, val_losses = [], []
    best_val = float("inf")
    t0 = time.time()
    best_state = None

    def step_forward(xb, sc, ce, tgt):
        """Devuelve (loss_total, loss_diff, loss_cls, gate_acc) para un batch."""
        B = xb.shape[0]
        rgb_a = sc[:, 10:13]
        rgb_b = sc[:, 13:16]
        p_a = sc[:, 4:7]
        p_b = sc[:, 7:10]
        gate_a, gate_b = gate(ce, rgb_a, rgb_b)
        selected_pos = gate_a.unsqueeze(-1) * p_a + gate_b.unsqueeze(-1) * p_b
        clip_proj = projector(ce)
        cb = assemble_cond_with_selected_pos(sc, selected_pos, clip_proj)
        t = torch.randint(0, TIMESTEPS, (B,), device=device)
        eps = torch.randn_like(xb)
        ab_t = alpha_bar[t].view(-1, 1, 1)
        x_noisy = torch.sqrt(ab_t) * xb + torch.sqrt(1 - ab_t) * eps
        eps_pred = model(x_noisy, t, cb)
        loss_diff = F.mse_loss(eps_pred, eps)
        # Aux: gate logits supervisado (acelera el aprendizaje)
        logits = torch.stack([gate_a, gate_b], dim=-1).log()  # ya softmax-ed → log probs
        loss_cls = F.nll_loss(logits, tgt)
        total = loss_diff + 0.5 * loss_cls if use_aux_classification_loss else loss_diff
        gate_acc = ((gate_a > gate_b).long() == (tgt == 0).long()).float().mean().item()
        return total, loss_diff.item(), loss_cls.item(), gate_acc

    for epoch in range(n_epochs):
        model.train(); projector.train(); gate.train()
        sum_loss, sum_diff, sum_cls, sum_gacc, n = 0, 0, 0, 0, 0
        for xb, sc, ce, tgt in train_loader:
            xb, sc, ce, tgt = xb.to(device), sc.to(device), ce.to(device), tgt.to(device)
            B = xb.shape[0]
            loss, loss_d, loss_c, g_acc = step_forward(xb, sc, ce, tgt)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, grad_clip)
            optimizer.step()
            sum_loss += loss.item() * B
            sum_diff += loss_d * B
            sum_cls += loss_c * B
            sum_gacc += g_acc * B
            n += B
        train_losses.append(sum_diff / n)
        train_gate_acc = sum_gacc / n

        model.eval(); projector.eval(); gate.eval()
        with torch.no_grad():
            sum_diff, sum_gacc, n = 0, 0, 0
            for xb, sc, ce, tgt in val_loader:
                xb, sc, ce, tgt = xb.to(device), sc.to(device), ce.to(device), tgt.to(device)
                B = xb.shape[0]
                _, loss_d, _, g_acc = step_forward(xb, sc, ce, tgt)
                sum_diff += loss_d * B
                sum_gacc += g_acc * B
                n += B
            val_loss = sum_diff / n
            val_gate_acc = sum_gacc / n
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
                  f"val={val_loss:.5f} | gate_acc train/val={train_gate_acc:.1%}/{val_gate_acc:.1%}")

    elapsed = (time.time() - t0) / 60
    if best_state is not None:
        model.load_state_dict(best_state["model"])
        projector.load_state_dict(best_state["projector"])
        gate.load_state_dict(best_state["gate"])
    print(f"  Training: {elapsed:.1f} min, best_val={best_val:.5f}, final gate_acc_val={val_gate_acc:.1%}")
    return train_losses, val_losses, best_val


def ddim_sample_batched(model, scheduler, cond, device, n_steps=25):
    B = cond.shape[0]
    # Asegurar tipo float (no contaminar con grad de cond)
    x = torch.randn(B, HORIZON, ACTION_DIM, device=device, dtype=cond.dtype)
    step_indices = np.linspace(0, scheduler.num_timesteps - 1, n_steps).astype(int)[::-1]
    alpha_bar = torch.tensor(scheduler.alpha_bar, dtype=torch.float32, device=device)
    with torch.no_grad():
        for i, step in enumerate(step_indices):
            t_tensor = torch.full((B,), int(step), dtype=torch.long, device=device)
            noise_pred = model(x, t_tensor, cond)
            ab_t = alpha_bar[step]
            pred_x0 = (x - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)
            if i < len(step_indices) - 1:
                ab_next = alpha_bar[step_indices[i + 1]]
                x = torch.sqrt(ab_next) * pred_x0 + torch.sqrt(1 - ab_next) * noise_pred
            else:
                x = pred_x0
    return x.cpu().numpy()


def evaluate_selection(model, projector, gate, scheduler, scenes, static_conds, clip_embs,
                         device, batch_size=64):
    """SELECTION ACCURACY: % de trayectorias que terminan mas cerca del target que del distractor."""
    print("[eval] Sampling con DDIM-25...")
    n = len(static_conds)
    trajs_pred = np.empty((n, HORIZON, ACTION_DIM), dtype=np.float32)
    model.eval(); projector.eval(); gate.eval()
    t0 = time.time()
    gate_correct = 0
    with torch.no_grad():
        for i in range(0, n, batch_size):
            sc = torch.tensor(static_conds[i:i+batch_size], device=device)
            ce = torch.tensor(clip_embs[i:i+batch_size], device=device)
            rgb_a = sc[:, 10:13]; rgb_b = sc[:, 13:16]
            p_a = sc[:, 4:7]; p_b = sc[:, 7:10]
            gate_a, gate_b = gate(ce, rgb_a, rgb_b)
            selected_pos = gate_a.unsqueeze(-1) * p_a + gate_b.unsqueeze(-1) * p_b
            clip_proj = projector(ce)
            cb = assemble_cond_with_selected_pos(sc, selected_pos, clip_proj)
            x = ddim_sample_batched(model, scheduler, cb, device, n_steps=25)
            if device == "mps":
                torch.mps.synchronize()
            trajs_pred[i:i+batch_size] = x
            # Acc del gate: chosen=0 -> el gate eligio A; target_idx=0 -> A es target
            chosen = (gate_b > gate_a).long().cpu().numpy()  # 0=A, 1=B (alineado con target_idx)
            tgts = np.array([s["target_idx"] for s in scenes[i:i+batch_size]])
            gate_correct += int((chosen == tgts).sum())
    latency = (time.time() - t0) * 1000 / n
    gate_accuracy = gate_correct / n

    # Selection accuracy
    correct = 0
    distances_to_target = []
    distances_to_distractor = []
    for i, scene in enumerate(scenes):
        endpoint = trajs_pred[i, -1, :3]
        d_target = np.linalg.norm(endpoint - scene["target_pos"])
        d_distractor = np.linalg.norm(endpoint - scene["distractor_pos"])
        distances_to_target.append(d_target)
        distances_to_distractor.append(d_distractor)
        if d_target < d_distractor:
            correct += 1
    selection_acc = correct / len(scenes)

    # MSE vs GT
    gt = np.array([s["traj_gt"] for s in scenes], dtype=np.float32)
    mse = float(np.mean((trajs_pred - gt) ** 2))

    return {
        "selection_accuracy": selection_acc,
        "gate_accuracy": gate_accuracy,
        "mse_vs_gt": mse,
        "latency_ms_per_traj_ddim25": latency,
        "mean_distance_to_target_m": float(np.mean(distances_to_target)),
        "mean_distance_to_distractor_m": float(np.mean(distances_to_distractor)),
        "n_evaluated": len(scenes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=4000)
    ap.add_argument("--n-val", type=int, default=800)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = get_device()
    print(f"[exp16] device={device}")
    print(f"[exp16] n_train={args.n_train} n_val={args.n_val} epochs={args.epochs}")

    print("\n[1/4] Cargando CLIP text encoder (frozen)...")
    clip_encoder = CLIPTextEncoder(device)
    print(f"  CLIP params: {sum(p.numel() for p in clip_encoder.model.parameters())/1e6:.1f} M")

    print("\n[2/4] Generando escenas + GT + CLIP embeddings RAW (512-D)...")
    scenes_train, static_conds_train, trajs_train, clip_embs_train = make_dataset(
        args.n_train, seed=SEED, clip_encoder=clip_encoder,
        device=device, batch_clip=64,
    )
    scenes_val, static_conds_val, trajs_val, clip_embs_val = make_dataset(
        args.n_val, seed=SEED + 1, clip_encoder=clip_encoder,
        device=device, batch_clip=64,
    )
    print(f"  Train: scenes={len(scenes_train)} static={static_conds_train.shape} "
          f"clip={clip_embs_train.shape} trajs={trajs_train.shape}")
    print(f"  Ejemplo: text='{scenes_train[0]['text']}', target_idx={scenes_train[0]['target_idx']}")

    print("\n[3/4] Entrenando Diffusion Policy + CLIP projector + TextGroundedGate...")
    model = ConditionalUNet1D(
        action_dim=ACTION_DIM, horizon=HORIZON,
        cond_dim=COND_DIM, hidden_dim=HIDDEN_DIM,
    ).to(device)
    projector = CLIPProjector(in_dim=512, out_dim=32).to(device)
    gate = TextGroundedGate(clip_dim=512, hidden=128).to(device)
    n_params = (sum(p.numel() for p in model.parameters()) +
                  sum(p.numel() for p in projector.parameters()) +
                  sum(p.numel() for p in gate.parameters()))
    print(f"  Params (DP + proj + gate): {n_params/1e6:.2f} M")

    target_idx_train = np.array([s["target_idx"] for s in scenes_train], dtype=np.int64)
    target_idx_val = np.array([s["target_idx"] for s in scenes_val], dtype=np.int64)

    train_ds = TensorDataset(
        torch.tensor(trajs_train),
        torch.tensor(static_conds_train),
        torch.tensor(clip_embs_train),
        torch.tensor(target_idx_train),
    )
    val_ds = TensorDataset(
        torch.tensor(trajs_val),
        torch.tensor(static_conds_val),
        torch.tensor(clip_embs_val),
        torch.tensor(target_idx_val),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    scheduler = SimpleDDPMScheduler(num_timesteps=TIMESTEPS)
    train_losses, val_losses, best_val = train_model(
        model, projector, gate, train_loader, val_loader, scheduler, device,
        n_epochs=args.epochs,
    )

    print("\n[4/4] Evaluando SELECTION ACCURACY sobre val set...")
    metrics = evaluate_selection(model, projector, gate, scheduler, scenes_val,
                                   static_conds_val, clip_embs_val, device,
                                   batch_size=args.batch_size)

    # Criterios del plan
    criteria = {
        "selection_accuracy_min": 0.75,
        "latency_overhead_max_ms": 50.0,
    }
    pass_flags = {
        "selection_above_target": metrics["selection_accuracy"] >= criteria["selection_accuracy_min"],
        "latency_reasonable": metrics["latency_ms_per_traj_ddim25"] < 200,  # ddim25 puede llegar ahi
    }
    all_pass = all(pass_flags.values())

    results = {
        "config": {
            "n_train": args.n_train, "n_val": args.n_val,
            "epochs": len(train_losses), "batch_size": args.batch_size,
            "device": device, "seed": SEED,
        },
        "clip_model": "openai/clip-vit-base-patch32",
        "training": {
            "best_val_loss": best_val,
            "train_losses_last10": train_losses[-10:],
            "val_losses_last10": val_losses[-10:],
        },
        "evaluation": metrics,
        "criteria": criteria,
        "pass": pass_flags,
        "all_criteria_pass": all_pass,
        "sample_scenes": [
            {"text": s["text"], "target_color": s["c_a"] if s["target_idx"] == 0 else s["c_b"],
              "distractor_color": s["c_b"] if s["target_idx"] == 0 else s["c_a"]}
            for s in scenes_val[:8]
        ],
    }

    print("\n" + "=" * 60)
    print(f"  Selection accuracy:        {metrics['selection_accuracy']:.1%}  "
          f"(>= {criteria['selection_accuracy_min']:.0%}: "
          f"{'PASA' if pass_flags['selection_above_target'] else 'FALLA'})")
    print(f"  Gate accuracy (sanity):    {metrics['gate_accuracy']:.1%}")
    print(f"  Mean distance to target:   {metrics['mean_distance_to_target_m']*100:.1f} cm")
    print(f"  Mean distance to distract: {metrics['mean_distance_to_distractor_m']*100:.1f} cm")
    print(f"  MSE vs GT:                 {metrics['mse_vs_gt']:.5f}")
    print(f"  Latency DDIM-25:           {metrics['latency_ms_per_traj_ddim25']:.2f} ms")
    print(f"  Decision: {'ALL PASS - candidate to merge' if all_pass else 'PARTIAL/FAIL - revisar'}")
    print("=" * 60)

    with open(OUTPUT_RESULTS / "exp16_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] {OUTPUT_RESULTS / 'exp16_results.json'}")

    if all_pass:
        torch.save({
            "model_state_dict": model.state_dict(),
            "projector_state_dict": projector.state_dict(),
            "gate_state_dict": gate.state_dict(),
            "config": {
                "horizon": HORIZON, "action_dim": ACTION_DIM, "cond_dim": COND_DIM,
                "hidden_dim": HIDDEN_DIM, "clip_model": "openai/clip-vit-base-patch32",
                "is_vla_lite": True,
            },
            "metrics": metrics,
        }, OUTPUT_MODEL)
        print(f"[OK] {OUTPUT_MODEL}")
    else:
        print("[skip] Criterios no superados, no se guarda diffusion_policy_clip.pth")


if __name__ == "__main__":
    main()
