#!/usr/bin/env python3
"""DPPO Phase A — RL fine-tune sobre DP v5 en CoppeliaSim (PoC).

NOTA HONESTA: este script implementa self-imitation learning weighted by
advantage (subset de DPPO). Sirve para validar el loop end-to-end con
shaped + binary reward. PPO completo con ratio re-evaluado va en Phase B.

Uso (CoppeliaSim running on :23000):
    python experiments/train_dppo_coppeliasim.py --episodes 500
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.planning.diffusion_policy import DiffusionGraspPlanner
from src.planning.visual_encoder import ResNet18RGBDEncoder
from src.simulation.coppeliasim_bridge import CoppeliaSimBridge
from src.rl.dppo_agent import DPPOAgent
from src.rl.replay_buffer import Episode, EpisodeBuffer
from src.rl.reward_fn import compute_terminal_reward
from src.rl.value_net import ValueNet
from experiments.run_pick_with_diffusion import pick_with_dp
from experiments.collect_diffusion_dataset import _capture_rgbd_for_pose
from experiments.eval_diffusion_iter2_sim import sample_pose_eval

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("dppo_phaseA")

SCENE = REPO / "data" / "scenes" / "bin_base.ttt"


def build_cond(planner, encoder, bridge, pose, device):
    """Captura RGB-D + corre encoder + construye cond 64d."""
    rgbd = _capture_rgbd_for_pose(bridge, pose)
    rgbd_t = torch.from_numpy(rgbd).unsqueeze(0).to(device)
    with torch.no_grad():
        visual_emb = encoder(rgbd_t).cpu().numpy()[0]
    cond = planner.encode_observation(pose, visual_emb=visual_emb)
    return cond.squeeze(0).cpu()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=2027)  # ≠ eval seed 2026
    parser.add_argument("--kl-coef", type=float, default=1.0,
                        help="Peso del KD anchor a la referencia v5. 0 = sin anclaje.")
    parser.add_argument("--checkpoint-in", type=Path,
                        default=REPO / "data/models/diffusion_policy_sim_v5.pth")
    parser.add_argument("--checkpoint-out", type=Path,
                        default=REPO / "data/models/diffusion_policy_v6_phaseA.pth")
    args = parser.parse_args()

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    logger.info(f"device: {device}")

    ckpt = torch.load(args.checkpoint_in, map_location=device, weights_only=True)
    hd = ckpt["config"]["hidden_dim"]
    planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device, hidden_dim=hd,
    )
    planner.model.load_state_dict(ckpt["model_state_dict"])
    # Reference model (v5 frozen) — para KD anchor en update
    ref_planner = DiffusionGraspPlanner(
        action_dim=7, horizon=16, n_diffusion_steps=100, device=device, hidden_dim=hd,
    )
    ref_planner.model.load_state_dict(ckpt["model_state_dict"])
    ref_planner.model.eval()

    enc_state = torch.load(
        REPO / "data/models/visual_encoder_iter5.pth", map_location=device, weights_only=True
    )
    encoder = ResNet18RGBDEncoder(out_dim=enc_state["out_dim"]).to(device).eval()
    encoder.load_state_dict(enc_state["state_dict"])

    value_net = ValueNet(cond_dim=64, hidden_dim=32).to(device)
    agent = DPPOAgent(
        planner, value_net, k_last_denoising=4, lr=args.lr,
        kl_coef=args.kl_coef, ref_model=ref_planner.model,
    )
    buffer = EpisodeBuffer()

    rng = np.random.default_rng(args.seed)
    rolling_rewards = []
    update_log = []

    for ep_idx in range(args.episodes):
        pose = sample_pose_eval(rng)
        try:
            with CoppeliaSimBridge() as bridge:
                bridge.load_scene(SCENE)
                cond_storage = build_cond(planner, encoder, bridge, pose, device)
                r = pick_with_dp(planner, pose, bridge, frames_dir=None, visual_encoder=encoder)
        except Exception as e:
            logger.warning(f"[{ep_idx}] sim fail: {e}")
            continue

        terminal_r = compute_terminal_reward(
            r["grasp_plausible"], r["deposit_plausible"], r["ik_converged"],
            distractor_collision=False,
        )
        value = float(value_net(cond_storage.unsqueeze(0).to(device)).item())
        ep = Episode(
            cond=cond_storage,
            actions=torch.tensor(r["waypoints"]),
            rewards=[terminal_r],
            log_probs=torch.zeros(4),
            value=value,
        )
        buffer.add(ep)
        rolling_rewards.append(terminal_r)

        if len(buffer) >= args.batch_size:
            buffer.compute_gae()
            stats = agent.update(buffer.episodes)
            buffer.clear()
            rolling = float(np.mean(rolling_rewards[-args.batch_size:]))
            update_log.append({"ep": ep_idx + 1, "rolling_reward": rolling, **stats})
            logger.info(
                f"ep {ep_idx+1}/{args.episodes}: rolling_R={rolling:.2f} "
                f"pol_loss={stats['policy_loss']:.3f} val_loss={stats['value_loss']:.3f} "
                f"pos={stats['n_positive_episodes']}/{args.batch_size}"
            )

    args.checkpoint_out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": planner.model.state_dict(),
        "value_state_dict": value_net.state_dict(),
        "config": {**ckpt["config"], "phase": "A", "episodes": args.episodes},
    }, args.checkpoint_out)
    log_path = REPO / "experiments/results/dppo_phaseA_log.json"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(json.dumps({
        "episodes": args.episodes,
        "rolling_rewards": rolling_rewards,
        "updates": update_log,
        "final_rolling_R": float(np.mean(rolling_rewards[-args.batch_size:])),
        "initial_rolling_R": float(np.mean(rolling_rewards[:args.batch_size])),
    }, indent=2))
    logger.info(f"ckpt: {args.checkpoint_out}")
    logger.info(f"initial rolling R = {np.mean(rolling_rewards[:args.batch_size]):.2f}")
    logger.info(f"final rolling R   = {np.mean(rolling_rewards[-args.batch_size:]):.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
