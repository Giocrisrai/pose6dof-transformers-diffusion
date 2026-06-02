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
from src.rl.dppo_agent import DPPOAgent, DPPOEpisode
from src.rl.replay_buffer import EpisodeBuffer
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

    from src.simulation.pick_sequence import (
        _move_tcp_via_ik, _setup_ik, set_gripper, setup_robot_control,
    )
    import math
    GRASP_THRESHOLD_M = 0.05
    DEPOSIT_TARGET = [-0.30, -0.30, 0.30]
    DEPOSIT_THRESHOLD_M = 0.30

    def run_episode_with_dppo(pose):
        """Sample con exploration + ejecuta + devuelve (steps, terminal_r, cond)."""
        with CoppeliaSimBridge() as bridge:
            bridge.load_scene(SCENE)
            cond = build_cond(planner, encoder, bridge, pose, device).to(device)
            # Sample con DPPO
            cond_b = cond.unsqueeze(0) if cond.ndim == 1 else cond
            waypoints, steps = agent.sample_action_with_steps(cond_b)

            # Execute igual que pick_with_dp
            setup_robot_control(bridge)
            env, ik_group, target_dummy, ik_joints, simIK = _setup_ik(bridge)
            bridge.set_stepping(True)
            bridge.start_simulation()
            sim = bridge.sim
            obj1 = sim.getObject("/object_1")
            tip_h = sim.getObject("/tip")
            sim.setObjectPosition(obj1, -1, list(pose[:3, 3]))

            cube_pos = sim.getObjectPosition(obj1, -1)
            grasp_idx = next(
                (k for k in range(len(waypoints)) if float(waypoints[k, 6]) < 0.5), 8,
            )
            grasp_wp = waypoints[grasp_idx]
            grasp_proximity_m = math.sqrt(
                sum((cube_pos[i] - float(grasp_wp[i])) ** 2 for i in range(3))
            )
            grasp_plausible = grasp_proximity_m < GRASP_THRESHOLD_M

            counter = [0]
            ik_convergence = []
            prev_gripper = 1.0
            attached = False
            for i, wp in enumerate(waypoints):
                x, y, z, _, _, _, gripper = wp.tolist()
                gripper_open = gripper > 0.5
                prev_open = prev_gripper > 0.5
                if gripper_open != prev_open:
                    set_gripper(bridge, gripper_open)
                    prev_gripper = gripper
                    if not gripper_open and not attached:
                        tip_pos = sim.getObjectPosition(tip_h, -1)
                        sim.setObjectInt32Param(obj1, sim.shapeintparam_respondable, 0)
                        sim.setObjectInt32Param(obj1, sim.shapeintparam_static, 1)
                        sim.setObjectPosition(obj1, -1, tip_pos)
                        sim.setObjectParent(obj1, tip_h, True)
                        attached = True
                    elif gripper_open and attached:
                        sim.setObjectParent(obj1, -1, True)
                        sim.setObjectInt32Param(obj1, sim.shapeintparam_respondable, 1)
                        sim.setObjectInt32Param(obj1, sim.shapeintparam_static, 0)
                        try: sim.resetDynamicObject(obj1)
                        except Exception: pass
                        attached = False
                _move_tcp_via_ik(
                    bridge, env, ik_group, target_dummy, ik_joints, simIK,
                    [x, y, z], None, counter,
                    n_substeps=8, steps_per_substep=2,
                    convergence_tracker=ik_convergence,
                )
            if not attached:
                for _ in range(30): bridge.step()

            cube_end = sim.getObjectPosition(obj1, -1)
            ik_converged = len(ik_convergence) > 0 and all(ik_convergence)
            deposit_error_m = math.sqrt(
                (cube_end[0] - DEPOSIT_TARGET[0]) ** 2 +
                (cube_end[1] - DEPOSIT_TARGET[1]) ** 2
            )
            deposit_plausible = deposit_error_m < DEPOSIT_THRESHOLD_M
            bridge.stop_simulation()
            try: simIK.eraseEnvironment(env)
            except Exception: pass

        terminal_r = compute_terminal_reward(
            grasp_plausible, deposit_plausible, ik_converged,
            distractor_collision=False,
        )
        return steps, terminal_r, cond.cpu(), waypoints

    for ep_idx in range(args.episodes):
        pose = sample_pose_eval(rng)
        try:
            steps, terminal_r, cond_cpu, waypoints = run_episode_with_dppo(pose)
        except Exception as e:
            logger.warning(f"[{ep_idx}] sim fail: {e}")
            continue

        value = float(value_net(cond_cpu.unsqueeze(0).to(device)).item())
        ep = DPPOEpisode(
            cond=cond_cpu,
            actions=torch.tensor(waypoints),
            rewards=[terminal_r],
            log_probs=None,
            value=value,
            denoising_steps=steps,
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
                f"pol={stats['policy_loss']:.3f} val={stats['value_loss']:.3f} "
                f"kl={stats['kl_term']:.4f} clip_f={stats['clip_fraction']:.2f} "
                f"ratio={stats['mean_ratio']:.3f}"
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
