"""E2E real (depth real -> pose -> pick) para varias piezas text-to-CAD."""
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/Users/giocrisraigodoy/Documents/MATLAB/TFM/repo_tfm"); sys.path.insert(0, str(REPO))
import open3d as o3d
import trimesh

from src.simulation.coppeliasim_bridge import CameraConfig, CoppeliaSimBridge
from src.simulation.pick_sequence import run_pick_sequence

RES = (1024, 768)   # sensor de mayor resolución -> nubes más densas
ASSETS = REPO / "experiments/results/exp27_text_to_cad/assets"
VOX = 0.003

def to_pcd(p):
    q = o3d.geometry.PointCloud(); q.points = o3d.utility.Vector3dVector(np.asarray(p)); return q
def fpfh(pc):
    return o3d.pipelines.registration.compute_fpfh_feature(pc, o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*5, max_nn=100))

def backproject(depth, mask, K, T_wc):
    H, W = depth.shape
    us, vs = np.meshgrid(np.arange(W), np.arange(H))
    d = depth.reshape(-1); u = us.reshape(-1); v = vs.reshape(-1)
    sel = mask.reshape(-1) & (d > 0.06) & (d < 1.95)
    d, u, v = d[sel], (W-1-u[sel]), (H-1-v[sel])
    pc = np.stack([(u-K[0,2])/K[0,0]*d, (v-K[1,2])/K[1,1]*d, d], 1)
    return (T_wc[:3,:3] @ pc.T).T + T_wc[:3,3]

def run_one(name, color, do_pick=True):
    tm = trimesh.load(ASSETS / f"{name}.stl"); tm.apply_scale(0.001)
    model = to_pcd(tm.sample(6000)).voxel_down_sample(VOX)
    model.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3, max_nn=30))
    mf = fpfh(model)
    with CoppeliaSimBridge() as b:
        b.set_stepping(True); b.load_scene(REPO/"data/scenes/bin_base.ttt"); sim = b.sim
        cam = b._camera_rgb_handle
        for c in {cam, b._camera_depth_handle}:
            sim.setObjectInt32Param(c, sim.visionintparam_resolution_x, RES[0])
            sim.setObjectInt32Param(c, sim.visionintparam_resolution_y, RES[1])
        b.camera_config = CameraConfig(resolution=RES)
        near = sim.getObjectFloatParam(cam, sim.visionfloatparam_near_clipping)
        far = sim.getObjectFloatParam(cam, sim.visionfloatparam_far_clipping)
        for i in range(1, 6):
            try:
                h = sim.getObject(f"/object_{i}"); sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
                sim.setObjectPosition(h, -1, [5., 5., 0.05+0.1*i])
            except Exception: pass
        obj = sim.importShape(0, str(ASSETS/f"{name}.obj"), 0, 0.0001, 0.001)
        sim.setObjectAlias(obj, name)
        sim.setObjectInt32Param(obj, sim.shapeintparam_static, 0)
        sim.setObjectInt32Param(obj, sim.shapeintparam_respondable, 1)
        sim.computeMassAndInertia(obj, 2700)
        sim.setObjectColor(obj, 0, sim.colorcomponent_ambient_diffuse, color)
        sim.setObjectPosition(obj, -1, [-0.05, -0.22, 0.06])
        sim.setObjectOrientation(obj, -1, [0.15, 0.1, 0.5])
        b.start_simulation()
        for _ in range(120): b.step()
        rgb, depth_b = b.capture_rgbd()
        depth = near + (depth_b - 0.1)/(3.0-0.1)*(far-near)
        K = b.camera_config.K; T_wc = b.get_camera_pose()
        gt = b.get_object_pose(name); settled = sim.getObjectPosition(obj, -1); sq = sim.getObjectQuaternion(obj, -1)
        r_, g_, bl_ = rgb[:,:,0].astype(int), rgb[:,:,1].astype(int), rgb[:,:,2].astype(int)
        # máscara según color dominante
        ci = int(np.argmax(color))
        chans = [r_, g_, bl_]
        mask = (chans[ci] > 90) & (sum(chans) - chans[ci] < chans[ci])
        world = backproject(depth, mask, K, T_wc)
        derr = np.linalg.norm(world.mean(0)[:2] - gt[:3,3][:2]) if len(world) else 9
        b.stop_simulation()
        if len(world) < 80 or derr > 0.06:
            return {"shape": name, "ok": False, "pts": int(len(world)), "err_centroide_cm": round(float(derr)*100,1)}
        obs = to_pcd(world).voxel_down_sample(VOX)
        obs, _ = obs.remove_statistical_outlier(20, 2.0)
        obs.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOX*3, max_nn=30))
        of = fpfh(obs); best = None
        for _ in range(6):
            res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                model, obs, mf, of, True, VOX*1.5,
                o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
                [o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOX*1.5)],
                o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))
            icp = o3d.pipelines.registration.registration_icp(model, obs, VOX*1.2, res.transformation,
                o3d.pipelines.registration.TransformationEstimationPointToPlane())
            if best is None or icp.fitness > best[1]: best = (np.array(icp.transformation), icp.fitness)
        Te = best[0]
        t_err = np.linalg.norm(Te[:3,3]-gt[:3,3])*1000
        dR = Te[:3,:3].T @ gt[:3,:3]
        r_err = np.degrees(np.arccos(np.clip((np.trace(dR)-1)/2, -1, 1)))
        out = {"shape": name, "ok": True, "pts": int(len(obs.points)),
               "err_centroide_cm": round(float(derr)*100,1),
               "t_err_mm": round(float(t_err),1), "R_err_deg": round(float(r_err),1),
               "fitness": round(float(best[1]),2)}
        if do_pick:
            sim.setObjectPosition(obj, -1, settled); sim.setObjectQuaternion(obj, -1, sq)
            pr = run_pick_sequence(b, None, target_object=f"/{name}", pose_override_xyz=Te[:3,3].tolist(),
                                   pose_source="cad_pose_from_real_depth")
            out["proximity_cm"] = round(pr.tip_grasp_proximity_m*100, 1)
            out["grasp_plausible"] = bool(pr.grasp_plausible)
            out["ik_converged"] = bool(pr.ik_converged)
        return out

if __name__ == "__main__":
    jobs = [("hex_nut", [0.1,0.8,0.15]), ("stepped_block", [0.15,0.3,0.9])]
    results = []
    for name, col in jobs:
        r = run_one(name, col, do_pick=True)
        print(json.dumps(r, ensure_ascii=False))
        results.append(r)
    out = REPO/"experiments/results/exp27_text_to_cad/batch_report.json"
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print("saved", out)
