"""Patch idempotente para FoundationPose dentro del contenedor GPU.

Aplica los mismos fallbacks que el notebook 01_foundationpose_eval.ipynb:
- Fallback Python para Utils.erode_depth / bilateral_filter_depth (Warp no está).
- Mock selectivo de open3d (solo voxel_down_sample + transform, que son los usados).
- Fallback Python para mycpp.cluster_poses.

Uso (desde Dockerfile o manualmente):
    FP_DIR=/opt/FoundationPose python3 docker/patch_foundationpose.py

Idempotente: re-ejecutar no duplica el parche (usa marker en Utils.py).
"""
import os
import sys
from pathlib import Path

FP_DIR = Path(os.environ.get("FP_DIR", "/opt/FoundationPose"))
UTILS_PATH = FP_DIR / "Utils.py"
MARKER = "erode_depth_FALLBACK_INJECTED"

UTILS_FALLBACK = '''

# ---- erode_depth_FALLBACK_INJECTED ----
# Fallbacks Python para kernels Warp (no disponibles en contenedores GPU sin Warp CUDA 11).
# Estos mantienen compatibilidad funcional con FoundationPose sin requerir Warp.
import numpy as _fb_np

if 'erode_depth' not in dir():
    def erode_depth(depth, radius=2, depth_diff_thres=0.001,
                    ratio_thres=0.8, zfar=100, device='cuda'):
        import torch as _t
        is_tensor = isinstance(depth, _t.Tensor)
        d = depth.cpu().numpy() if is_tensor else _fb_np.array(depth)
        H, W = d.shape
        valid = (d >= 0.001) & (d < zfar)
        padded = _fb_np.pad(d, radius, mode='constant', constant_values=0)
        pv = _fb_np.pad(valid, radius, mode='constant', constant_values=False)
        bad = _fb_np.zeros((H, W), dtype=_fb_np.float32)
        total = 0
        for dh in range(-radius, radius + 1):
            for dw in range(-radius, radius + 1):
                nh, nw = radius + dh, radius + dw
                nb = padded[nh:nh+H, nw:nw+W]
                nv = pv[nh:nh+H, nw:nw+W]
                bad += ((~nv) | (_fb_np.abs(nb - d) > depth_diff_thres)).astype(_fb_np.float32)
                total += 1
        out = _fb_np.where((bad / total > ratio_thres) | (~valid), 0.0, d).astype(_fb_np.float32)
        return _t.from_numpy(out).to(device) if is_tensor else out

if 'bilateral_filter_depth' not in dir():
    def bilateral_filter_depth(depth, radius=2, zfar=100,
                               sigmaD=2, sigmaR=100000, device='cuda'):
        import torch as _t, cv2
        is_tensor = isinstance(depth, _t.Tensor)
        d = depth.cpu().numpy() if is_tensor else _fb_np.array(depth)
        valid = (d >= 0.001) & (d < zfar)
        out = cv2.bilateralFilter(d.astype(_fb_np.float32),
                                  d=2*radius+1, sigmaColor=float(sigmaR), sigmaSpace=float(sigmaD))
        out = _fb_np.where(valid, out, 0.0).astype(_fb_np.float32)
        return _t.from_numpy(out).to(device) if is_tensor else out
'''


def patch_utils_py():
    """Inyecta fallbacks en Utils.py si no están presentes."""
    if not UTILS_PATH.exists():
        print(f"[ERROR] {UTILS_PATH} no existe. ¿Se clonó FoundationPose en {FP_DIR}?", file=sys.stderr)
        sys.exit(1)

    content = UTILS_PATH.read_text()
    if MARKER in content:
        print(f"[OK] Utils.py ya parcheado ({MARKER} presente)")
        return False

    UTILS_PATH.write_text(content + UTILS_FALLBACK)
    print(f"[OK] Utils.py parcheado: erode_depth, bilateral_filter_depth (Python fallbacks)")
    return True


def install_runtime_shims():
    """Instala mocks de open3d y mycpp en el sitio de Python para que FoundationPose importe.

    Estos shims se activan en import-time a través de un .pth file en site-packages.
    Es equivalente a lo que hace el notebook en runtime con sys.modules.
    """
    try:
        import site
        site_pkgs = Path(site.getsitepackages()[0])
    except Exception:
        site_pkgs = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"

    shim_dir = site_pkgs / "fp_shims"
    shim_dir.mkdir(exist_ok=True)

    # Mock open3d
    (shim_dir / "__init__.py").write_text("")
    (shim_dir / "_open3d_shim.py").write_text('''"""Minimal open3d mock for FoundationPose (only voxel_down_sample + transform used)."""
import sys
import types
import numpy as _np

_mock = types.ModuleType('open3d')
for sub in ['geometry', 'io', 'utility', 'visualization', 'core', 't', 'pipelines']:
    m = types.ModuleType(f'open3d.{sub}')
    setattr(_mock, sub, m)
    sys.modules[f'open3d.{sub}'] = m


class _MockPC:
    def __init__(self):
        self.points = None
        self.colors = None
        self.normals = None

    def voxel_down_sample(self, voxel_size):
        if self.points is None:
            return self
        pts = _np.asarray(self.points)
        if len(pts) == 0:
            return self
        q = _np.round(pts / voxel_size).astype(_np.int32)
        _, idx = _np.unique(q, axis=0, return_index=True)
        r = _MockPC()
        r.points = _mock.utility.Vector3dVector(pts[idx])
        if self.normals is not None:
            r.normals = _mock.utility.Vector3dVector(_np.asarray(self.normals)[idx])
        return r

    def paint_uniform_color(self, c):
        return self

    def estimate_normals(self, **kw):
        pass

    def transform(self, T):
        if self.points is not None:
            pts = _np.asarray(self.points)
            pts_h = _np.c_[pts, _np.ones(len(pts))]
            pts_t = (T @ pts_h.T).T[:, :3]
            self.points = _mock.utility.Vector3dVector(pts_t)
        return self


_mock.geometry.PointCloud = _MockPC
_mock.utility.Vector3dVector = lambda x: _np.asarray(x)
_mock.io.write_point_cloud = lambda *a, **k: None
_mock.io.read_point_cloud = lambda *a, **k: _MockPC()
sys.modules['open3d'] = _mock
''')

    # Mock mycpp.cluster_poses
    (shim_dir / "_mycpp_shim.py").write_text('''"""Python fallback para mycpp.cluster_poses."""
import sys
import types
import numpy as _np


def _rotation_geodesic(R1, R2):
    cos_a = ((_np.trace(R1 @ R2.T) - 1.0) / 2.0)
    cos_a = _np.clip(cos_a, -1.0, 1.0)
    return _np.arccos(cos_a)


def cluster_poses(angle_diff_deg, dist_diff, poses_in, symmetry_tfs):
    """Greedy pose clustering respetando simetrías del objeto."""
    radian_thres = angle_diff_deg / 180.0 * _np.pi
    poses_in = _np.asarray(poses_in)
    symmetry_tfs = _np.asarray(symmetry_tfs)
    if len(poses_in) == 0:
        return poses_in
    out = [poses_in[0]]
    for i in range(1, len(poses_in)):
        cur = poses_in[i]
        is_new = True
        t1 = cur[:3, 3]
        for c in out:
            t0 = c[:3, 3]
            if _np.linalg.norm(t0 - t1) >= dist_diff:
                continue
            for tf in symmetry_tfs:
                cur_sym = cur @ tf
                rd = _rotation_geodesic(cur_sym[:3, :3], c[:3, :3])
                if rd < radian_thres:
                    is_new = False
                    break
            if not is_new:
                break
        if is_new:
            out.append(cur)
    return _np.array(out)


# Try real mycpp first, fallback to this
try:
    import mycpp  # noqa: F401
    if not hasattr(sys.modules.get('mycpp'), 'cluster_poses'):
        raise ImportError
except Exception:
    _mod = types.ModuleType('mycpp')
    _mod.cluster_poses = cluster_poses
    _build = types.ModuleType('mycpp.build')
    _inner = types.ModuleType('mycpp.build.mycpp')
    _inner.cluster_poses = cluster_poses
    _build.mycpp = _inner
    _mod.build = _build
    sys.modules['mycpp'] = _mod
    sys.modules['mycpp.build'] = _build
    sys.modules['mycpp.build.mycpp'] = _inner
''')

    # Auto-load shims via .pth file
    pth_file = site_pkgs / "fp_shims_autoload.pth"
    pth_file.write_text(
        "import fp_shims._open3d_shim; import fp_shims._mycpp_shim\n"
    )
    print(f"[OK] Shims open3d + mycpp instalados en {shim_dir}")
    print(f"[OK] Auto-load via {pth_file}")


if __name__ == "__main__":
    patch_utils_py()
    install_runtime_shims()
    print("\n[OK] FoundationPose parcheado para el contenedor GPU.")
