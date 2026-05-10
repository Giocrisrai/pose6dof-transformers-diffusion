# utils — Utilidades comunes (geometría, métricas, visualización)

from .dataset_loader import BOPDataset, verify_dataset
from .lie_groups import (
    geodesic_distance_SE3,
    geodesic_distance_SO3,
    pose_from_Rt,
    pose_to_Rt,
    se3_adjoint,
    se3_compose,
    se3_exp,
    se3_hat,
    se3_inverse,
    se3_log,
    so3_exp,
    so3_hat,
    so3_log,
    so3_vee,
)
from .metrics import (
    add_metric,
    add_s_metric,
    compute_add,
    compute_adds,
    compute_auc,
    compute_recall,
    mspd,
    mssd,
    vsd,
)
from .rotations import (
    axisangle_to_matrix,
    euler_to_matrix,
    matrix_to_6d,
    matrix_to_6d_torch,
    matrix_to_axisangle,
    matrix_to_euler,
    matrix_to_quat,
    quat_angular_distance,
    quat_conjugate,
    quat_multiply,
    quat_to_matrix,
    sixd_to_matrix,
    sixd_to_matrix_torch,
)
from .visualization import (
    draw_bbox_3d,
    draw_pose_axes,
    draw_projected_points,
    plot_metrics_comparison,
    plot_pose_comparison,
)
