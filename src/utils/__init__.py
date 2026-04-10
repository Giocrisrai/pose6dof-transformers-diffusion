# utils — Utilidades comunes (geometría, métricas, visualización)

from .lie_groups import (
    so3_exp, so3_log, so3_hat, so3_vee,
    se3_exp, se3_log, se3_hat,
    se3_compose, se3_inverse, se3_adjoint,
    pose_from_Rt, pose_to_Rt,
    geodesic_distance_SO3, geodesic_distance_SE3,
)

from .rotations import (
    quat_to_matrix, matrix_to_quat,
    quat_multiply, quat_conjugate, quat_angular_distance,
    axisangle_to_matrix, matrix_to_axisangle,
    euler_to_matrix, matrix_to_euler,
    matrix_to_6d, sixd_to_matrix,
    sixd_to_matrix_torch, matrix_to_6d_torch,
)

from .metrics import (
    add_metric, add_s_metric,
    vsd, mssd, mspd,
    compute_recall, compute_auc,
    compute_add, compute_adds,
)

from .visualization import (
    draw_pose_axes, draw_projected_points, draw_bbox_3d,
    plot_pose_comparison, plot_metrics_comparison,
)

from .dataset_loader import BOPDataset, verify_dataset
