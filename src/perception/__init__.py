# perception — Estimación de pose 6-DoF
#
# Métodos implementados:
#   - FoundationPose (Wen et al., CVPR 2024) — método principal
#   - GDR-Net++ (Wang et al., CVPR 2021) — baseline comparativo

from .foundation_pose import FoundationPoseEstimator
from .gdrnet import GDRNetEstimator
from .detector import GTDetector, SimpleSegmentor, Detection
from .evaluator import PoseEvaluator
