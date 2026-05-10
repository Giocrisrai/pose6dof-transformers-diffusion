# perception — Estimación de pose 6-DoF
#
# Métodos implementados:
#   - FoundationPose (Wen et al., CVPR 2024) — método principal
#   - GDR-Net++ (Wang et al., CVPR 2021) — baseline comparativo

from .detector import Detection, GTDetector, SimpleSegmentor
from .evaluator import PoseEvaluator
from .foundation_pose import FoundationPoseEstimator
from .gdrnet import GDRNetEstimator
