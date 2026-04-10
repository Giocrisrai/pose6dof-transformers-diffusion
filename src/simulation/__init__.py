# simulation — CoppeliaSim + ROS 2 + MoveIt 2 + Visual Servoing
#
# Módulos:
#   - visual_servoing: PBVS, IBVS, Hybrid controller
#   - coppeliasim_bridge: ZMQ Remote API interface
#   - ros2_interface: ROS 2 node wrappers (requires rclpy)

from .visual_servoing import PBVSController, IBVSController, HybridServoController
from .coppeliasim_bridge import CoppeliaSimBridge, CameraConfig, RobotConfig

try:
    from .ros2_interface import PoseEstimationNode, GraspPlannerNode
except ImportError:
    pass  # ROS 2 only available inside Docker container
