# simulation — CoppeliaSim + ROS 2 + MoveIt 2 + Visual Servoing
#
# Módulos:
#   - visual_servoing: PBVS, IBVS, Hybrid controller
#   - coppeliasim_bridge: ZMQ Remote API interface
#   - ros2_interface: ROS 2 node wrappers (requires rclpy)

from .coppeliasim_bridge import CameraConfig, CoppeliaSimBridge, RobotConfig
from .visual_servoing import HybridServoController, IBVSController, PBVSController

try:
    from .ros2_interface import GraspPlannerNode, PoseEstimationNode
except ImportError:
    pass  # ROS 2 only available inside Docker container
