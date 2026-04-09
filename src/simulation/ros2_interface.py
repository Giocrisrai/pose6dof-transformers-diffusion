"""
ROS 2 Interface — Nodes and topics for the bin picking pipeline.

Provides ROS 2 wrappers that connect the perception/planning pipeline
to the robot in CoppeliaSim via MoveIt 2.

Architecture:
    Camera (CoppeliaSim) → /camera/rgb, /camera/depth
        → PoseEstimationNode → /object_pose (PoseStamped)
        → GraspPlannerNode  → /grasp_trajectory (JointTrajectory)
        → MoveIt 2          → /joint_commands
        → CoppeliaSim Robot

Requires:
    - ROS 2 Humble (installed via Docker or native)
    - MoveIt 2
    - sensor_msgs, geometry_msgs, trajectory_msgs

Note:
    This module is designed to run inside the Docker container
    (docker/Dockerfile) where ROS 2 is available.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Check if ROS 2 is available
_ROS2_AVAILABLE = False
try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import Image, CameraInfo
    from geometry_msgs.msg import PoseStamped, TransformStamped
    from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
    from std_msgs.msg import Bool
    _ROS2_AVAILABLE = True
except ImportError:
    logger.info("ROS 2 not available — using stub implementations")


def _require_ros2():
    if not _ROS2_AVAILABLE:
        raise ImportError(
            "ROS 2 (rclpy) not found. Run inside the Docker container:\n"
            "  docker compose -f docker/docker-compose.yml run ros2-sim"
        )


# ================================================================
# Stub base class for when ROS 2 is not available
# ================================================================

class _StubNode:
    """Placeholder when rclpy is not available."""
    def __init__(self, name: str):
        self.name = name
        logger.warning(f"ROS 2 stub: {name} (rclpy not available)")

    def get_logger(self):
        return logger


_BaseNode = Node if _ROS2_AVAILABLE else _StubNode


# ================================================================
# Perception Node
# ================================================================

class PoseEstimationNode(_BaseNode):
    """ROS 2 node for 6-DoF pose estimation.

    Subscribes:
        /camera/rgb        (sensor_msgs/Image)
        /camera/depth      (sensor_msgs/Image)
        /camera/info       (sensor_msgs/CameraInfo)

    Publishes:
        /object_pose       (geometry_msgs/PoseStamped)
        /detection_mask    (sensor_msgs/Image)
    """

    def __init__(self):
        super().__init__("pose_estimation_node")
        self.get_logger().info("PoseEstimationNode initialized")

        if _ROS2_AVAILABLE:
            # Subscribers
            self._rgb_sub = self.create_subscription(
                Image, "/camera/rgb", self._rgb_callback, 10)
            self._depth_sub = self.create_subscription(
                Image, "/camera/depth", self._depth_callback, 10)
            self._info_sub = self.create_subscription(
                CameraInfo, "/camera/info", self._info_callback, 10)

            # Publishers
            self._pose_pub = self.create_publisher(
                PoseStamped, "/object_pose", 10)

            # State
            self._latest_rgb = None
            self._latest_depth = None
            self._camera_K = None

            # Timer for periodic estimation
            self._timer = self.create_timer(0.1, self._estimate_callback)  # 10 Hz

    def _rgb_callback(self, msg):
        self._latest_rgb = self._imgmsg_to_numpy(msg)

    def _depth_callback(self, msg):
        self._latest_depth = self._imgmsg_to_numpy(msg, depth=True)

    def _info_callback(self, msg):
        self._camera_K = np.array(msg.k).reshape(3, 3)

    def _estimate_callback(self):
        """Periodic pose estimation."""
        if self._latest_rgb is None or self._latest_depth is None:
            return
        if self._camera_K is None:
            return

        # TODO: Call FoundationPose estimator
        # pose = self.estimator.estimate(rgb, depth, mask, K)
        # self._publish_pose(pose)

    def _publish_pose(self, T: np.ndarray):
        """Publish estimated pose as PoseStamped."""
        if not _ROS2_AVAILABLE:
            return

        from src.utils.rotations import matrix_to_quat
        msg = PoseStamped()
        msg.header.frame_id = "camera_link"
        # msg.header.stamp = self.get_clock().now().to_msg()

        msg.pose.position.x = float(T[0, 3])
        msg.pose.position.y = float(T[1, 3])
        msg.pose.position.z = float(T[2, 3])

        q = matrix_to_quat(T[:3, :3])  # [w, x, y, z]
        msg.pose.orientation.w = float(q[0])
        msg.pose.orientation.x = float(q[1])
        msg.pose.orientation.y = float(q[2])
        msg.pose.orientation.z = float(q[3])

        self._pose_pub.publish(msg)

    @staticmethod
    def _imgmsg_to_numpy(msg, depth=False):
        """Convert ROS Image message to numpy array."""
        if depth:
            return np.frombuffer(msg.data, dtype=np.float32).reshape(
                msg.height, msg.width)
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(
            msg.height, msg.width, 3)


# ================================================================
# Grasp Planning Node
# ================================================================

class GraspPlannerNode(_BaseNode):
    """ROS 2 node for grasp trajectory planning.

    Subscribes:
        /object_pose       (geometry_msgs/PoseStamped)

    Publishes:
        /grasp_trajectory  (trajectory_msgs/JointTrajectory)
        /gripper_command   (std_msgs/Bool)  True=open, False=close
    """

    def __init__(self):
        super().__init__("grasp_planner_node")
        self.get_logger().info("GraspPlannerNode initialized")

        if _ROS2_AVAILABLE:
            self._pose_sub = self.create_subscription(
                PoseStamped, "/object_pose", self._pose_callback, 10)
            self._traj_pub = self.create_publisher(
                JointTrajectory, "/grasp_trajectory", 10)
            self._gripper_pub = self.create_publisher(
                Bool, "/gripper_command", 10)

    def _pose_callback(self, msg):
        """Received new object pose → plan grasp."""
        T = self._posemsg_to_matrix(msg)
        # TODO: Use DiffusionGraspPlanner or GraspSampler
        # trajectory = self.planner.plan_grasp(T)
        # self._publish_trajectory(trajectory)

    @staticmethod
    def _posemsg_to_matrix(msg) -> np.ndarray:
        """Convert PoseStamped to SE(3) matrix."""
        from src.utils.rotations import quat_to_matrix
        p = msg.pose.position
        o = msg.pose.orientation
        T = np.eye(4)
        T[:3, 3] = [p.x, p.y, p.z]
        T[:3, :3] = quat_to_matrix(np.array([o.w, o.x, o.y, o.z]))
        return T


# ================================================================
# Pipeline Launcher
# ================================================================

def launch_pipeline():
    """Launch the full ROS 2 bin picking pipeline.

    Spins up:
        - PoseEstimationNode
        - GraspPlannerNode

    Usage (inside Docker):
        python -m src.simulation.ros2_interface
    """
    _require_ros2()

    rclpy.init()
    logger.info("Launching bin picking pipeline...")

    pose_node = PoseEstimationNode()
    grasp_node = GraspPlannerNode()

    executor = rclpy.executors.MultiThreadedExecutor()
    executor.add_node(pose_node)
    executor.add_node(grasp_node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        pose_node.destroy_node()
        grasp_node.destroy_node()
        rclpy.shutdown()
        logger.info("Pipeline shut down")


if __name__ == "__main__":
    launch_pipeline()
