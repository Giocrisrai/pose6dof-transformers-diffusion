"""
CoppeliaSim Bridge — ZMQ Remote API interface for bin picking simulation.

Provides a Python interface to control a CoppeliaSim scene containing:
    - A robot manipulator (UR5e or Franka Panda)
    - A parallel-jaw gripper
    - A virtual RGB-D camera (Intel RealSense D435)
    - A bin with objects for picking

Uses the CoppeliaSim ZMQ Remote API (coppeliasim_zmqremoteapi_client).

References:
    - CoppeliaSim Remote API: https://manual.coppeliarobotics.com/en/zmqRemoteApiOverview.htm
    - UR5e URDF: https://github.com/UniversalRobots/Universal_Robots_ROS2_Description
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import logging
import time

logger = logging.getLogger(__name__)


@dataclass
class CameraConfig:
    """Configuration for the virtual RGB-D camera."""
    resolution: Tuple[int, int] = (640, 480)
    fov: float = 1.047  # ~60 degrees (RealSense D435)
    near_clip: float = 0.1
    far_clip: float = 3.0
    # Intrinsics (computed from resolution and fov)
    fx: float = 0.0
    fy: float = 0.0
    cx: float = 0.0
    cy: float = 0.0

    def __post_init__(self):
        w, h = self.resolution
        self.cx = w / 2.0
        self.cy = h / 2.0
        self.fx = w / (2.0 * np.tan(self.fov / 2.0))
        self.fy = self.fx  # square pixels

    @property
    def K(self) -> np.ndarray:
        """Camera intrinsic matrix."""
        return np.array([
            [self.fx, 0, self.cx],
            [0, self.fy, self.cy],
            [0, 0, 1]
        ])


@dataclass
class RobotConfig:
    """Robot arm configuration."""
    name: str = "UR5e"
    n_joints: int = 6
    joint_names: Tuple = (
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    )
    home_position: Tuple = (0, -np.pi/2, np.pi/2, -np.pi/2, -np.pi/2, 0)
    max_velocity: float = 1.0  # rad/s
    max_acceleration: float = 2.0  # rad/s²


class CoppeliaSimBridge:
    """Interface to CoppeliaSim simulation for bin picking.

    Manages:
        - Scene objects (robot, gripper, camera, bin, workpieces)
        - Robot joint control (position, velocity)
        - Camera capture (RGB + depth)
        - Object pose manipulation
        - Gripper actuation

    Usage:
        bridge = CoppeliaSimBridge()
        bridge.connect()
        bridge.start_simulation()

        rgb, depth = bridge.capture_rgbd()
        bridge.move_joints(target_joints)
        bridge.actuate_gripper(open=False)

        bridge.stop_simulation()
        bridge.disconnect()
    """

    def __init__(
        self,
        host: str = "localhost",
        port: int = 23000,
        robot_config: Optional[RobotConfig] = None,
        camera_config: Optional[CameraConfig] = None,
    ):
        self.host = host
        self.port = port
        self.robot_config = robot_config or RobotConfig()
        self.camera_config = camera_config or CameraConfig()

        self._client = None
        self._sim = None
        self._connected = False

        # Object handles (populated on connect)
        self._robot_handle = None
        self._joint_handles = []
        self._camera_rgb_handle = None
        self._camera_depth_handle = None
        self._gripper_handle = None
        self._tip_handle = None
        self._object_handles: Dict[str, int] = {}

    def connect(self):
        """Connect to CoppeliaSim via ZMQ Remote API.

        Raises:
            ImportError: If coppeliasim_zmqremoteapi_client not installed
            ConnectionError: If CoppeliaSim is not running
        """
        try:
            from coppeliasim_zmqremoteapi_client import RemoteAPIClient
        except ImportError:
            logger.error(
                "Install: pip install coppeliasim-zmqremoteapi-client\n"
                "Or run CoppeliaSim with the ZMQ plugin enabled."
            )
            raise ImportError(
                "coppeliasim_zmqremoteapi_client not found. "
                "Install with: pip install coppeliasim-zmqremoteapi-client"
            )

        try:
            self._client = RemoteAPIClient(self.host, self.port)
            self._sim = self._client.require("sim")
            self._connected = True
            logger.info(f"Connected to CoppeliaSim at {self.host}:{self.port}")

            # Get object handles
            self._init_handles()
        except Exception as e:
            raise ConnectionError(
                f"Cannot connect to CoppeliaSim at {self.host}:{self.port}. "
                f"Make sure CoppeliaSim is running with ZMQ plugin. Error: {e}"
            )

    def _init_handles(self):
        """Initialize handles for scene objects."""
        sim = self._sim

        # Robot joints
        self._joint_handles = []
        for name in self.robot_config.joint_names:
            try:
                handle = sim.getObject(f"/{name}")
                self._joint_handles.append(handle)
            except Exception:
                logger.warning(f"Joint '{name}' not found in scene")

        # Camera
        try:
            self._camera_rgb_handle = sim.getObject("/rgb_camera")
            self._camera_depth_handle = sim.getObject("/depth_camera")
        except Exception:
            logger.warning("Camera handles not found — using /Vision_sensor")
            try:
                self._camera_rgb_handle = sim.getObject("/Vision_sensor")
                self._camera_depth_handle = self._camera_rgb_handle
            except Exception:
                logger.warning("No camera found in scene")

        # Gripper
        try:
            self._gripper_handle = sim.getObject("/gripper")
        except Exception:
            logger.warning("Gripper not found in scene")

        # Tip (TCP)
        try:
            self._tip_handle = sim.getObject("/tip")
        except Exception:
            logger.warning("Tip/TCP not found in scene")

        logger.info(
            f"Handles: {len(self._joint_handles)} joints, "
            f"camera={'OK' if self._camera_rgb_handle else 'NONE'}, "
            f"gripper={'OK' if self._gripper_handle else 'NONE'}"
        )

    def disconnect(self):
        """Disconnect from CoppeliaSim."""
        self._connected = False
        self._client = None
        self._sim = None
        logger.info("Disconnected from CoppeliaSim")

    def start_simulation(self):
        """Start the simulation."""
        self._check_connected()
        self._sim.startSimulation()
        logger.info("Simulation started")

    def stop_simulation(self):
        """Stop the simulation."""
        self._check_connected()
        self._sim.stopSimulation()
        logger.info("Simulation stopped")

    def step(self):
        """Trigger a single simulation step (for stepped mode)."""
        self._check_connected()
        self._client.step()

    # ── Camera ────────────────────────────────────────────────

    def capture_rgbd(self) -> Tuple[np.ndarray, np.ndarray]:
        """Capture RGB and depth images from the virtual camera.

        Returns:
            rgb: (H, W, 3) uint8 RGB image
            depth: (H, W) float32 depth in meters
        """
        self._check_connected()
        sim = self._sim
        w, h = self.camera_config.resolution

        # RGB
        if self._camera_rgb_handle is not None:
            img_raw, res = sim.getVisionSensorImg(self._camera_rgb_handle)
            rgb = np.frombuffer(img_raw, dtype=np.uint8).reshape(h, w, 3)
            rgb = np.flipud(rgb)  # CoppeliaSim returns flipped
        else:
            rgb = np.zeros((h, w, 3), dtype=np.uint8)

        # Depth
        if self._camera_depth_handle is not None:
            depth_raw, res = sim.getVisionSensorDepth(self._camera_depth_handle)
            depth = np.frombuffer(depth_raw, dtype=np.float32).reshape(h, w)
            depth = np.flipud(depth)
            # Convert from normalized [0,1] to meters
            depth = (
                self.camera_config.near_clip
                + depth * (self.camera_config.far_clip - self.camera_config.near_clip)
            )
        else:
            depth = np.zeros((h, w), dtype=np.float32)

        return rgb, depth

    def get_camera_pose(self) -> np.ndarray:
        """Get camera pose in world frame.

        Returns:
            (4, 4) SE(3) transformation
        """
        self._check_connected()
        if self._camera_rgb_handle is None:
            return np.eye(4)

        pos = self._sim.getObjectPosition(self._camera_rgb_handle, -1)
        quat = self._sim.getObjectQuaternion(self._camera_rgb_handle, -1)

        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = self._quat_to_rot(quat)
        return T

    # ── Robot Control ─────────────────────────────────────────

    def get_joint_positions(self) -> np.ndarray:
        """Get current joint positions.

        Returns:
            (n_joints,) array of joint angles in radians
        """
        self._check_connected()
        positions = []
        for handle in self._joint_handles:
            pos = self._sim.getJointPosition(handle)
            positions.append(pos)
        return np.array(positions)

    def move_joints(
        self,
        target: np.ndarray,
        max_velocity: Optional[float] = None,
        blocking: bool = True,
        timeout: float = 10.0,
    ):
        """Move robot to target joint configuration.

        Args:
            target: (n_joints,) target joint angles
            max_velocity: Max joint velocity (rad/s). Uses config default if None.
            blocking: Wait until motion completes.
            timeout: Max wait time in seconds.
        """
        self._check_connected()
        vel = max_velocity or self.robot_config.max_velocity

        for i, handle in enumerate(self._joint_handles):
            self._sim.setJointTargetPosition(handle, float(target[i]))

        if blocking:
            self._wait_until_reached(target, timeout)

    def move_to_home(self):
        """Move robot to home configuration."""
        home = np.array(self.robot_config.home_position)
        self.move_joints(home)

    def get_tip_pose(self) -> np.ndarray:
        """Get TCP (tool center point) pose.

        Returns:
            (4, 4) SE(3) transformation in world frame
        """
        self._check_connected()
        if self._tip_handle is None:
            logger.warning("No tip handle — returning identity")
            return np.eye(4)

        pos = self._sim.getObjectPosition(self._tip_handle, -1)
        quat = self._sim.getObjectQuaternion(self._tip_handle, -1)

        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = self._quat_to_rot(quat)
        return T

    # ── Gripper ───────────────────────────────────────────────

    def actuate_gripper(self, open: bool = True):
        """Open or close the gripper.

        Args:
            open: True to open, False to close.
        """
        self._check_connected()
        if self._gripper_handle is None:
            logger.warning("No gripper handle")
            return

        # Convention: positive velocity = open, negative = close
        vel = 0.04 if open else -0.04
        self._sim.setJointTargetVelocity(self._gripper_handle, vel)

    def is_grasping(self) -> bool:
        """Check if gripper is holding an object (force threshold)."""
        self._check_connected()
        if self._gripper_handle is None:
            return False
        force = self._sim.getJointForce(self._gripper_handle)
        return abs(force) > 0.5  # N

    # ── Object Management ─────────────────────────────────────

    def get_object_pose(self, name: str) -> np.ndarray:
        """Get object pose by name.

        Args:
            name: Object name in CoppeliaSim scene

        Returns:
            (4, 4) SE(3) transformation
        """
        self._check_connected()
        if name not in self._object_handles:
            self._object_handles[name] = self._sim.getObject(f"/{name}")

        handle = self._object_handles[name]
        pos = self._sim.getObjectPosition(handle, -1)
        quat = self._sim.getObjectQuaternion(handle, -1)

        T = np.eye(4)
        T[:3, 3] = pos
        T[:3, :3] = self._quat_to_rot(quat)
        return T

    def set_object_pose(self, name: str, pose: np.ndarray):
        """Set object pose by name.

        Args:
            name: Object name
            pose: (4, 4) SE(3) transformation
        """
        self._check_connected()
        if name not in self._object_handles:
            self._object_handles[name] = self._sim.getObject(f"/{name}")

        handle = self._object_handles[name]
        pos = pose[:3, 3].tolist()
        quat = self._rot_to_quat(pose[:3, :3]).tolist()

        self._sim.setObjectPosition(handle, -1, pos)
        self._sim.setObjectQuaternion(handle, -1, quat)

    def randomize_object_poses(
        self,
        names: List[str],
        workspace_bounds: Tuple[Tuple[float, float], ...] = (
            (-0.15, 0.15), (-0.15, 0.15), (0.02, 0.10)
        ),
    ):
        """Randomize positions of objects in the bin.

        Args:
            names: List of object names
            workspace_bounds: ((x_min, x_max), (y_min, y_max), (z_min, z_max))
        """
        for name in names:
            pos = np.array([
                np.random.uniform(*workspace_bounds[0]),
                np.random.uniform(*workspace_bounds[1]),
                np.random.uniform(*workspace_bounds[2]),
            ])
            # Random rotation
            angle = np.random.uniform(0, 2 * np.pi)
            axis = np.random.randn(3)
            axis = axis / np.linalg.norm(axis)
            from src.utils.lie_groups import so3_exp, pose_from_Rt
            R = so3_exp(axis * angle)
            T = pose_from_Rt(R, pos)
            self.set_object_pose(name, T)

    # ── Bin Picking Cycle ─────────────────────────────────────

    def execute_pick(
        self,
        grasp_pose: np.ndarray,
        pre_grasp_offset: float = 0.10,
        lift_height: float = 0.15,
        place_pose: Optional[np.ndarray] = None,
    ) -> bool:
        """Execute a complete pick-and-place cycle.

        1. Move to pre-grasp (above the grasp pose)
        2. Approach to grasp pose
        3. Close gripper
        4. Lift
        5. (Optional) Move to place pose

        Args:
            grasp_pose: (4, 4) SE(3) grasp pose
            pre_grasp_offset: Offset along approach for pre-grasp
            lift_height: How high to lift after grasping
            place_pose: (4, 4) optional place pose

        Returns:
            True if pick was successful (gripper has object)
        """
        # This is a high-level method that would use IK and motion planning
        # For now, returns a placeholder
        logger.info("Executing pick cycle...")

        # TODO: Implement with MoveIt 2 or direct IK
        # 1. IK for pre-grasp
        # 2. Linear approach
        # 3. Gripper close
        # 4. Check grasp success
        # 5. Lift
        # 6. Place

        return False

    # ── Helpers ────────────────────────────────────────────────

    def _check_connected(self):
        if not self._connected:
            raise RuntimeError("Not connected to CoppeliaSim. Call connect() first.")

    def _wait_until_reached(self, target: np.ndarray, timeout: float = 10.0):
        """Wait until joints reach target configuration."""
        start = time.time()
        while time.time() - start < timeout:
            current = self.get_joint_positions()
            error = np.max(np.abs(current - target))
            if error < 0.01:  # ~0.5 degrees
                return
            time.sleep(0.05)
        logger.warning(f"Timeout waiting for joints (error: {error:.4f} rad)")

    @staticmethod
    def _quat_to_rot(quat: list) -> np.ndarray:
        """Convert CoppeliaSim quaternion [x,y,z,w] to rotation matrix."""
        x, y, z, w = quat
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
            [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)],
        ])

    @staticmethod
    def _rot_to_quat(R: np.ndarray) -> np.ndarray:
        """Convert rotation matrix to CoppeliaSim quaternion [x,y,z,w]."""
        from src.utils.rotations import matrix_to_quat
        q_wxyz = matrix_to_quat(R)  # [w, x, y, z]
        return np.array([q_wxyz[1], q_wxyz[2], q_wxyz[3], q_wxyz[0]])
