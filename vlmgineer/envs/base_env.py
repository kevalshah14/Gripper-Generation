"""
Base environment class for VLMgineer tasks.
"""

import pybullet as p
import pybullet_data
import numpy as np
import os
import tempfile
import shutil
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from PIL import Image
import io


@dataclass
class ToolActionPair:
    """A tool design and its corresponding action sequence."""
    
    tool_urdf: str  # URDF XML string for the tool
    actions: np.ndarray  # Nx7 array: [x, y, z, roll, pitch, yaw, gripper]
    
    # Metadata
    tool_description: str = ""
    action_description: str = ""


@dataclass
class EvaluationResult:
    """Result from evaluating a tool-action pair."""
    
    reward: float  # Task reward (0-1)
    success: bool  # Whether task was completed
    distance_traversed: float  # Total end-effector distance
    final_state: Dict[str, Any]  # Final state of objects
    
    # Optional
    trajectory: Optional[np.ndarray] = None  # Recorded trajectory


class BaseEnv(ABC):
    """
    Base class for VLMgineer task environments.
    
    Subclasses must implement:
    - _setup_task(): Set up task-specific objects
    - _compute_reward(): Calculate the task reward
    - get_task_description(): Return task description for VLM
    """
    
    def __init__(
        self,
        robot_urdf_path: str,
        gui: bool = False,
        time_step: float = 1/240,
        tool_mount_link: str = "tool_mount",
    ):
        """
        Initialize the environment.
        
        Args:
            robot_urdf_path: Path to the robot URDF file
            gui: Whether to show the GUI
            time_step: Simulation time step
            tool_mount_link: Name of the link to attach tools to
        """
        self.robot_urdf_path = robot_urdf_path
        self.gui = gui
        self.time_step = time_step
        self.tool_mount_link = tool_mount_link
        
        # Will be set after connecting
        self.physics_client = None
        self.robot_id = None
        self.plane_id = None
        self.tool_link_index = None
        
        # Joint information
        self.joint_indices: Dict[str, int] = {}
        self.joint_limits: Dict[str, Tuple[float, float]] = {}
        # Franka Panda joints (7 DOF arm + 2 finger joints)
        self.controllable_joints = [
            "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
            "panda_joint5", "panda_joint6", "panda_joint7"
        ]
        self.finger_joints = ["panda_finger_joint1", "panda_finger_joint2"]
        self.end_effector_link = "tool_mount"
        
        # Temp directory for modified URDFs
        self.temp_dir: Optional[str] = None
        
        # Task objects (to be set by subclasses)
        self.task_objects: Dict[str, int] = {}
        
        # Initial state for reset
        self._initial_object_states: Dict[str, Dict] = {}
        
    def connect(self) -> None:
        """Connect to the physics server."""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=1.2,
                cameraYaw=90,
                cameraPitch=-30,
                cameraTargetPosition=[0.4, 0, 0.2]
            )
            # Disable debug visualizers for stability on M1 Macs
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 0)
            p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 0)
            # Don't use real-time simulation - manual stepping is more stable
            p.setRealTimeSimulation(0)
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setTimeStep(self.time_step)
        
    def disconnect(self) -> None:
        """Disconnect from the physics server and cleanup."""
        try:
            if p.isConnected():
                p.disconnect()
        except:
            pass
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            self.temp_dir = None
            
    def _resolve_package_paths(self, urdf_content: str) -> str:
        """Resolve package:// paths in URDF content."""
        import pybullet_data
        
        # Get the pybullet_data directory for Franka Panda meshes
        pybullet_data_dir = pybullet_data.getDataPath()
        
        # Replace package://franka_panda/ paths with pybullet_data path
        resolved = urdf_content.replace(
            "package://franka_panda/",
            os.path.join(pybullet_data_dir, "franka_panda") + "/"
        )
        
        return resolved
    
    def _create_robot_with_tool(self, tool_urdf: Optional[str] = None) -> str:
        """
        Create a combined robot URDF with an attached tool.
        
        Args:
            tool_urdf: URDF XML string for the tool (links and joints only)
            
        Returns:
            Path to the combined URDF file
        """
        # Read the base robot URDF (Franka Panda from project)
        vlmgineer_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        project_dir = os.path.dirname(vlmgineer_dir)
        robot_path = os.path.join(project_dir, self.robot_urdf_path)
        
        with open(robot_path, 'r') as f:
            robot_urdf = f.read()
        
        # Resolve package paths (meshes from pybullet_data)
        robot_urdf = self._resolve_package_paths(robot_urdf)
        
        # Insert tool URDF before </robot> if provided
        if tool_urdf:
            robot_urdf = robot_urdf.replace("</robot>", f"{tool_urdf}\n</robot>")
        
        # Write to temp file
        if self.temp_dir is None:
            self.temp_dir = tempfile.mkdtemp()
        
        combined_path = os.path.join(self.temp_dir, "robot_with_tool.urdf")
        with open(combined_path, 'w') as f:
            f.write(robot_urdf)
        
        return combined_path
    
    def load_robot(self, tool_urdf: Optional[str] = None) -> None:
        """
        Load the robot (optionally with a tool attached).
        
        Args:
            tool_urdf: URDF XML string for the tool
        """
        # Create combined URDF
        urdf_path = self._create_robot_with_tool(tool_urdf)
        
        # Load robot
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
        
        # Get joint information
        num_joints = p.getNumJoints(self.robot_id)
        for i in range(num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            link_name = joint_info[12].decode('utf-8')
            
            # Track tool mount link
            if link_name == self.tool_mount_link:
                self.tool_link_index = i
            
            # Track controllable joints
            if joint_type != p.JOINT_FIXED:
                self.joint_indices[joint_name] = i
                self.joint_limits[joint_name] = (joint_info[8], joint_info[9])
    
    def setup_environment(self) -> None:
        """Set up the base environment (ground plane, etc.)."""
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Add a table
        table_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.4, 0.6, 0.02]
        )
        table_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.4, 0.6, 0.02],
            rgbaColor=[0.6, 0.4, 0.2, 1]
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_collision,
            baseVisualShapeIndex=table_visual,
            basePosition=[0.5, 0, 0.02]
        )
        
        # Call task-specific setup
        self._setup_task()
        
        # Store initial object states
        self._store_initial_states()
    
    def _store_initial_states(self) -> None:
        """Store initial states of task objects for reset."""
        for name, obj_id in self.task_objects.items():
            pos, orn = p.getBasePositionAndOrientation(obj_id)
            vel, ang_vel = p.getBaseVelocity(obj_id)
            self._initial_object_states[name] = {
                "position": pos,
                "orientation": orn,
                "velocity": vel,
                "angular_velocity": ang_vel
            }
    
    @abstractmethod
    def _setup_task(self) -> None:
        """Set up task-specific objects. Override in subclasses."""
        pass
    
    @abstractmethod
    def _compute_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute the task reward.
        
        Returns:
            reward: Float between 0 and 1
            success: Whether the task is complete
            info: Additional information
        """
        pass
    
    @abstractmethod
    def get_task_description(self) -> str:
        """Return a description of the task for the VLM."""
        pass
    
    @abstractmethod
    def get_environment_code(self) -> str:
        """Return the environment setup code for the VLM."""
        pass
    
    def reset(self) -> None:
        """Reset the environment to initial state."""
        # Reset robot joints to home position
        for joint_name in self.controllable_joints:
            if joint_name in self.joint_indices:
                p.resetJointState(self.robot_id, self.joint_indices[joint_name], 0)
        
        # Reset task objects
        for name, state in self._initial_object_states.items():
            obj_id = self.task_objects[name]
            p.resetBasePositionAndOrientation(
                obj_id, state["position"], state["orientation"]
            )
            p.resetBaseVelocity(
                obj_id, state["velocity"], state["angular_velocity"]
            )
    
    def step(self) -> None:
        """Step the simulation."""
        p.stepSimulation()
    
    def get_tool_mount_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get the current pose of the tool mount link."""
        if self.tool_link_index is None:
            raise ValueError("Tool mount link not found")
        
        state = p.getLinkState(self.robot_id, self.tool_link_index)
        position = np.array(state[0])
        orientation = np.array(state[1])
        return position, orientation
    
    def set_end_effector_pose(
        self,
        position: np.ndarray,
        orientation: np.ndarray,
        max_iterations: int = 100
    ) -> bool:
        """
        Set the end effector to a target pose using IK.
        
        Args:
            position: Target position [x, y, z]
            orientation: Target orientation as quaternion [x, y, z, w]
            max_iterations: Max IK iterations
            
        Returns:
            success: Whether IK converged
        """
        if self.tool_link_index is None:
            raise ValueError("Tool mount link not found")
        
        # Compute IK
        joint_poses = p.calculateInverseKinematics(
            self.robot_id,
            self.tool_link_index,
            position,
            orientation,
            maxNumIterations=max_iterations
        )
        
        # Apply joint positions
        for i, joint_name in enumerate(self.controllable_joints):
            if joint_name in self.joint_indices and i < len(joint_poses):
                idx = self.joint_indices[joint_name]
                p.setJointMotorControl2(
                    self.robot_id,
                    idx,
                    p.POSITION_CONTROL,
                    targetPosition=joint_poses[i],
                    force=500,
                    maxVelocity=2.0
                )
        
        return True
    
    def set_gripper(self, open_amount: float) -> None:
        """
        Set the gripper opening.
        
        Args:
            open_amount: 0 = fully closed, 1 = fully open
        """
        # Panda gripper has 0-0.04m range per finger
        target_pos = open_amount * 0.04
        
        for finger_joint in self.finger_joints:
            if finger_joint in self.joint_indices:
                idx = self.joint_indices[finger_joint]
                p.setJointMotorControl2(
                    self.robot_id,
                    idx,
                    p.POSITION_CONTROL,
                    targetPosition=target_pos,
                    force=50,
                    maxVelocity=0.5
                )
    
    def capture_image(self, width: int = 640, height: int = 480) -> Image.Image:
        """
        Capture an image of the current scene.
        
        Returns:
            PIL Image of the scene
        """
        # Camera parameters
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=[1.0, 0.5, 0.8],
            cameraTargetPosition=[0.4, 0, 0.2],
            cameraUpVector=[0, 0, 1]
        )
        proj_matrix = p.computeProjectionMatrixFOV(
            fov=60,
            aspect=width / height,
            nearVal=0.1,
            farVal=100
        )
        
        # Render image
        _, _, rgb, _, _ = p.getCameraImage(
            width=width,
            height=height,
            viewMatrix=view_matrix,
            projectionMatrix=proj_matrix,
            renderer=p.ER_BULLET_HARDWARE_OPENGL if self.gui else p.ER_TINY_RENDERER
        )
        
        # Convert to PIL Image
        rgb_array = np.array(rgb, dtype=np.uint8).reshape(height, width, 4)
        return Image.fromarray(rgb_array[:, :, :3])
    
    def evaluate(
        self,
        tool_action: ToolActionPair,
        max_steps: int = 1000
    ) -> EvaluationResult:
        """
        Evaluate a tool-action pair.
        
        Args:
            tool_action: The tool design and action sequence to evaluate
            max_steps: Maximum simulation steps
            
        Returns:
            EvaluationResult with reward and metrics
        """
        # Reset environment
        self.reset()
        
        # Track trajectory and distance
        trajectory = []
        total_distance = 0.0
        last_pos = None
        
        # Execute actions
        actions = tool_action.actions
        n_waypoints = len(actions)
        
        for i in range(n_waypoints):
            waypoint = actions[i]
            position = waypoint[:3]
            orientation_euler = waypoint[3:6]
            # Gripper: 0 = open, 1 = closed (from VLMgineer paper convention)
            gripper_closed = waypoint[6] if len(waypoint) > 6 else 0
            
            # Convert euler to quaternion
            orientation = p.getQuaternionFromEuler(orientation_euler)
            
            # Set target pose
            self.set_end_effector_pose(position, orientation)
            
            # Set gripper (invert: 0=closed->1.0 open, 1=closed->0.0 open)
            self.set_gripper(1.0 - gripper_closed)
            
            # Step simulation to reach target
            steps_per_waypoint = max_steps // n_waypoints
            for _ in range(steps_per_waypoint):
                self.step()
                
                # Track position
                current_pos, _ = self.get_tool_mount_pose()
                trajectory.append(current_pos.copy())
                
                if last_pos is not None:
                    total_distance += np.linalg.norm(current_pos - last_pos)
                last_pos = current_pos.copy()
        
        # Compute final reward
        reward, success, info = self._compute_reward()
        
        return EvaluationResult(
            reward=reward,
            success=success,
            distance_traversed=total_distance,
            final_state=info,
            trajectory=np.array(trajectory)
        )
    
    def get_frame_description(self) -> str:
        """Return the coordinate frame description for the VLM."""
        return """
(Frame Clarification) In the world frame:
- Positive X: Towards the front (away from robot base)
- Negative X: Towards the back (towards robot base)
- Positive Y: Towards the left
- Negative Y: Towards the right
- Positive Z: Up, towards the ceiling
- Negative Z: Down, towards the floor

The robot base is at the origin. The table is in front of the robot.
The tool_mount link orientation matches the world frame when all joints are at zero.
"""
