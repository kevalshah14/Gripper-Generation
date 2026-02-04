"""
BringCube task environment.

Task: A red cube is placed out of reach. Use a tool to bring it to the target zone.
"""

import pybullet as p
import numpy as np
from typing import Tuple, Dict, Any

from .base_env import BaseEnv


class BringCubeEnv(BaseEnv):
    """
    BringCube task environment.
    
    A red cube is placed far from the robot. The goal is to design
    a tool that can push/drag the cube closer to the target zone.
    """
    
    def __init__(
        self,
        robot_urdf_path: str = "robot_descriptions/franka_panda/panda_with_tool_mount.urdf",
        gui: bool = False,
        cube_start_pos: Tuple[float, float, float] = (0.7, 0.0, 0.06),
        target_pos: Tuple[float, float, float] = (0.3, 0.0, 0.06),
        cube_size: float = 0.04,
    ):
        """
        Initialize BringCube environment.
        
        Args:
            robot_urdf_path: Path to robot URDF
            gui: Whether to show GUI
            cube_start_pos: Starting position of the cube
            target_pos: Target position for the cube
            cube_size: Size of the cube (edge length)
        """
        super().__init__(robot_urdf_path=robot_urdf_path, gui=gui)
        
        self.cube_start_pos = np.array(cube_start_pos)
        self.target_pos = np.array(target_pos)
        self.cube_size = cube_size
        
        # Will be set in _setup_task
        self.cube_id = None
        self.target_marker_id = None
        
        # Initial distance for reward calculation
        self.initial_distance = np.linalg.norm(
            self.cube_start_pos[:2] - self.target_pos[:2]
        )
    
    def _setup_task(self) -> None:
        """Set up the cube and target marker."""
        # Create the cube
        cube_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[self.cube_size/2] * 3
        )
        cube_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[self.cube_size/2] * 3,
            rgbaColor=[0.9, 0.2, 0.2, 1]  # Red
        )
        self.cube_id = p.createMultiBody(
            baseMass=0.1,
            baseCollisionShapeIndex=cube_collision,
            baseVisualShapeIndex=cube_visual,
            basePosition=self.cube_start_pos.tolist()
        )
        
        # Set friction
        p.changeDynamics(self.cube_id, -1, lateralFriction=0.5)
        
        # Create target marker (visual only)
        target_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.08,
            length=0.005,
            rgbaColor=[0.2, 0.8, 0.2, 0.5]  # Green, semi-transparent
        )
        self.target_marker_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=[self.target_pos[0], self.target_pos[1], 0.045]
        )
        
        # Register task objects
        self.task_objects = {"cube": self.cube_id}
    
    def _compute_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute the reward based on cube position.
        
        Returns:
            reward: 0-1 based on distance improvement
            success: True if cube is in target zone
            info: Additional information
        """
        # Get current cube position
        cube_pos, _ = p.getBasePositionAndOrientation(self.cube_id)
        cube_pos = np.array(cube_pos)
        
        # Calculate distance to target (XY plane)
        current_distance = np.linalg.norm(cube_pos[:2] - self.target_pos[:2])
        
        # Reward is improvement ratio (1 = at target, 0 = at start)
        distance_improvement = self.initial_distance - current_distance
        reward = np.clip(distance_improvement / self.initial_distance, 0, 1)
        
        # Success if within threshold
        success = current_distance < 0.05
        if success:
            reward = 1.0
        
        info = {
            "cube_position": cube_pos.tolist(),
            "current_distance": current_distance,
            "initial_distance": self.initial_distance,
        }
        
        return reward, success, info
    
    def get_task_description(self) -> str:
        """Return task description for VLM."""
        return f"""
## Task: Bring Cube

A red cube (size: {self.cube_size}m) is placed on the table at position 
({self.cube_start_pos[0]:.2f}, {self.cube_start_pos[1]:.2f}, {self.cube_start_pos[2]:.2f}).

The goal is to move the cube to the target zone at 
({self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}, {self.target_pos[2]:.2f}).

The cube is beyond the robot's normal reach, so you need to design a tool 
that can push or drag the cube towards the target.

**Challenge**: The cube is far from the robot base. A simple gripper cannot reach it.

**Hints**:
- A long reaching tool could push the cube
- A hook-shaped tool could pull the cube
- Consider the friction and sliding behavior
"""
    
    def get_environment_code(self) -> str:
        """Return environment setup code for VLM."""
        return f"""
class BringCubeEnv(BaseEnv):
    def __init__(self):
        self.cube_start_pos = {self.cube_start_pos.tolist()}
        self.target_pos = {self.target_pos.tolist()}
        self.cube_size = {self.cube_size}
        
    def _setup_task(self):
        # Red cube at starting position
        self.cube_id = create_box(
            size=[{self.cube_size}] * 3,
            position=self.cube_start_pos,
            mass=0.1,
            color=[0.9, 0.2, 0.2, 1]
        )
        
        # Green target marker
        self.target_marker = create_cylinder(
            radius=0.08, height=0.005,
            position=self.target_pos,
            color=[0.2, 0.8, 0.2, 0.5]
        )
    
    def compute_reward(self):
        cube_pos = get_position(self.cube_id)
        distance_to_target = distance_xy(cube_pos, self.target_pos)
        reward = 1 - (distance_to_target / initial_distance)
        return clip(reward, 0, 1)
"""
