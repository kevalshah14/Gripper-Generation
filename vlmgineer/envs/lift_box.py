"""
LiftBox task environment.

Task: Lift a box above a height threshold.
"""

import pybullet as p
import numpy as np
from typing import Tuple, Dict, Any

from .base_env import BaseEnv


class LiftBoxEnv(BaseEnv):
    """
    LiftBox task environment.
    
    A box sits on the table. The goal is to lift it above a specified height.
    """
    
    def __init__(
        self,
        robot_urdf_path: str = "robot_descriptions/franka_panda/panda_with_tool_mount.urdf",
        gui: bool = False,
        box_pos: Tuple[float, float, float] = (0.45, 0.0, 0.08),
        box_size: Tuple[float, float, float] = (0.08, 0.08, 0.06),
        target_height: float = 0.25,
    ):
        """
        Initialize LiftBox environment.
        
        Args:
            robot_urdf_path: Path to robot URDF
            gui: Whether to show GUI
            box_pos: Position of the box
            box_size: Size of the box [x, y, z]
            target_height: Height threshold for success
        """
        super().__init__(robot_urdf_path=robot_urdf_path, gui=gui)
        
        self.box_pos = np.array(box_pos)
        self.box_size = np.array(box_size)
        self.target_height = target_height
        
        self.box_id = None
        self.initial_height = box_pos[2]
    
    def _setup_task(self) -> None:
        """Set up the box."""
        # Create the box
        box_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(self.box_size / 2).tolist()
        )
        box_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=(self.box_size / 2).tolist(),
            rgbaColor=[0.6, 0.4, 0.2, 1]  # Brown
        )
        self.box_id = p.createMultiBody(
            baseMass=0.3,
            baseCollisionShapeIndex=box_collision,
            baseVisualShapeIndex=box_visual,
            basePosition=self.box_pos.tolist()
        )
        
        # Set friction
        p.changeDynamics(self.box_id, -1, lateralFriction=0.8)
        
        # Create height marker (visual line)
        marker_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.2, 0.001, 0.001],
            rgbaColor=[0, 1, 0, 0.8]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=marker_visual,
            basePosition=[0.45, 0, self.target_height]
        )
        
        self.task_objects = {"box": self.box_id}
    
    def _compute_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute reward based on box height.
        
        Returns:
            reward: 0-1 based on height progress
            success: True if box is above target height
            info: Additional information
        """
        # Get current box position
        box_pos, _ = p.getBasePositionAndOrientation(self.box_id)
        current_height = box_pos[2]
        
        # Reward is height progress towards target
        height_range = self.target_height - self.initial_height
        height_progress = current_height - self.initial_height
        reward = np.clip(height_progress / height_range, 0, 1)
        
        # Success if above target
        success = current_height >= self.target_height
        if success:
            reward = 1.0
        
        info = {
            "box_position": list(box_pos),
            "current_height": current_height,
            "target_height": self.target_height,
        }
        
        return reward, success, info
    
    def get_task_description(self) -> str:
        """Return task description for VLM."""
        return f"""
## Task: Lift Box

A brown box (size: {self.box_size[0]:.2f}m x {self.box_size[1]:.2f}m x {self.box_size[2]:.2f}m) 
sits on the table at position ({self.box_pos[0]:.2f}, {self.box_pos[1]:.2f}, {self.box_pos[2]:.2f}).

The goal is to lift the box above {self.target_height}m height.

**Challenge**: The robot doesn't have a gripper that can grasp the box directly.
You need to design a tool that can lift the box from below or hook onto it.

**Hints**:
- A flat platform/spatula could scoop under the box
- A fork-like tool could lift from the sides
- A cage structure could trap and lift the box
- Consider stability during lifting
"""
    
    def get_environment_code(self) -> str:
        """Return environment setup code for VLM."""
        return f"""
class LiftBoxEnv(BaseEnv):
    def __init__(self):
        self.box_pos = {self.box_pos.tolist()}
        self.box_size = {self.box_size.tolist()}
        self.target_height = {self.target_height}
        
    def _setup_task(self):
        # Brown box on table
        self.box_id = create_box(
            size=self.box_size,
            position=self.box_pos,
            mass=0.3,
            color=[0.6, 0.4, 0.2, 1]
        )
    
    def compute_reward(self):
        box_pos = get_position(self.box_id)
        height = box_pos[2]
        reward = (height - initial_height) / (target_height - initial_height)
        return clip(reward, 0, 1)
"""
