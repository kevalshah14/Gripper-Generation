"""
MoveBall task environment.

Task: Move a ball from one side of the workspace to the other.
"""

import pybullet as p
import numpy as np
from typing import Tuple, Dict, Any

from .base_env import BaseEnv


class MoveBallEnv(BaseEnv):
    """
    MoveBall task environment.
    
    A ball is on the left side of the table. Move it to the right side.
    The ball can roll, making this a challenging control task.
    """
    
    def __init__(
        self,
        robot_urdf_path: str = "robot_descriptions/franka_panda/panda_with_tool_mount.urdf",
        gui: bool = False,
        ball_start_pos: Tuple[float, float, float] = (0.45, 0.2, 0.07),
        target_pos: Tuple[float, float, float] = (0.45, -0.2, 0.07),
        ball_radius: float = 0.03,
    ):
        """
        Initialize MoveBall environment.
        
        Args:
            robot_urdf_path: Path to robot URDF
            gui: Whether to show GUI
            ball_start_pos: Starting position of the ball (left side)
            target_pos: Target position (right side)
            ball_radius: Radius of the ball
        """
        super().__init__(robot_urdf_path=robot_urdf_path, gui=gui)
        
        self.ball_start_pos = np.array(ball_start_pos)
        self.target_pos = np.array(target_pos)
        self.ball_radius = ball_radius
        
        self.ball_id = None
        self.initial_distance = np.linalg.norm(
            self.ball_start_pos[:2] - self.target_pos[:2]
        )
    
    def _setup_task(self) -> None:
        """Set up the ball and target markers."""
        # Create the ball
        ball_collision = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=self.ball_radius
        )
        ball_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=self.ball_radius,
            rgbaColor=[0.9, 0.2, 0.2, 1]  # Red
        )
        self.ball_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=ball_collision,
            baseVisualShapeIndex=ball_visual,
            basePosition=self.ball_start_pos.tolist()
        )
        
        # Set ball dynamics
        p.changeDynamics(
            self.ball_id, -1,
            lateralFriction=0.5,
            rollingFriction=0.01,
            restitution=0.5
        )
        
        # Create target marker
        target_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.06,
            length=0.005,
            rgbaColor=[0.2, 0.8, 0.2, 0.5]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=target_visual,
            basePosition=[self.target_pos[0], self.target_pos[1], 0.045]
        )
        
        # Create start marker
        start_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.06,
            length=0.005,
            rgbaColor=[0.8, 0.8, 0.2, 0.3]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=start_visual,
            basePosition=[self.ball_start_pos[0], self.ball_start_pos[1], 0.044]
        )
        
        self.task_objects = {"ball": self.ball_id}
    
    def _compute_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute reward based on ball position and velocity.
        
        Returns:
            reward: 0-1 based on distance to target (with velocity penalty)
            success: True if ball is at target
            info: Additional information
        """
        # Get ball state
        ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
        ball_vel, _ = p.getBaseVelocity(self.ball_id)
        
        ball_pos = np.array(ball_pos)
        ball_vel = np.array(ball_vel)
        
        # Distance to target
        current_distance = np.linalg.norm(ball_pos[:2] - self.target_pos[:2])
        
        # Distance improvement reward
        distance_reward = np.clip(
            (self.initial_distance - current_distance) / self.initial_distance,
            0, 1
        )
        
        # Velocity penalty (ball should be relatively still at target)
        speed = np.linalg.norm(ball_vel)
        velocity_penalty = min(speed * 0.2, 0.3)  # Max 0.3 penalty
        
        reward = max(0, distance_reward - velocity_penalty)
        
        # Success if at target with low velocity
        success = current_distance < 0.05 and speed < 0.1
        if success:
            reward = 1.0
        
        info = {
            "ball_position": ball_pos.tolist(),
            "ball_velocity": ball_vel.tolist(),
            "current_distance": current_distance,
            "speed": speed,
        }
        
        return reward, success, info
    
    def get_task_description(self) -> str:
        """Return task description for VLM."""
        return f"""
## Task: Move Ball

A red ball (radius: {self.ball_radius}m) is on the left side of the table at 
position ({self.ball_start_pos[0]:.2f}, {self.ball_start_pos[1]:.2f}, {self.ball_start_pos[2]:.2f}).

The goal is to move the ball to the right side at target position
({self.target_pos[0]:.2f}, {self.target_pos[1]:.2f}, {self.target_pos[2]:.2f}).

**Challenge**: The ball can roll freely, making precise control difficult.
If pushed too hard, it may overshoot or roll off the table.

**Hints**:
- A curved/cupped tool could cradle the ball while moving
- Gentle, controlled pushes are better than fast movements
- A tool with walls could prevent the ball from rolling away
- Consider the ball's tendency to bounce and roll
"""
    
    def get_environment_code(self) -> str:
        """Return environment setup code for VLM."""
        return f"""
class MoveBallEnv(BaseEnv):
    def __init__(self):
        self.ball_start_pos = {self.ball_start_pos.tolist()}
        self.target_pos = {self.target_pos.tolist()}
        self.ball_radius = {self.ball_radius}
        
    def _setup_task(self):
        # Red ball (can roll)
        self.ball_id = create_sphere(
            radius=self.ball_radius,
            position=self.ball_start_pos,
            mass=0.05,
            color=[0.9, 0.2, 0.2, 1]
        )
    
    def compute_reward(self):
        ball_pos = get_position(self.ball_id)
        ball_vel = get_velocity(self.ball_id)
        
        distance_to_target = distance_xy(ball_pos, self.target_pos)
        speed = magnitude(ball_vel)
        
        distance_reward = 1 - (distance_to_target / initial_distance)
        velocity_penalty = min(speed * 0.2, 0.3)
        
        return clip(distance_reward - velocity_penalty, 0, 1)
"""
