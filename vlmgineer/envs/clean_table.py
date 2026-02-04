"""
CleanTable task environment.

Task: Push multiple small cubes into a target zone.
"""

import pybullet as p
import numpy as np
from typing import Tuple, Dict, Any, List

from .base_env import BaseEnv


class CleanTableEnv(BaseEnv):
    """
    CleanTable task environment.
    
    Multiple small cubes (representing clutter/dust) are scattered on the table.
    The goal is to push them all into a target collection zone.
    """
    
    def __init__(
        self,
        robot_urdf_path: str = "robot_descriptions/franka_panda/panda_with_tool_mount.urdf",
        gui: bool = False,
        n_cubes: int = 5,
        target_zone_center: Tuple[float, float] = (0.6, 0.0),
        target_zone_radius: float = 0.1,
        cube_size: float = 0.025,
    ):
        """
        Initialize CleanTable environment.
        
        Args:
            robot_urdf_path: Path to robot URDF
            gui: Whether to show GUI
            n_cubes: Number of cubes to clean
            target_zone_center: Center of the target zone (XY)
            target_zone_radius: Radius of the target zone
            cube_size: Size of each cube
        """
        super().__init__(robot_urdf_path=robot_urdf_path, gui=gui)
        
        self.n_cubes = n_cubes
        self.target_zone_center = np.array(target_zone_center)
        self.target_zone_radius = target_zone_radius
        self.cube_size = cube_size
        
        self.cube_ids: List[int] = []
        self.initial_distances: List[float] = []
    
    def _setup_task(self) -> None:
        """Set up the cubes and target zone."""
        # Create scattered cubes with different colors
        colors = [
            [0.9, 0.2, 0.2, 1],  # Red
            [0.2, 0.9, 0.2, 1],  # Green
            [0.2, 0.2, 0.9, 1],  # Blue
            [0.9, 0.9, 0.2, 1],  # Yellow
            [0.9, 0.2, 0.9, 1],  # Magenta
            [0.2, 0.9, 0.9, 1],  # Cyan
            [0.9, 0.5, 0.2, 1],  # Orange
        ]
        
        # Scatter positions around the workspace
        np.random.seed(42)  # Reproducible positions
        scatter_center = np.array([0.4, 0.0])
        scatter_radius = 0.15
        
        for i in range(self.n_cubes):
            # Random position in circle
            angle = (i / self.n_cubes) * 2 * np.pi + np.random.uniform(-0.3, 0.3)
            r = scatter_radius * (0.5 + 0.5 * np.random.random())
            pos = [
                scatter_center[0] + r * np.cos(angle),
                scatter_center[1] + r * np.sin(angle),
                0.055 + self.cube_size / 2
            ]
            
            # Create cube
            cube_collision = p.createCollisionShape(
                p.GEOM_BOX,
                halfExtents=[self.cube_size/2] * 3
            )
            cube_visual = p.createVisualShape(
                p.GEOM_BOX,
                halfExtents=[self.cube_size/2] * 3,
                rgbaColor=colors[i % len(colors)]
            )
            cube_id = p.createMultiBody(
                baseMass=0.02,
                baseCollisionShapeIndex=cube_collision,
                baseVisualShapeIndex=cube_visual,
                basePosition=pos
            )
            
            p.changeDynamics(cube_id, -1, lateralFriction=0.4)
            self.cube_ids.append(cube_id)
            
            # Store initial distance
            dist = np.linalg.norm(np.array(pos[:2]) - self.target_zone_center)
            self.initial_distances.append(dist)
        
        # Create target zone marker
        zone_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.target_zone_radius,
            length=0.005,
            rgbaColor=[0.2, 0.8, 0.2, 0.3]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=zone_visual,
            basePosition=[self.target_zone_center[0], self.target_zone_center[1], 0.044]
        )
        
        # Register task objects
        self.task_objects = {f"cube_{i}": cid for i, cid in enumerate(self.cube_ids)}
    
    def _compute_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute reward based on average cube progress.
        
        Returns:
            reward: Average distance improvement across all cubes (0-1)
            success: True if all cubes are in target zone
            info: Additional information
        """
        total_reward = 0.0
        cubes_in_zone = 0
        cube_positions = []
        
        for i, cube_id in enumerate(self.cube_ids):
            pos, _ = p.getBasePositionAndOrientation(cube_id)
            pos = np.array(pos)
            cube_positions.append(pos.tolist())
            
            # Distance to zone center
            dist = np.linalg.norm(pos[:2] - self.target_zone_center)
            
            # Progress reward
            progress = (self.initial_distances[i] - dist) / self.initial_distances[i]
            total_reward += np.clip(progress, 0, 1)
            
            # Check if in zone
            if dist < self.target_zone_radius:
                cubes_in_zone += 1
        
        reward = total_reward / self.n_cubes
        success = cubes_in_zone == self.n_cubes
        
        if success:
            reward = 1.0
        
        info = {
            "cube_positions": cube_positions,
            "cubes_in_zone": cubes_in_zone,
            "total_cubes": self.n_cubes,
        }
        
        return reward, success, info
    
    def get_task_description(self) -> str:
        """Return task description for VLM."""
        return f"""
## Task: Clean Table

{self.n_cubes} small colored cubes (size: {self.cube_size}m) are scattered 
across the table, representing clutter that needs to be cleaned up.

The goal is to push ALL cubes into the target zone:
- Target zone center: ({self.target_zone_center[0]:.2f}, {self.target_zone_center[1]:.2f})
- Target zone radius: {self.target_zone_radius}m (green circle on table)

**Challenge**: Multiple objects need to be moved efficiently. Pushing one cube
might affect others.

**Hints**:
- A wide, flat pusher could sweep multiple cubes at once
- A curved/concave surface could collect cubes as you push
- Side guards could prevent cubes from escaping sideways
- Plan a sweeping motion that gathers cubes together
"""
    
    def get_environment_code(self) -> str:
        """Return environment setup code for VLM."""
        return f"""
class CleanTableEnv(BaseEnv):
    def __init__(self):
        self.n_cubes = {self.n_cubes}
        self.target_zone_center = {self.target_zone_center.tolist()}
        self.target_zone_radius = {self.target_zone_radius}
        self.cube_size = {self.cube_size}
        
    def _setup_task(self):
        # Scatter small cubes around the workspace
        for i in range(self.n_cubes):
            angle = (i / n_cubes) * 2 * pi + random(-0.3, 0.3)
            r = 0.15 * random(0.5, 1.0)
            pos = [0.4 + r*cos(angle), r*sin(angle), 0.055]
            
            self.cubes[i] = create_box(
                size=[{self.cube_size}] * 3,
                position=pos,
                mass=0.02
            )
        
        # Green target zone
        create_target_zone(
            center=self.target_zone_center,
            radius={self.target_zone_radius}
        )
    
    def compute_reward(self):
        total_progress = 0
        for cube in self.cubes:
            pos = get_position(cube)
            dist = distance_xy(pos, self.target_zone_center)
            progress = (initial_dist - dist) / initial_dist
            total_progress += clip(progress, 0, 1)
        
        return total_progress / self.n_cubes
"""
