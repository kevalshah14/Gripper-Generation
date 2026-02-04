"""
DrillPress task environment.

Task: Operate a drill press by turning the spindle handle to lower the drill bit.
This requires a specialized gripper/tool that can grip and rotate the handle.
"""

import pybullet as p
import numpy as np
from typing import Tuple, Dict, Any
import math

from .base_env import BaseEnv


class DrillPressEnv(BaseEnv):
    """
    DrillPress task environment.
    
    A realistic drill press with a rotating spindle wheel and handle. 
    The goal is to design a tool that can grip the handle and rotate it 
    to lower the drill bit onto the workpiece.
    
    This is a challenging manipulation task requiring:
    - Gripping a cylindrical handle
    - Applying rotational force
    - Coordinated arm motion during rotation
    """
    
    def __init__(
        self,
        robot_urdf_path: str = "robot_descriptions/franka_panda/panda_with_tool_mount.urdf",
        gui: bool = False,
        drill_press_pos: Tuple[float, float, float] = (0.55, -0.15, 0.0),
        target_rotations: float = 1.0,  # Number of full rotations needed
    ):
        """
        Initialize DrillPress environment.
        
        Args:
            robot_urdf_path: Path to robot URDF
            gui: Whether to show GUI
            drill_press_pos: Position of drill press base
            target_rotations: Number of rotations to complete the task
        """
        super().__init__(robot_urdf_path=robot_urdf_path, gui=gui)
        
        self.drill_press_pos = np.array(drill_press_pos)
        self.target_rotations = target_rotations
        self.target_angle = target_rotations * 2 * math.pi
        
        # Pre-calculate key positions for description
        self.table_height = 0.05
        self.column_height = 0.45
        self.wheel_center = self.drill_press_pos + np.array([0.12, 0.12, self.table_height + 0.38])
        self.handle_radius = 0.055  # 55mm from wheel center to handle
        self.handle_diameter = 0.02  # 20mm diameter handle
        
        # Will be set in _setup_task
        self.initial_handle_pos = None
        self.handle_id = None
        
    def _setup_task(self) -> None:
        """Set up a detailed drill press scene."""
        
        # ============== WORKBENCH ==============
        # Main workbench surface
        bench_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.35, 0.25, 0.025],
            rgbaColor=[0.55, 0.35, 0.2, 1]  # Wood color
        )
        bench_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.35, 0.25, 0.025]
        )
        self.bench_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=bench_collision,
            baseVisualShapeIndex=bench_visual,
            basePosition=[self.drill_press_pos[0], self.drill_press_pos[1] + 0.1, self.table_height]
        )
        
        # Bench edge trim
        trim_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.36, 0.01, 0.04],
            rgbaColor=[0.4, 0.25, 0.15, 1]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=trim_visual,
            basePosition=[self.drill_press_pos[0], self.drill_press_pos[1] - 0.14, self.table_height - 0.015]
        )
        
        # ============== DRILL PRESS BASE ==============
        # Heavy cast iron base
        base_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.08, 0.1, 0.02],
            rgbaColor=[0.2, 0.22, 0.25, 1]  # Dark iron
        )
        base_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.08, 0.1, 0.02]
        )
        base_pos = self.drill_press_pos + np.array([0, 0, self.table_height + 0.045])
        self.base_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=base_collision,
            baseVisualShapeIndex=base_visual,
            basePosition=base_pos.tolist()
        )
        
        # ============== VERTICAL COLUMN ==============
        column_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.03,
            length=self.column_height,
            rgbaColor=[0.35, 0.37, 0.4, 1]  # Brushed steel
        )
        column_collision = p.createCollisionShape(
            p.GEOM_CYLINDER,
            radius=0.03,
            height=self.column_height
        )
        column_pos = base_pos + np.array([0, 0.05, self.column_height/2 + 0.02])
        self.column_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=column_collision,
            baseVisualShapeIndex=column_visual,
            basePosition=column_pos.tolist()
        )
        
        # ============== DRILL HEAD ASSEMBLY ==============
        # Main head housing
        head_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.06, 0.05, 0.08],
            rgbaColor=[0.25, 0.27, 0.3, 1]
        )
        head_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.06, 0.05, 0.08]
        )
        head_pos = column_pos + np.array([0.08, -0.02, 0.12])
        self.head_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=head_collision,
            baseVisualShapeIndex=head_visual,
            basePosition=head_pos.tolist()
        )
        
        # Motor housing (top)
        motor_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.045,
            length=0.12,
            rgbaColor=[0.15, 0.4, 0.15, 1]  # Green motor
        )
        motor_pos = head_pos + np.array([0, 0, 0.14])
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=motor_visual,
            basePosition=motor_pos.tolist()
        )
        
        # ============== SPINDLE AND QUILL ==============
        # Quill housing
        quill_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.02,
            length=0.1,
            rgbaColor=[0.4, 0.42, 0.45, 1]
        )
        quill_pos = head_pos + np.array([0, 0, -0.13])
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=quill_visual,
            basePosition=quill_pos.tolist()
        )
        
        # Chuck
        chuck_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.018,
            length=0.04,
            rgbaColor=[0.5, 0.52, 0.55, 1]
        )
        chuck_pos = quill_pos + np.array([0, 0, -0.07])
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=chuck_visual,
            basePosition=chuck_pos.tolist()
        )
        
        # Drill bit
        bit_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.004,
            length=0.08,
            rgbaColor=[0.7, 0.72, 0.75, 1]  # Silver
        )
        bit_pos = chuck_pos + np.array([0, 0, -0.06])
        self.bit_id = p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=bit_visual,
            basePosition=bit_pos.tolist()
        )
        
        # ============== SPINDLE WHEEL (3-spoke) ==============
        wheel_pos = head_pos + np.array([0.0, 0.08, 0.0])
        self.wheel_center = wheel_pos
        
        # Wheel hub
        hub_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.02,
            length=0.015,
            rgbaColor=[0.3, 0.3, 0.35, 1]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=hub_visual,
            basePosition=wheel_pos.tolist(),
            baseOrientation=p.getQuaternionFromEuler([math.pi/2, 0, 0])
        )
        
        # Wheel rim
        rim_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.065,
            length=0.012,
            rgbaColor=[0.25, 0.25, 0.3, 1]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=rim_visual,
            basePosition=wheel_pos.tolist(),
            baseOrientation=p.getQuaternionFromEuler([math.pi/2, 0, 0])
        )
        
        # Inner rim (visual detail)
        inner_rim_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.055,
            length=0.014,
            rgbaColor=[0.35, 0.35, 0.4, 1]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=inner_rim_visual,
            basePosition=wheel_pos.tolist(),
            baseOrientation=p.getQuaternionFromEuler([math.pi/2, 0, 0])
        )
        
        # Wheel spokes (3 spokes)
        spoke_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.025, 0.004, 0.006],
            rgbaColor=[0.3, 0.3, 0.35, 1]
        )
        for i in range(3):
            angle = i * (2 * math.pi / 3)
            spoke_offset = np.array([0.025 * math.cos(angle), 0, 0.025 * math.sin(angle)])
            p.createMultiBody(
                baseMass=0,
                baseVisualShapeIndex=spoke_visual,
                basePosition=(wheel_pos + spoke_offset).tolist(),
                baseOrientation=p.getQuaternionFromEuler([0, angle, 0])
            )
        
        # ============== HANDLE (the manipulable object) ==============
        handle_visual = p.createVisualShape(
            p.GEOM_CAPSULE,
            radius=self.handle_diameter / 2,
            length=0.06,
            rgbaColor=[0.9, 0.15, 0.1, 1]  # Bright red for visibility
        )
        handle_collision = p.createCollisionShape(
            p.GEOM_CAPSULE,
            radius=self.handle_diameter / 2,
            height=0.06
        )
        
        # Handle position - extends from wheel rim
        handle_pos = wheel_pos + np.array([0, 0.01, self.handle_radius])
        self.handle_id = p.createMultiBody(
            baseMass=0.15,
            baseCollisionShapeIndex=handle_collision,
            baseVisualShapeIndex=handle_visual,
            basePosition=handle_pos.tolist(),
            baseOrientation=p.getQuaternionFromEuler([math.pi/2, 0, 0])
        )
        
        # Handle grip (rubber coating visual)
        grip_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=self.handle_diameter / 2 + 0.002,
            length=0.04,
            rgbaColor=[0.2, 0.2, 0.2, 1]  # Black rubber
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=grip_visual,
            basePosition=handle_pos.tolist(),
            baseOrientation=p.getQuaternionFromEuler([math.pi/2, 0, 0])
        )
        
        # ============== WORKPIECE ==============
        # Metal block to be drilled
        workpiece_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.04, 0.03, 0.015],
            rgbaColor=[0.6, 0.6, 0.65, 1]  # Aluminum
        )
        workpiece_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[0.04, 0.03, 0.015]
        )
        workpiece_pos = base_pos + np.array([0.12, -0.02, 0.04])
        self.workpiece_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=workpiece_collision,
            baseVisualShapeIndex=workpiece_visual,
            basePosition=workpiece_pos.tolist()
        )
        
        # Drill mark on workpiece (target point)
        mark_visual = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=0.003,
            rgbaColor=[1.0, 0.8, 0.0, 1]  # Yellow dot
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=mark_visual,
            basePosition=(workpiece_pos + np.array([0, 0, 0.016])).tolist()
        )
        
        # ============== SAFETY FEATURES ==============
        # Depth stop rod
        stop_visual = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.004,
            length=0.15,
            rgbaColor=[0.8, 0.6, 0.1, 1]  # Brass
        )
        stop_pos = head_pos + np.array([-0.05, 0.02, -0.05])
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=stop_visual,
            basePosition=stop_pos.tolist()
        )
        
        # ============== SET DYNAMICS ==============
        p.changeDynamics(
            self.handle_id, -1,
            lateralFriction=1.2,
            spinningFriction=0.3,
            rollingFriction=0.1,
            restitution=0.1
        )
        
        # Store initial state
        self.initial_handle_pos, _ = p.getBasePositionAndOrientation(self.handle_id)
        self.initial_handle_pos = np.array(self.initial_handle_pos)
        
        # Register task objects
        self.task_objects = {
            "handle": self.handle_id,
        }
    
    def _compute_reward(self) -> Tuple[float, bool, Dict[str, Any]]:
        """
        Compute reward based on handle displacement (simulating spindle turning).
        """
        handle_pos, handle_orn = p.getBasePositionAndOrientation(self.handle_id)
        handle_pos = np.array(handle_pos)
        
        # Calculate displacement from initial position
        displacement = np.linalg.norm(handle_pos - self.initial_handle_pos)
        
        # Also consider rotation of the handle
        euler = p.getEulerFromQuaternion(handle_orn)
        rotation = abs(euler[1])
        
        # Target: move handle around arc or rotate significantly
        target_displacement = 0.08  # ~80mm arc movement
        target_rotation = math.pi / 2
        
        displacement_reward = np.clip(displacement / target_displacement, 0, 1)
        rotation_reward = np.clip(rotation / target_rotation, 0, 1)
        
        reward = max(displacement_reward, rotation_reward)
        success = reward >= 0.8
        
        if success:
            reward = 1.0
        
        info = {
            "handle_position": handle_pos.tolist(),
            "displacement": displacement,
            "rotation_deg": math.degrees(rotation),
            "displacement_reward": displacement_reward,
            "rotation_reward": rotation_reward,
        }
        
        return reward, success, info
    
    def get_task_description(self) -> str:
        """Return task description for VLM."""
        return f"""
## Task: Operate Drill Press Spindle

A floor-standing drill press is mounted on a workbench. To lower the drill bit 
onto the workpiece, you must turn the spindle wheel using its handle.

**Drill Press Layout:**
- Workbench height: {self.table_height}m above ground
- Spindle wheel center: ({self.wheel_center[0]:.2f}, {self.wheel_center[1]:.2f}, {self.wheel_center[2]:.2f})
- The wheel has 3 spokes with one RED HANDLE extending outward

**Handle Specifications:**
- Handle offset from wheel center: {self.handle_radius:.3f}m ({self.handle_radius*1000:.0f}mm)
- Handle diameter: {self.handle_diameter*1000:.0f}mm (cylindrical with rubber grip)
- Handle orientation: Extends horizontally from wheel edge

**Goal:** Turn the spindle wheel by pushing/pulling the handle in a circular arc.
Target: {self.target_rotations} rotation(s) = {math.degrees(self.target_angle):.0f}°

**Challenges:**
1. Handle is cylindrical - tool must securely engage it
2. Handle follows circular arc as wheel turns
3. Must maintain contact while applying tangential force
4. Re-gripping may be needed for full rotation

**Tool Design Strategies:**
- HOOK: Curved shape to wrap behind handle, pull in arc
- C-CLAMP: Two jaws to squeeze handle from sides  
- FORK: Two-prong design to capture handle between tines
- PADDLE: Flat surface to push handle along arc
"""
    
    def get_environment_code(self) -> str:
        """Return environment setup code for VLM."""
        return f"""
class DrillPressEnv(BaseEnv):
    # Drill press on workbench
    table_height = {self.table_height}
    
    # Spindle wheel (3-spoke design)
    wheel_center = {[round(x, 3) for x in self.wheel_center.tolist()]}
    wheel_radius = 0.065  # outer rim
    
    # RED HANDLE (target to manipulate)
    handle_offset = {self.handle_radius}  # from wheel center
    handle_diameter = {self.handle_diameter}
    handle_length = 0.06
    
    # The handle starts at:
    handle_position = wheel_center + [0, 0.01, {self.handle_radius}]
    
    # Workpiece below drill bit (aluminum block)
    workpiece_position = [{self.drill_press_pos[0] + 0.12:.2f}, {self.drill_press_pos[1] - 0.02:.2f}, {self.table_height + 0.085:.2f}]
    
    def compute_reward(self):
        # Reward based on handle displacement/rotation
        displacement = distance(handle_pos, initial_handle_pos)
        return clip(displacement / 0.08, 0, 1)
"""
