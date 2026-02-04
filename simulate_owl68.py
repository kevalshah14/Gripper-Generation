"""
PyBullet simulation for the OWL 6.8 robot arm.
"""

import pybullet as p
import pybullet_data
import time
import os
import re
import tempfile
import shutil
import numpy as np


def resolve_package_paths(urdf_path: str, package_dir: str) -> str:
    """
    Create a temporary URDF file with resolved package:// paths.
    
    Args:
        urdf_path: Path to the original URDF file
        package_dir: Path to the package directory (owl_68_robot_description)
    
    Returns:
        Path to the temporary URDF with resolved paths
    """
    with open(urdf_path, 'r') as f:
        urdf_content = f.read()
    
    # Replace package:// paths with absolute paths
    package_name = os.path.basename(package_dir)
    pattern = f'package://{package_name}/'
    replacement = package_dir + '/'
    
    resolved_content = urdf_content.replace(pattern, replacement)
    
    # Create a temporary file
    temp_dir = tempfile.mkdtemp()
    temp_urdf = os.path.join(temp_dir, 'robot.urdf')
    
    with open(temp_urdf, 'w') as f:
        f.write(resolved_content)
    
    return temp_urdf, temp_dir


class OWL68Simulator:
    """Simulator for the OWL 6.8 robot arm."""
    
    def __init__(self, urdf_variant: str = "owl_68.urdf", gui: bool = True):
        """
        Initialize the simulator.
        
        Args:
            urdf_variant: Which URDF to load. Options:
                - "owl_68.urdf" (base robot)
                - "owl_68_hand_e.urdf" (with Hand-E gripper)
                - "owl_68_suction.urdf" (with suction gripper)
                - "owl_68_gripper_drill.urdf" (with drill gripper)
            gui: Whether to show the GUI
        """
        self.gui = gui
        self.urdf_variant = urdf_variant
        self.temp_dir = None
        
        # Get paths
        self.script_dir = os.path.dirname(os.path.abspath(__file__))
        self.package_dir = os.path.join(self.script_dir, "owl_68_robot_description")
        self.urdf_path = os.path.join(self.package_dir, "urdf", urdf_variant)
        
        # Joint information (will be populated after loading)
        self.robot_id = None
        self.joint_indices = {}
        self.joint_limits = {}
        self.num_joints = 0
        
        # Joint names for the OWL 6.8
        self.controllable_joints = ["BJ", "SJ", "EJ", "W1J", "W2J", "W3J"]
        
    def connect(self):
        """Connect to the physics server."""
        if self.gui:
            self.physics_client = p.connect(p.GUI)
            # Set up camera for better view
            p.resetDebugVisualizerCamera(
                cameraDistance=1.5,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0, 0, 0.5]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        # Set additional search path for PyBullet data
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        
    def setup_environment(self):
        """Set up the simulation environment."""
        # Set gravity
        p.setGravity(0, 0, -9.81)
        
        # Load ground plane
        self.plane_id = p.loadURDF("plane.urdf")
        
        # Set time step
        p.setTimeStep(1/240)
        
    def load_robot(self):
        """Load the robot URDF."""
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        
        # Resolve package:// paths
        temp_urdf, self.temp_dir = resolve_package_paths(self.urdf_path, self.package_dir)
        
        # Load robot
        self.robot_id = p.loadURDF(
            temp_urdf,
            basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
            useFixedBase=True
        )
        
        # Get joint information
        self.num_joints = p.getNumJoints(self.robot_id)
        print(f"\nLoaded robot with {self.num_joints} joints:")
        print("-" * 50)
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            lower_limit = joint_info[8]
            upper_limit = joint_info[9]
            
            # Map joint types
            joint_type_names = {
                p.JOINT_REVOLUTE: "revolute",
                p.JOINT_PRISMATIC: "prismatic",
                p.JOINT_FIXED: "fixed",
                p.JOINT_SPHERICAL: "spherical",
            }
            
            type_name = joint_type_names.get(joint_type, "unknown")
            
            if joint_type != p.JOINT_FIXED:
                self.joint_indices[joint_name] = i
                self.joint_limits[joint_name] = (lower_limit, upper_limit)
                print(f"  [{i}] {joint_name}: {type_name}, limits: [{lower_limit:.2f}, {upper_limit:.2f}] rad")
            else:
                print(f"  [{i}] {joint_name}: {type_name}")
        
        print("-" * 50)
        
    def get_joint_states(self) -> dict:
        """Get current joint positions and velocities."""
        states = {}
        for name, idx in self.joint_indices.items():
            pos, vel, _, _ = p.getJointState(self.robot_id, idx)
            states[name] = {"position": pos, "velocity": vel}
        return states
    
    def set_joint_positions(self, positions: dict, use_control: bool = True):
        """
        Set joint positions.
        
        Args:
            positions: Dict mapping joint names to target positions
            use_control: If True, use position control. If False, reset directly.
        """
        for name, pos in positions.items():
            if name in self.joint_indices:
                idx = self.joint_indices[name]
                if use_control:
                    p.setJointMotorControl2(
                        self.robot_id, idx,
                        p.POSITION_CONTROL,
                        targetPosition=pos,
                        force=500,
                        maxVelocity=2.0
                    )
                else:
                    p.resetJointState(self.robot_id, idx, pos)
    
    def set_all_joints(self, positions: list, use_control: bool = True):
        """
        Set all controllable joints at once.
        
        Args:
            positions: List of 6 joint positions [BJ, SJ, EJ, W1J, W2J, W3J]
            use_control: If True, use position control. If False, reset directly.
        """
        if len(positions) != len(self.controllable_joints):
            raise ValueError(f"Expected {len(self.controllable_joints)} positions, got {len(positions)}")
        
        pos_dict = dict(zip(self.controllable_joints, positions))
        self.set_joint_positions(pos_dict, use_control)
    
    def get_end_effector_pose(self) -> tuple:
        """Get the end effector (tcp) position and orientation."""
        # Find the tcp link index
        tcp_idx = None
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            link_name = joint_info[12].decode('utf-8')
            if link_name == "tcp":
                tcp_idx = i
                break
        
        if tcp_idx is not None:
            state = p.getLinkState(self.robot_id, tcp_idx)
            position = state[0]
            orientation = state[1]
            return position, orientation
        return None, None
    
    def step(self):
        """Step the simulation."""
        p.stepSimulation()
    
    def add_debug_sliders(self) -> dict:
        """Add debug sliders for joint control (GUI only)."""
        sliders = {}
        if self.gui:
            for name in self.controllable_joints:
                if name in self.joint_limits:
                    lower, upper = self.joint_limits[name]
                    slider_id = p.addUserDebugParameter(
                        name, lower, upper, 0
                    )
                    sliders[name] = slider_id
        return sliders
    
    def read_sliders(self, sliders: dict) -> dict:
        """Read values from debug sliders."""
        values = {}
        for name, slider_id in sliders.items():
            values[name] = p.readUserDebugParameter(slider_id)
        return values
    
    def disconnect(self):
        """Disconnect from the physics server and cleanup."""
        try:
            if p.isConnected():
                p.disconnect()
        except p.error:
            pass  # Already disconnected
        
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


def demo_joint_motion(sim: OWL68Simulator, fast: bool = False):
    """Demo: Move each joint through its range."""
    print("\nDemo: Moving joints through their ranges...")
    
    # Simulation steps per movement
    steps = 50 if fast else 200
    sleep_time = 0 if fast else 1/120  # Faster visual update
    
    for joint_name in sim.controllable_joints:
        if joint_name not in sim.joint_limits:
            continue
        
        # Check if still connected
        if sim.gui and not p.isConnected():
            print("\nPyBullet window closed.")
            return
            
        lower, upper = sim.joint_limits[joint_name]
        
        print(f"  Moving {joint_name}...")
        
        # Move to lower limit
        sim.set_joint_positions({joint_name: lower})
        for _ in range(steps):
            sim.step()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Move to upper limit
        sim.set_joint_positions({joint_name: upper})
        for _ in range(steps):
            sim.step()
            if sleep_time > 0:
                time.sleep(sleep_time)
        
        # Return to center
        sim.set_joint_positions({joint_name: 0})
        for _ in range(steps):
            sim.step()
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    print("\nDemo complete!")


def interactive_mode(sim: OWL68Simulator):
    """Run interactive mode with sliders."""
    print("\nInteractive mode - use sliders to control joints")
    print("Close the PyBullet window or press Ctrl+C to exit")
    
    sliders = sim.add_debug_sliders()
    
    # Enable real-time simulation for smoother interaction
    p.setRealTimeSimulation(1)
    
    # Track debug text ID to replace instead of creating new ones
    tcp_text_id = None
    
    try:
        while True:
            # Check if GUI is still connected
            if not p.isConnected():
                print("\nPyBullet window closed.")
                break
            
            try:
                # Read slider values
                joint_values = sim.read_sliders(sliders)
                
                # Apply to robot
                sim.set_joint_positions(joint_values)
                
                # Get end effector pose and display
                pos, orn = sim.get_end_effector_pose()
                if pos:
                    text = f"TCP: ({pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f})"
                    if tcp_text_id is None:
                        # Create text for the first time
                        tcp_text_id = p.addUserDebugText(
                            text,
                            [0, 0, 1.2],
                            textColorRGB=[1, 1, 1]
                        )
                    else:
                        # Replace existing text
                        tcp_text_id = p.addUserDebugText(
                            text,
                            [0, 0, 1.2],
                            textColorRGB=[1, 1, 1],
                            replaceItemUniqueId=tcp_text_id
                        )
                
                # Small sleep to prevent CPU overload
                time.sleep(1/60)
                
            except p.error:
                # GUI was closed
                print("\nPyBullet window closed.")
                break
            
    except KeyboardInterrupt:
        print("\nExiting interactive mode...")


def main():
    """Main function to run the simulation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="OWL 6.8 Robot Simulation")
    parser.add_argument(
        "--urdf", 
        type=str, 
        default="owl_68.urdf",
        choices=["owl_68.urdf", "owl_68_hand_e.urdf", "owl_68_suction.urdf", "owl_68_gripper_drill.urdf"],
        help="Which URDF variant to load"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="interactive",
        choices=["interactive", "demo"],
        help="Simulation mode"
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Run without GUI (headless mode)"
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run demo mode faster (skip visual delays)"
    )
    
    args = parser.parse_args()
    
    # Create simulator
    sim = OWL68Simulator(urdf_variant=args.urdf, gui=not args.no_gui)
    
    try:
        # Initialize
        sim.connect()
        sim.setup_environment()
        sim.load_robot()
        
        # Run selected mode
        if args.mode == "demo":
            demo_joint_motion(sim, fast=args.fast)
        else:
            interactive_mode(sim)
            
    finally:
        sim.disconnect()


if __name__ == "__main__":
    main()
