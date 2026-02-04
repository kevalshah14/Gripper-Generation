#!/usr/bin/env python3
"""
Generate a tool for a task and visualize it on the robot.

Usage:
    uv run visualize_tool.py --task "push a cube" --object "red cube at (0.7, 0, 0.06)"
    uv run visualize_tool.py --task "scoop spheres" --object "small balls in a container"
"""

import argparse
import sys
import pybullet as p
import pybullet_data
import time
import numpy as np
import os
from pathlib import Path
from typing import List
from pydantic import BaseModel, Field

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from vlmgineer.prompts import (
    TOOL_SPECIFICATION_PROMPT,
    ACTION_SPECIFICATION_PROMPT,
    FRAME_CLARIFICATION_PROMPT
)


# Pydantic schema for structured output
class Waypoint(BaseModel):
    x: float = Field(description="X position of tool_mount in world frame (meters)")
    y: float = Field(description="Y position of tool_mount in world frame (meters)")
    z: float = Field(description="Z position of tool_mount in world frame (meters)")
    roll: float = Field(description="Roll angle in radians")
    pitch: float = Field(description="Pitch angle in radians") 
    yaw: float = Field(description="Yaw angle in radians")
    gripper: float = Field(description="Gripper state: 0=open, 1=closed")


class ToolDesign(BaseModel):
    strategy: str = Field(description="Brief description of the tool design strategy (1-2 sentences)")
    urdf: str = Field(description="Complete URDF XML for the tool links and joints. Must attach to 'tool_mount' link.")
    waypoints: List[Waypoint] = Field(description="Sequence of waypoints for the action. At least 3 waypoints.")


def generate_tool(task: str, object_desc: str, model: str = "gemini-2.5-flash") -> tuple[str, np.ndarray, str]:
    """
    Generate a tool design for a task using structured output.
    
    Returns:
        (tool_urdf, actions, description)
    """
    from google import genai
    
    print(f"\n{'='*60}")
    print("Generating Tool Design (Structured Output)")
    print('='*60)
    print(f"Task: {task}")
    print(f"Object: {object_desc}")
    print(f"Model: {model}")
    print('='*60)
    
    # Build prompt
    prompt = f"""You are a robotic tool designer. Design a tool to help a robot accomplish a task.

## Task
{task}

## Object to Manipulate
{object_desc}

## Robot Setup
- Robot: Franka Panda arm at origin (0, 0, 0)
- Tool mount: "tool_mount" link on robot end-effector
- Workspace: ~0.8m radius around robot
- Table height: Z = 0.04m

{TOOL_SPECIFICATION_PROMPT}

{ACTION_SPECIFICATION_PROMPT}

{FRAME_CLARIFICATION_PROMPT}

## Your Task

Design ONE tool and ONE action sequence to accomplish the task.

For the URDF, provide complete <link> and <joint> elements. The first joint MUST have parent="tool_mount".

For waypoints, provide at least 5 waypoints that:
1. Start above the object
2. Approach the object
3. Interact with the object (push/scoop/etc)
4. Complete the manipulation
5. Retract to safe position

Be creative! The tool should be specialized for this specific task."""

    # Load API key (same logic as vlmgineer config)
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("API_KEY")
    if not api_key:
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k in ("API_KEY", "GOOGLE_API_KEY"):
                            api_key = v
                            break
    
    if not api_key:
        print("ERROR: No API key found. Set GOOGLE_API_KEY in .env file.")
        sys.exit(1)
    
    # Query VLM with structured output
    client = genai.Client(api_key=api_key)
    
    print("\nQuerying VLM with structured output...")
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config={
            "response_mime_type": "application/json",
            "response_json_schema": ToolDesign.model_json_schema(),
        },
    )
    
    print(f"Response received ({len(response.text):,} chars)")
    
    # Parse structured response
    print(f"\nParsing structured response...")
    
    try:
        design = ToolDesign.model_validate_json(response.text)
        
        # Convert waypoints to numpy array
        actions = np.array([
            [wp.x, wp.y, wp.z, wp.roll, wp.pitch, wp.yaw, wp.gripper]
            for wp in design.waypoints
        ])
        
        print(f"\n✓ Tool generated:")
        print(f"  - URDF: {len(design.urdf)} chars")
        print(f"  - Actions: {actions.shape}")
        print(f"  - Description: {design.strategy}")
        
        return design.urdf, actions, design.strategy
        
    except Exception as e:
        print(f"\n❌ ERROR: Failed to parse structured response: {e}")
        print(f"\n--- Raw Response ---")
        print(response.text[:3000])
        sys.exit(1)


def visualize_tool(tool_urdf: str, actions: np.ndarray, description: str):
    """Visualize the tool on the robot in PyBullet."""
    print(f"\n{'='*60}")
    print("Starting Visualization")
    print('='*60)
    
    # Connect to PyBullet
    physics_client = p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1/240)
    
    # Camera setup
    p.resetDebugVisualizerCamera(
        cameraDistance=1.2,
        cameraYaw=45,
        cameraPitch=-30,
        cameraTargetPosition=[0.5, 0, 0.2]
    )
    
    # Load ground plane
    p.loadURDF("plane.urdf")
    
    # Load table
    table_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.02])
    table_visual = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.6, 0.02], rgbaColor=[0.6, 0.4, 0.2, 1])
    p.createMultiBody(baseMass=0, baseCollisionShapeIndex=table_collision, 
                     baseVisualShapeIndex=table_visual, basePosition=[0.5, 0, 0.02])
    
    # Load robot with tool
    print("\nLoading robot with tool...")
    robot_urdf_path = Path(__file__).parent / "robot_descriptions/franka_panda/panda_with_tool_mount.urdf"
    robot_descriptions_dir = Path(__file__).parent / "robot_descriptions"
    
    # Read and merge URDF
    with open(robot_urdf_path) as f:
        robot_urdf = f.read()
    
    # Insert tool before </robot>
    combined_urdf = robot_urdf.replace("</robot>", f"{tool_urdf}\n</robot>")
    
    # Replace package:// paths with pybullet_data path (where Franka meshes are bundled)
    pybullet_data_dir = pybullet_data.getDataPath()
    combined_urdf = combined_urdf.replace(
        "package://franka_panda/",
        pybullet_data_dir + "/franka_panda/"
    )
    
    # Save temp file
    temp_urdf = robot_descriptions_dir / "temp_robot_with_tool.urdf"
    with open(temp_urdf, 'w') as f:
        f.write(combined_urdf)
    
    robot_id = p.loadURDF(str(temp_urdf), basePosition=[0, 0, 0], useFixedBase=True)
    
    # Get joint info
    num_joints = p.getNumJoints(robot_id)
    joint_indices = {}
    tool_link_index = None
    
    for i in range(num_joints):
        joint_info = p.getJointInfo(robot_id, i)
        joint_name = joint_info[1].decode('utf-8')
        link_name = joint_info[12].decode('utf-8')
        
        if link_name == "tool_mount":
            tool_link_index = i
        
        if joint_info[2] != p.JOINT_FIXED:
            joint_indices[joint_name] = i
    
    print(f"Robot loaded: {num_joints} joints, tool_link_index={tool_link_index}")
    print(f"\nTool: {description}")
    print(f"\nAction sequence: {len(actions)} waypoints")
    
    # Display instructions
    print("\n" + "="*60)
    print("CONTROLS")
    print("="*60)
    print("  Space: Start/pause action execution")
    print("  R:     Reset to initial pose")
    print("  Q:     Quit")
    print("="*60)
    
    # Get controllable joints
    controllable_joints = [
        "panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
        "panda_joint5", "panda_joint6", "panda_joint7"
    ]
    
    # Execute actions
    waypoint_idx = 0
    paused = True
    steps_per_waypoint = 240  # 1 second per waypoint
    step_counter = 0
    
    print("\nPress SPACE to start action execution...")
    
    try:
        while True:
            # Check keyboard
            keys = p.getKeyboardEvents()
            if ord('q') in keys and keys[ord('q')] & p.KEY_WAS_TRIGGERED:
                break
            if ord(' ') in keys and keys[ord(' ')] & p.KEY_WAS_TRIGGERED:
                paused = not paused
                if not paused:
                    print(f"\n▶ Executing waypoints...")
                else:
                    print(f"\n⏸ Paused at waypoint {waypoint_idx}/{len(actions)}")
            if ord('r') in keys and keys[ord('r')] & p.KEY_WAS_TRIGGERED:
                waypoint_idx = 0
                step_counter = 0
                print("\n↻ Reset to start")
            
            # Execute action if not paused
            if not paused and waypoint_idx < len(actions):
                if step_counter == 0:
                    print(f"  → Waypoint {waypoint_idx + 1}/{len(actions)}: {actions[waypoint_idx][:3]}")
                
                waypoint = actions[waypoint_idx]
                position = waypoint[:3]
                orientation_euler = waypoint[3:6]
                gripper = waypoint[6] if len(waypoint) > 6 else 0
                
                # Convert euler to quaternion
                orientation = p.getQuaternionFromEuler(orientation_euler)
                
                # IK
                if tool_link_index is not None:
                    joint_poses = p.calculateInverseKinematics(
                        robot_id, tool_link_index, position, orientation
                    )
                    
                    # Apply to controllable joints
                    for i, joint_name in enumerate(controllable_joints):
                        if joint_name in joint_indices and i < len(joint_poses):
                            idx = joint_indices[joint_name]
                            p.setJointMotorControl2(
                                robot_id, idx, p.POSITION_CONTROL,
                                targetPosition=joint_poses[i], force=500
                            )
                
                step_counter += 1
                if step_counter >= steps_per_waypoint:
                    waypoint_idx += 1
                    step_counter = 0
                    
                    if waypoint_idx >= len(actions):
                        print("✓ Action sequence complete!")
                        paused = True
            
            p.stepSimulation()
            time.sleep(1/240)
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    finally:
        p.disconnect()
        import os
        os.unlink(temp_urdf)


def main():
    parser = argparse.ArgumentParser(
        description="Generate and visualize a robot tool for a task",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run visualize_tool.py --task "push a cube" --object "red cube at (0.7, 0, 0.06)"
  uv run visualize_tool.py --task "scoop spheres" --object "small balls"
  uv run visualize_tool.py --task "lift a box" --object "heavy box"
        """
    )
    
    parser.add_argument("--task", required=True, help="Task description (e.g., 'push a cube')")
    parser.add_argument("--object", required=True, help="Object description (e.g., 'red cube at X')")
    parser.add_argument("--model", default="gemini-2.5-flash", help="VLM model to use")
    
    args = parser.parse_args()
    
    # Generate tool
    tool_urdf, actions, description = generate_tool(args.task, args.object, args.model)
    
    # Visualize
    visualize_tool(tool_urdf, actions, description)


if __name__ == "__main__":
    main()
