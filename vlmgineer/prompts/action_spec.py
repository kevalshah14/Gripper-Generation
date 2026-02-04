"""
Action specification prompts for VLMgineer.
"""

ACTION_SPECIFICATION_PROMPT = """Your action will be an Nx7 numpy array of waypoints, where:
- N = number of waypoints (typically 5-15)
- Each row = [x, y, z, roll, pitch, yaw, gripper]
  - x, y, z: Position in meters (world frame)
  - roll, pitch, yaw: Orientation in radians (Euler angles)
  - gripper: 0 = open, 1 = closed (ignored for tools without gripper)

The waypoints control the `tool_mount` link position and orientation.

**IMPORTANT**: Account for the tool dimensions when planning waypoints! 
If your tool extends 0.15m in the Z direction, the tool tip will be 0.15m below the tool_mount position.

Example action array:
```python
action = np.array([
    [0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 0],   # Start position above table
    [0.5, 0.0, 0.3, 0.0, 0.5, 0.0, 0],   # Lower and tilt forward
    [0.6, 0.0, 0.15, 0.0, 0.5, 0.0, 0],  # Contact with object
    [0.4, 0.0, 0.15, 0.0, 0.5, 0.0, 0],  # Push object back
    [0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 0],   # Lift up
])
```

Workspace constraints for Franka Panda robot:
- Reachable range: ~0.855m from base
- Table surface at Z ≈ 0.04m
- Safe operating height: Z > 0.1m
- Robot base at origin (0, 0, 0)
"""

ACTION_DIVERSITY_PROMPT = """For each tool, generate diverse action sequences that:

1. **Maximize Task Success**: Design motions that accomplish the goal
2. **Maximize Motion Diversity**: Try different approaches:
   - Different approach angles
   - Different contact points
   - Different motion speeds (via waypoint density)
   - Different tool orientations

Think about how the tool can interact with objects from multiple sides, angles, and directions.

Good diversity means: at each timestep, the tool is in a different location/orientation across your action sequences.
"""

FRAME_CLARIFICATION_PROMPT = """
## Coordinate Frame Reference

In the world frame:
- **+X**: Front (away from robot base, towards the table)
- **-X**: Back (towards robot base)
- **+Y**: Left (robot's left side)
- **-Y**: Right (robot's right side)
- **+Z**: Up (towards ceiling)
- **-Z**: Down (towards floor)

Orientation (Euler angles from world frame):
- **+roll** (rotation about X): Tilts tool head left
- **-roll**: Tilts tool head right
- **+pitch** (rotation about Y): Tilts tool head down
- **-pitch**: Tilts tool head up
- **+yaw** (rotation about Z): Rotates tool counter-clockwise (from above)
- **-yaw**: Rotates tool clockwise

The robot base is at the origin. The table is in front (+X direction).
At zero joint configuration, the tool_mount faces straight out along +X.
"""
