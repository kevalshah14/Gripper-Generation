"""
Action specification prompts - matching the paper exactly (Appendix D.4-D.6).
"""

# Action specification without gripper (from paper Appendix D.4)
ACTION_SPECIFICATION_PROMPT = """(Action Specifications) Your tool-using action will be a Nx7 numpy array of action waypoints, where N is the number of waypoints, and each waypoint is of dimension 7 (xyz position + roll-pitch-yaw euler angle orientations + gripper state).

Your action needs to be precisely seven numbers per waypoint: [x, y, z, roll, pitch, yaw, gripper]
- Position (x, y, z) in meters
- Orientation (roll, pitch, yaw) in radians  
- Gripper: 0 = open, 1 = closed

It is important to stress this: the action waypoints are controlling the robot end-effector "tool_mount" link position and orientation. This means you have to carefully take into account the dimensions of the tool and the thickness of its parts when designing effective waypoints.

For example, if your tool extends 0.3m in the +X direction and you want the tool tip to reach an object at X=0.7:
- Set the waypoint X position to approximately 0.4 (so 0.4 + 0.3 tool length = 0.7)

The end-effector starts with the same orientation as the world frame. Use pitch (rotation about Y-axis) to tilt the tool down towards objects on the table."""

# Action diversity specification (from paper Appendix D.5)
ACTION_DIVERSITY_PROMPT = """(Desired Action Criteria Definitions) For the description below, we will call a single sequential set of waypoints in a single rollout as one "action set".

For each tool you created, the goal is to generate action sets that optimize task success and motion differentiation:

1. Task success is optimized when an action set is able to complete the task successfully.

2. Motion differentiation is optimized when there exists a large variance in the motion taken across all action sets you design for the same tool. A large variance in motion means the tool, at each time step, is located at a different location in the 3D space.

Think about how a tool can be used to interact with the object from many different sides, angles, and ways. When both conditions are met, you have successfully designed a good set of action sets."""

# Frame clarification (from paper Appendix D.6)
FRAME_CLARIFICATION_PROMPT = """(Frame Clarification) In the world frame, front/back is along the x axis, left/right is along the y axis, and up/down is along the z axis with the following directions:

Position:
- Positive X: Towards the front of the table (away from robot base, towards objects)
- Negative X: Towards the back of the table (towards robot base)
- Positive Y: Towards the left
- Negative Y: Towards the right
- Positive Z: Up, towards the ceiling
- Negative Z: Down, towards the floor

Orientation (starting from the world frame):
- Positive rotation about the X-axis (roll): tilting the end-effector head to the left
- Negative rotation about the X-axis (roll): tilting the end-effector head to the right
- Positive rotation about the Y-axis (pitch): tilting the end-effector head DOWN (towards table)
- Negative rotation about the Y-axis (pitch): tilting the end-effector head UP
- Positive rotation about the Z-axis (yaw): rotating the end-effector head counter-clockwise
- Negative rotation about the Z-axis (yaw): rotating the end-effector head clockwise

The robot base is at the origin (0, 0, 0). Objects are placed on the table in front of the robot (+X direction). The table surface is at approximately Z = 0.04m."""
