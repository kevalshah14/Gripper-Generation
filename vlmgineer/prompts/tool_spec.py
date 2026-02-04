"""
Tool specification prompts - matching the paper exactly (Appendix D.3).
"""

# Tool specification without gripper usage (from paper Appendix D.3)
TOOL_SPECIFICATION_PROMPT = """(Tool Specifications) Your design of the tool must follow these rules:

(1) You must only use 3D rectangles (boxes) for each component.

(2) Your tool will be outputted in a URDF block format, which should be directly added to the end of a panda URDF file, before the robot closing declaration.

(3) Make sure your tools weigh very little in the URDF file, where each tool part should weigh no more than a few grams (0.001-0.01 kg). These weights do not have to be realistic, it is just for the robot inverse kinematics to have an easier time converging.

(4) Your design will be a single rigid tool, which should be attached directly to the "tool_mount" link, which you can safely assume to have the same orientation as the world frame.

(5) Any attachments you design should geometrically be directly connected to their parent links in the URDF (there should be no gaps in between!)

(6) As a general observation, you perform better when the tools you design are complex and intricate.

(7) Tools should extend in the +X direction (towards the front/objects) from the tool_mount link.

Example URDF for a 30cm long pusher tool extending forward:
```xml
<link name="pusher_arm">
  <inertial>
    <origin xyz="0.15 0 0"/>
    <mass value="0.005"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
  <visual>
    <origin xyz="0.15 0 0"/>
    <geometry><box size="0.30 0.02 0.02"/></geometry>
    <material name="grey"><color rgba="0.5 0.5 0.5 1"/></material>
  </visual>
  <collision>
    <origin xyz="0.15 0 0"/>
    <geometry><box size="0.30 0.02 0.02"/></geometry>
  </collision>
</link>
<joint name="pusher_joint" type="fixed">
  <origin xyz="0 0 0" rpy="0 0 0"/>
  <parent link="tool_mount"/>
  <child link="pusher_arm"/>
</joint>
```

Note: The origin xyz="0.15 0 0" centers the 0.30m box so it extends from 0 to 0.30m in +X."""

# Tool specification with gripper usage (from paper Appendix D.3)
TOOL_SPECIFICATION_WITH_GRIPPER_PROMPT = """(Tool Specifications) Your design of the tool must follow these rules:

(1) You must only use 3D rectangles (boxes) for each component.

(2) Your tool will be outputted in a URDF block format, which should be directly added to the end of a panda URDF file, before the robot closing declaration.

(3) Make sure your tools weigh very little in the URDF file, where each tool part should weigh no more than a few grams (0.001-0.01 kg). These weights do not have to be realistic, it is just for the robot inverse kinematics to have an easier time converging.

(4) Your design will be a pair of attachments to the robot gripper fingers (which allows the tool to be actuated with the robot gripper). You should attach the left attachment to "panda_leftfinger" and the right attachment to "panda_rightfinger".

(5) Any attachments you design should geometrically be directly connected to their parent links in the URDF (there should be no gaps in between!)

(6) As a general observation, you perform better when the tools you design are complex and intricate."""
