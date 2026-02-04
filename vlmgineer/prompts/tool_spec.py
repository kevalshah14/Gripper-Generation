"""
Tool specification prompts (minimal version).
"""

TOOL_SPECIFICATION_PROMPT = """## Tool URDF Rules
- Boxes only (no cylinders/spheres)
- Attach to `tool_mount` via fixed joint
- Mass < 0.01 kg per part

Example:
```xml
<link name="tool_part">
  <inertial><origin xyz="0 0 0"/><mass value="0.001"/><inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0 0 0.1"/><geometry><box size="0.02 0.02 0.2"/></geometry></visual>
  <collision><origin xyz="0 0 0.1"/><geometry><box size="0.02 0.02 0.2"/></geometry></collision>
</link>
<joint name="tool_joint" type="fixed">
  <parent link="tool_mount"/><child link="tool_part"/>
</joint>
```"""

TOOL_SPECIFICATION_WITH_GRIPPER_PROMPT = TOOL_SPECIFICATION_PROMPT
