"""
Tool specification prompts for VLMgineer.
"""

TOOL_SPECIFICATION_PROMPT = """Your tool design must follow these rules:

1. **Geometry**: Use only 3D boxes (rectangular prisms) for each component. No cylinders, spheres, or meshes.

2. **Format**: Output tool as URDF XML blocks that will be inserted into the robot URDF. Include both <link> and <joint> elements.

3. **Mass**: Keep each part very lightweight (mass < 0.01 kg). This ensures stable inverse kinematics. Example: `<mass value="0.001"/>`

4. **Attachment**: All tools attach to the `tool_mount` link via fixed joints. The tool_mount has the same orientation as the world frame when all robot joints are at zero.

5. **Connectivity**: Each part must be geometrically connected to its parent (no floating parts).

6. **Complexity**: More intricate, specialized tools often perform better than simple ones. Don't be afraid to add multiple components.

Example tool URDF (a simple pusher stick):
```xml
<link name="pusher_shaft">
  <inertial>
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <mass value="0.001"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
  </inertial>
  <visual>
    <origin rpy="0 0 0" xyz="0 0 0.1"/>
    <geometry>
      <box size="0.02 0.02 0.2"/>
    </geometry>
    <material name="blue">
      <color rgba="0.2 0.2 0.8 1"/>
    </material>
  </visual>
  <collision>
    <origin rpy="0 0 0" xyz="0 0 0.1"/>
    <geometry>
      <box size="0.02 0.02 0.2"/>
    </geometry>
  </collision>
</link>
<joint name="pusher_shaft_joint" type="fixed">
  <origin rpy="0 0 0" xyz="0 0 0"/>
  <parent link="tool_mount"/>
  <child link="pusher_shaft"/>
</joint>
```

Tool design tips:
- Hooks are great for pulling objects
- Wide flat surfaces help push multiple objects
- Cages/enclosures prevent objects from escaping
- Long extensions increase reach
- Angled surfaces can redirect motion
"""

TOOL_SPECIFICATION_WITH_GRIPPER_PROMPT = """Your tool design must follow these rules:

1. **Geometry**: Use only 3D boxes for each component.

2. **Format**: Output as URDF XML blocks with <link> and <joint> elements.

3. **Mass**: Keep parts lightweight (mass < 0.01 kg).

4. **Gripper Attachment**: Design tools that attach to the gripper fingers:
   - Left finger attachment: parent link = "panda_leftfinger"
   - Right finger attachment: parent link = "panda_rightfinger"
   - This allows the tool to be actuated by the Franka Panda gripper

5. **Connectivity**: Parts must connect to their parent link.

This enables gripping/grasping actions with specialized finger attachments.
"""
