"""
URDF utilities for VLMgineer.

Handles parsing, validation, and merging of URDF files for tool attachment.
"""

import xml.etree.ElementTree as ET
import re
import os
import tempfile
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass


@dataclass
class URDFValidationResult:
    """Result of URDF validation."""
    
    valid: bool
    errors: List[str]
    warnings: List[str]
    
    def __str__(self) -> str:
        result = "Valid" if self.valid else "Invalid"
        if self.errors:
            result += f"\nErrors: {self.errors}"
        if self.warnings:
            result += f"\nWarnings: {self.warnings}"
        return result


def validate_tool_urdf(urdf_string: str) -> URDFValidationResult:
    """
    Validate a tool URDF string.
    
    Args:
        urdf_string: URDF XML string containing tool links and joints
        
    Returns:
        URDFValidationResult with validation status and messages
    """
    errors = []
    warnings = []
    
    # Try to parse XML
    try:
        # Wrap in a temporary root if needed
        if not urdf_string.strip().startswith('<?xml'):
            wrapped = f"<robot name='temp'>{urdf_string}</robot>"
        else:
            wrapped = urdf_string
            
        root = ET.fromstring(wrapped)
    except ET.ParseError as e:
        errors.append(f"XML parse error: {e}")
        return URDFValidationResult(valid=False, errors=errors, warnings=warnings)
    
    # Check for links
    links = root.findall('.//link')
    if len(links) == 0:
        errors.append("No links found in tool URDF")
    
    # Check for joints
    joints = root.findall('.//joint')
    if len(joints) == 0:
        errors.append("No joints found in tool URDF")
    
    # Validate each link
    for link in links:
        link_name = link.get('name', 'unnamed')
        
        # Check for visual geometry
        visual = link.find('visual')
        if visual is None:
            warnings.append(f"Link '{link_name}' has no visual geometry")
        else:
            geom = visual.find('geometry')
            if geom is None:
                warnings.append(f"Link '{link_name}' visual has no geometry")
            else:
                # Check geometry type (should be box for VLMgineer)
                box = geom.find('box')
                if box is None:
                    warnings.append(f"Link '{link_name}' uses non-box geometry (recommended: box)")
        
        # Check for collision geometry
        collision = link.find('collision')
        if collision is None:
            warnings.append(f"Link '{link_name}' has no collision geometry")
        
        # Check for inertial
        inertial = link.find('inertial')
        if inertial is not None:
            mass = inertial.find('mass')
            if mass is not None:
                mass_val = float(mass.get('value', 1.0))
                if mass_val > 0.1:
                    warnings.append(
                        f"Link '{link_name}' has large mass ({mass_val}kg). "
                        "Consider reducing for stable IK."
                    )
    
    # Validate each joint
    for joint in joints:
        joint_name = joint.get('name', 'unnamed')
        joint_type = joint.get('type', 'fixed')
        
        if joint_type != 'fixed':
            warnings.append(
                f"Joint '{joint_name}' is type '{joint_type}'. "
                "VLMgineer tools should use fixed joints."
            )
        
        # Check parent/child
        parent = joint.find('parent')
        child = joint.find('child')
        
        if parent is None:
            errors.append(f"Joint '{joint_name}' has no parent link")
        if child is None:
            errors.append(f"Joint '{joint_name}' has no child link")
    
    valid = len(errors) == 0
    return URDFValidationResult(valid=valid, errors=errors, warnings=warnings)


def merge_tool_into_robot(
    robot_urdf_path: str,
    tool_urdf: str,
    tool_mount_link: str = "tool_mount",
    output_path: Optional[str] = None
) -> str:
    """
    Merge a tool URDF into a robot URDF.
    
    Args:
        robot_urdf_path: Path to the robot URDF file
        tool_urdf: URDF XML string for the tool
        tool_mount_link: Name of the link to attach the tool to
        output_path: Optional output path (creates temp file if not provided)
        
    Returns:
        Path to the merged URDF file
    """
    # Read robot URDF
    with open(robot_urdf_path, 'r') as f:
        robot_urdf = f.read()
    
    # Insert tool URDF before </robot>
    merged = robot_urdf.replace("</robot>", f"\n{tool_urdf}\n</robot>")
    
    # Write to output
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix='.urdf')
        os.close(fd)
    
    with open(output_path, 'w') as f:
        f.write(merged)
    
    return output_path


def resolve_package_paths(urdf_content: str, package_map: Dict[str, str]) -> str:
    """
    Resolve package:// URIs in URDF content.
    
    Args:
        urdf_content: URDF XML content
        package_map: Dict mapping package names to absolute paths
        
    Returns:
        URDF content with resolved paths
    """
    resolved = urdf_content
    
    for package_name, package_path in package_map.items():
        pattern = f"package://{package_name}/"
        replacement = package_path.rstrip('/') + '/'
        resolved = resolved.replace(pattern, replacement)
    
    return resolved


def create_box_link(
    name: str,
    size: Tuple[float, float, float],
    origin: Tuple[float, float, float] = (0, 0, 0),
    rpy: Tuple[float, float, float] = (0, 0, 0),
    color: Tuple[float, float, float, float] = (0.5, 0.5, 0.5, 1.0),
    mass: float = 0.001
) -> str:
    """
    Create a URDF link element for a box.
    
    Args:
        name: Link name
        size: Box dimensions [x, y, z]
        origin: Position offset from parent
        rpy: Rotation offset from parent (roll, pitch, yaw)
        color: RGBA color
        mass: Mass in kg
        
    Returns:
        URDF XML string for the link
    """
    # Compute inertia for a box
    ix = mass * (size[1]**2 + size[2]**2) / 12
    iy = mass * (size[0]**2 + size[2]**2) / 12
    iz = mass * (size[0]**2 + size[1]**2) / 12
    
    return f"""
  <link name="{name}">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="{mass}"/>
      <inertia ixx="{ix}" ixy="0" ixz="0" iyy="{iy}" iyz="0" izz="{iz}"/>
    </inertial>
    <visual>
      <origin rpy="{rpy[0]} {rpy[1]} {rpy[2]}" xyz="{origin[0]} {origin[1]} {origin[2]}"/>
      <geometry>
        <box size="{size[0]} {size[1]} {size[2]}"/>
      </geometry>
      <material name="{name}_material">
        <color rgba="{color[0]} {color[1]} {color[2]} {color[3]}"/>
      </material>
    </visual>
    <collision>
      <origin rpy="{rpy[0]} {rpy[1]} {rpy[2]}" xyz="{origin[0]} {origin[1]} {origin[2]}"/>
      <geometry>
        <box size="{size[0]} {size[1]} {size[2]}"/>
      </geometry>
    </collision>
  </link>
"""


def create_fixed_joint(
    name: str,
    parent: str,
    child: str,
    origin: Tuple[float, float, float] = (0, 0, 0),
    rpy: Tuple[float, float, float] = (0, 0, 0)
) -> str:
    """
    Create a URDF fixed joint element.
    
    Args:
        name: Joint name
        parent: Parent link name
        child: Child link name
        origin: Position offset
        rpy: Rotation offset (roll, pitch, yaw)
        
    Returns:
        URDF XML string for the joint
    """
    return f"""
  <joint name="{name}" type="fixed">
    <origin rpy="{rpy[0]} {rpy[1]} {rpy[2]}" xyz="{origin[0]} {origin[1]} {origin[2]}"/>
    <parent link="{parent}"/>
    <child link="{child}"/>
  </joint>
"""


def create_simple_tool(
    parts: List[Dict],
    tool_mount_link: str = "tool_mount"
) -> str:
    """
    Create a simple tool from a list of part specifications.
    
    Args:
        parts: List of dicts with keys:
            - name: Part name
            - size: [x, y, z] dimensions
            - position: [x, y, z] position relative to tool mount
            - rotation: [roll, pitch, yaw] rotation (optional)
            - color: [r, g, b, a] color (optional)
            - parent: Parent link name (optional, defaults to tool_mount)
        tool_mount_link: Name of the tool mount link
        
    Returns:
        URDF XML string for the complete tool
    """
    urdf_parts = []
    
    for i, part in enumerate(parts):
        name = part.get('name', f'tool_part_{i}')
        size = tuple(part['size'])
        position = tuple(part.get('position', [0, 0, 0]))
        rotation = tuple(part.get('rotation', [0, 0, 0]))
        color = tuple(part.get('color', [0.3, 0.3, 0.8, 1.0]))
        parent = part.get('parent', tool_mount_link)
        mass = part.get('mass', 0.001)
        
        # Create link
        link_urdf = create_box_link(
            name=name,
            size=size,
            color=color,
            mass=mass
        )
        urdf_parts.append(link_urdf)
        
        # Create joint
        joint_urdf = create_fixed_joint(
            name=f"{name}_joint",
            parent=parent,
            child=name,
            origin=position,
            rpy=rotation
        )
        urdf_parts.append(joint_urdf)
    
    return '\n'.join(urdf_parts)


def extract_tool_from_response(response: str) -> List[str]:
    """
    Extract URDF tool definitions from a VLM response.
    
    Args:
        response: VLM response text that may contain URDF in code blocks
        
    Returns:
        List of extracted URDF strings
    """
    tools = []
    
    # Strategy 1: Find all ```xml or ```urdf code blocks and combine them
    xml_blocks = []
    patterns = [
        r'```xml\s*([\s\S]*?)\s*```',
        r'```urdf\s*([\s\S]*?)\s*```',
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, response, re.IGNORECASE)
        for match in matches:
            if '<link' in match or '<joint' in match:
                xml_blocks.append(match.strip())
    
    # If we found multiple blocks, they might be one tool split across blocks
    # or multiple tools. Try to combine them if they look like one tool.
    if xml_blocks:
        # Combine all blocks into one URDF (common VLM behavior)
        combined = '\n'.join(xml_blocks)
        tools.append(combined)
    
    # Strategy 2: If no code blocks, try to extract raw URDF from text
    if not tools:
        # Find all URDF-like content between <link and </joint>
        all_urdf = []
        
        # Extract all link definitions
        link_pattern = r'<link\s+name=["\'][^"\']+["\'][\s\S]*?</link>'
        links = re.findall(link_pattern, response)
        all_urdf.extend(links)
        
        # Extract all joint definitions  
        joint_pattern = r'<joint\s+name=["\'][^"\']+["\'][\s\S]*?</joint>'
        joints = re.findall(joint_pattern, response)
        all_urdf.extend(joints)
        
        if all_urdf:
            tools.append('\n'.join(all_urdf))
    
    # Strategy 3: Last resort - find anything between first <link and last </joint>
    if not tools and '<link' in response:
        start = response.find('<link')
        # Find the last </joint> or </link>
        end_joint = response.rfind('</joint>')
        end_link = response.rfind('</link>')
        end = max(end_joint + len('</joint>') if end_joint != -1 else 0,
                  end_link + len('</link>') if end_link != -1 else 0)
        if start != -1 and end > start:
            tools.append(response[start:end].strip())
    
    return tools


def extract_actions_from_response(response: str) -> List[List[List[float]]]:
    """
    Extract action waypoints from a VLM response.
    
    Args:
        response: VLM response text that may contain numpy array
        
    Returns:
        List of waypoint sequences (each is a list of waypoints)
    """
    import ast
    
    all_actions = []
    
    # Strategy 1: Find np.array([...]) patterns
    # Use greedy match and find the matching closing ])
    np_array_starts = [m.end() for m in re.finditer(r'np\.array\s*\(\s*\[', response)]
    
    for start_pos in np_array_starts:
        # Find the matching closing bracket by counting brackets
        bracket_count = 1
        pos = start_pos
        while pos < len(response) and bracket_count > 0:
            if response[pos] == '[':
                bracket_count += 1
            elif response[pos] == ']':
                bracket_count -= 1
            pos += 1
        
        if bracket_count == 0:
            # Extract content between the brackets
            content = response[start_pos:pos-1]
            waypoints = _parse_array_content(content)
            if waypoints:
                all_actions.append(waypoints)
    
    # Strategy 2: Find action = [...] or actions = [...] patterns
    if not all_actions:
        action_pattern = r'actions?\s*=\s*\[([\s\S]*?)\](?=\s*(?:\n|$|```))'
        matches = re.findall(action_pattern, response)
        for match in matches:
            waypoints = _parse_array_content(match)
            if waypoints:
                all_actions.append(waypoints)
    
    # Strategy 3: Find ```python code blocks and look for arrays inside
    if not all_actions:
        python_blocks = re.findall(r'```python\s*([\s\S]*?)\s*```', response, re.IGNORECASE)
        for block in python_blocks:
            # Try to find array in the block - multiple patterns
            patterns = [
                r'\[\s*\[([\s\S]*)\]\s*\]',  # [[...]]
                r'=\s*\[([\s\S]*)\]',  # = [...]
            ]
            for pattern in patterns:
                array_match = re.search(pattern, block)
                if array_match:
                    content = array_match.group(1)
                    waypoints = _parse_array_content(content)
                    if waypoints:
                        all_actions.append(waypoints)
                        break
    
    # Strategy 4: Find any nested list that looks like waypoints
    if not all_actions:
        # More flexible pattern for nested arrays
        bracket_pattern = r'\[\s*\[[\d\.\-\+eE\,\s]+\](?:[\s,]*\[[\d\.\-\+eE\,\s]+\])*\s*\]'
        matches = re.findall(bracket_pattern, response)
        for match in matches:
            try:
                waypoints = ast.literal_eval(match)
                if isinstance(waypoints, list) and len(waypoints) >= 1:
                    if isinstance(waypoints[0], list) and len(waypoints[0]) >= 6:
                        all_actions.append(waypoints)
            except:
                continue
    
    # Strategy 5: Find individual waypoint lines and combine them
    if not all_actions:
        # Pattern for lines like [0.5, 0.0, 0.3, 0.0, 0.0, 0.0, 0]
        waypoint_pattern = r'\[\s*([\d\.\-\+eE]+)\s*,\s*([\d\.\-\+eE]+)\s*,\s*([\d\.\-\+eE]+)\s*,\s*([\d\.\-\+eE]+)\s*,\s*([\d\.\-\+eE]+)\s*,\s*([\d\.\-\+eE]+)\s*(?:,\s*([\d\.\-\+eE]+))?\s*\]'
        matches = re.findall(waypoint_pattern, response)
        if len(matches) >= 2:  # Need at least 2 waypoints
            waypoints = []
            for m in matches:
                wp = [float(x) if x else 0.0 for x in m]
                if len(wp) == 6:
                    wp.append(0.0)  # Add gripper
                waypoints.append(wp)
            if waypoints:
                all_actions.append(waypoints)
        
    return all_actions


def _eval_numpy_expr(expr_str: str) -> str:
    """Evaluate numpy expressions like np.pi/6 and replace with numeric values."""
    import re
    import math
    
    # Replace np.pi with actual value
    result = expr_str.replace('np.pi', str(math.pi))
    result = result.replace('numpy.pi', str(math.pi))
    result = result.replace('math.pi', str(math.pi))
    
    # Find and evaluate simple expressions like 3.14159.../6
    # Pattern: number / number or number * number
    def eval_simple_expr(match):
        try:
            return str(eval(match.group(0)))
        except:
            return match.group(0)
    
    # Match floating point divisions/multiplications
    result = re.sub(r'[\d.]+\s*[/\*]\s*[\d.]+', eval_simple_expr, result)
    
    return result


def _parse_array_content(content: str) -> Optional[List[List[float]]]:
    """Parse array content string into a list of waypoints."""
    import ast
    
    try:
        # First evaluate numpy expressions like np.pi/6
        content = _eval_numpy_expr(content)
        
        # Clean up the array content
        array_str = content.strip()
        
        # Remove Python comments (# ...)
        lines = array_str.split('\n')
        cleaned_lines = []
        for line in lines:
            # Remove comments
            if '#' in line:
                line = line[:line.index('#')]
            line = line.strip()
            if line:
                cleaned_lines.append(line)
        
        array_str = ' '.join(cleaned_lines)
        
        # Clean up whitespace
        while '  ' in array_str:
            array_str = array_str.replace('  ', ' ')
        
        # Make sure it's wrapped in brackets
        array_str = array_str.strip()
        if not array_str.startswith('['):
            array_str = f'[{array_str}]'
        
        # Parse
        waypoints = ast.literal_eval(array_str)
        
        # Validate
        if isinstance(waypoints, list) and len(waypoints) > 0:
            if isinstance(waypoints[0], list) and len(waypoints[0]) >= 6:
                return waypoints
            # Maybe it's a flat list that needs reshaping
            elif isinstance(waypoints[0], (int, float)) and len(waypoints) >= 7:
                # Reshape flat list to Nx7
                n = len(waypoints) // 7
                reshaped = [waypoints[i*7:(i+1)*7] for i in range(n)]
                if all(len(w) == 7 for w in reshaped):
                    return reshaped
    except Exception:
        pass
    
    return None
