"""
VLM client for VLMgineer.

Interfaces with Google Gemini for tool and action generation.
"""

import json
import base64
import asyncio
from io import BytesIO
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

try:
    import google.generativeai as genai
except ImportError:
    genai = None

from .config import VLMConfig
from .urdf_utils import extract_tool_from_response, extract_actions_from_response, validate_tool_urdf


@dataclass
class ToolActionDesign:
    """A generated tool and action design."""
    
    tool_urdf: str
    actions: np.ndarray  # Nx7 array
    tool_description: str = ""
    action_description: str = ""
    raw_response: str = ""
    
    @property
    def is_valid(self) -> bool:
        """Check if the design is valid."""
        return (
            len(self.tool_urdf) > 0 and 
            self.actions is not None and 
            len(self.actions) > 0
        )


class GeminiClient:
    """Client for Google Gemini VLM."""
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """
        Initialize the Gemini client.
        
        Args:
            config: VLM configuration
        """
        if genai is None:
            raise ImportError(
                "google-generativeai is required. "
                "Install with: pip install google-generativeai"
            )
        
        self.config = config or VLMConfig()
        self.config.validate()
        
        # Configure API
        genai.configure(api_key=self.config.api_key)
        
        # Initialize model
        self.model = genai.GenerativeModel(
            model_name=self.config.model_name,
            generation_config={
                "temperature": self.config.temperature,
                "max_output_tokens": self.config.max_output_tokens,
            }
        )
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return base64.b64encode(buffer.getvalue()).decode()
    
    def _build_prompt(
        self,
        task_description: str,
        environment_code: str,
        frame_description: str,
        system_prompt: str,
        tool_spec: str,
        action_spec: str,
        n_tools: int = 1,
        n_actions: int = 1,
        previous_designs: Optional[List[Dict]] = None,
        evolution_prompt: Optional[str] = None,
    ) -> str:
        """
        Build the full prompt for the VLM.
        
        Args:
            task_description: Description of the task
            environment_code: Python code showing environment setup
            frame_description: Coordinate frame description
            system_prompt: System/mission introduction
            tool_spec: Tool specification prompt
            action_spec: Action specification prompt
            n_tools: Number of tools to generate
            n_actions: Number of actions per tool
            previous_designs: Previous designs for evolution
            evolution_prompt: Evolution instructions
            
        Returns:
            Complete prompt string
        """
        prompt_parts = [system_prompt]
        
        # Add task description
        prompt_parts.append(f"\n## Task Description\n{task_description}")
        
        # Add environment code
        prompt_parts.append(f"\n## Environment Code\n```python\n{environment_code}\n```")
        
        # Add frame description
        prompt_parts.append(f"\n## Coordinate Frame\n{frame_description}")
        
        # Add tool specification
        prompt_parts.append(f"\n## Tool Specification\n{tool_spec}")
        
        # Add action specification
        prompt_parts.append(f"\n## Action Specification\n{action_spec}")
        
        # Add generation instructions
        prompt_parts.append(f"""
## Generation Instructions

Generate {n_tools} different tool design(s). For each tool, generate {n_actions} action sequence(s).

For each tool-action pair, output:
1. Tool description (brief text)
2. Tool URDF (in ```xml``` code block)
3. Action description (brief text)  
4. Action waypoints as numpy array (in ```python``` code block)

Use this exact format for each design:

### Design [N]

**Tool Description:** [describe the tool]

**Tool URDF:**
```xml
<link name="tool_part_1">
  ...
</link>
<joint name="tool_joint_1" type="fixed">
  ...
</joint>
```

**Action Description:** [describe how to use the tool]

**Action Waypoints:**
```python
action = np.array([
    [x, y, z, roll, pitch, yaw, gripper],
    [x, y, z, roll, pitch, yaw, gripper],
    ...
])
```
""")
        
        # Add evolution context if provided
        if previous_designs and evolution_prompt:
            prompt_parts.append(f"\n## Previous Designs\n")
            for i, design in enumerate(previous_designs):
                prompt_parts.append(f"""
### Previous Design {i+1} (reward: {design.get('reward', 'N/A')})
Tool: {design.get('tool_description', 'N/A')}
```xml
{design.get('tool_urdf', '')}
```
""")
            prompt_parts.append(f"\n## Evolution Instructions\n{evolution_prompt}")
        
        return "\n".join(prompt_parts)
    
    def generate_designs(
        self,
        task_description: str,
        environment_code: str,
        frame_description: str,
        system_prompt: str,
        tool_spec: str,
        action_spec: str,
        scene_image: Optional[Image.Image] = None,
        n_tools: int = 1,
        n_actions: int = 1,
        previous_designs: Optional[List[Dict]] = None,
        evolution_prompt: Optional[str] = None,
    ) -> List[ToolActionDesign]:
        """
        Generate tool and action designs.
        
        Args:
            task_description: Description of the task
            environment_code: Python code showing environment setup
            frame_description: Coordinate frame description
            system_prompt: System/mission introduction
            tool_spec: Tool specification prompt
            action_spec: Action specification prompt
            scene_image: Optional scene image
            n_tools: Number of tools to generate
            n_actions: Number of actions per tool
            previous_designs: Previous designs for evolution
            evolution_prompt: Evolution instructions
            
        Returns:
            List of ToolActionDesign objects
        """
        # Build prompt
        prompt = self._build_prompt(
            task_description=task_description,
            environment_code=environment_code,
            frame_description=frame_description,
            system_prompt=system_prompt,
            tool_spec=tool_spec,
            action_spec=action_spec,
            n_tools=n_tools,
            n_actions=n_actions,
            previous_designs=previous_designs,
            evolution_prompt=evolution_prompt,
        )
        
        # Prepare content
        content = []
        if scene_image is not None:
            content.append(scene_image)
        content.append(prompt)
        
        # Generate response
        response = self.model.generate_content(content)
        response_text = response.text
        
        # Debug output (only in verbose mode)
        # print(f"VLM Response: {len(response_text)} chars")
        
        # Parse response into designs
        designs = self._parse_response(response_text, n_tools, n_actions)
        
        return designs
    
    def _parse_response(
        self,
        response: str,
        expected_tools: int,
        expected_actions: int
    ) -> List[ToolActionDesign]:
        """
        Parse VLM response into tool-action designs.
        
        Args:
            response: Raw VLM response text
            expected_tools: Expected number of tools
            expected_actions: Expected actions per tool
            
        Returns:
            List of parsed designs
        """
        designs = []
        
        # Split by design markers
        design_sections = self._split_into_sections(response)
        
        for section in design_sections:
            # Extract tool URDF
            tool_urdf = extract_tool_from_response(section)
            if tool_urdf is None:
                continue
            
            # Validate tool
            validation = validate_tool_urdf(tool_urdf)
            if not validation.valid:
                continue
            
            # Extract actions
            actions_list = extract_actions_from_response(section)
            if actions_list is None:
                continue
            
            actions = np.array(actions_list)
            
            # Ensure actions have 7 columns (add gripper=0 if needed)
            if actions.shape[1] == 6:
                gripper_col = np.zeros((actions.shape[0], 1))
                actions = np.hstack([actions, gripper_col])
            
            # Extract descriptions
            tool_desc = self._extract_description(section, "Tool Description")
            action_desc = self._extract_description(section, "Action Description")
            
            designs.append(ToolActionDesign(
                tool_urdf=tool_urdf,
                actions=actions,
                tool_description=tool_desc,
                action_description=action_desc,
                raw_response=section
            ))
        
        return designs
    
    def _split_into_sections(self, response: str) -> List[str]:
        """Split response into design sections."""
        import re
        
        # Try splitting by "Design" headers
        pattern = r'###?\s*Design\s*\[?\d+\]?'
        parts = re.split(pattern, response, flags=re.IGNORECASE)
        
        # Filter empty parts
        sections = [p.strip() for p in parts if p.strip()]
        
        # If no sections found, treat entire response as one section
        if not sections:
            sections = [response]
        
        return sections
    
    def _extract_description(self, text: str, marker: str) -> str:
        """Extract a description following a marker."""
        import re
        
        pattern = rf'\*?\*?{marker}:?\*?\*?\s*(.+?)(?:\n\n|\*\*|```|$)'
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        
        if match:
            return match.group(1).strip()
        return ""


class VLMClient:
    """
    High-level VLM client that manages multiple agents.
    
    Supports parallel queries for population generation.
    """
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """
        Initialize the VLM client.
        
        Args:
            config: VLM configuration
        """
        self.config = config or VLMConfig()
        self.gemini = GeminiClient(self.config)
    
    def sample_population(
        self,
        task_description: str,
        environment_code: str,
        frame_description: str,
        system_prompt: str,
        tool_spec: str,
        action_spec: str,
        scene_image: Optional[Image.Image] = None,
        n_agents: int = 1,
        n_tools: int = 1,
        n_actions: int = 1,
        previous_designs: Optional[List[Dict]] = None,
        evolution_prompt: Optional[str] = None,
    ) -> List[ToolActionDesign]:
        """
        Sample a population of designs from multiple agents.
        
        Args:
            task_description: Description of the task
            environment_code: Python code showing environment setup
            frame_description: Coordinate frame description
            system_prompt: System/mission introduction
            tool_spec: Tool specification prompt
            action_spec: Action specification prompt
            scene_image: Optional scene image
            n_agents: Number of parallel agents
            n_tools: Tools per agent
            n_actions: Actions per tool
            previous_designs: Previous designs for evolution
            evolution_prompt: Evolution instructions
            
        Returns:
            List of all generated designs
        """
        all_designs = []
        
        # Query each agent (could be parallelized with asyncio)
        for i in range(n_agents):
            try:
                designs = self.gemini.generate_designs(
                    task_description=task_description,
                    environment_code=environment_code,
                    frame_description=frame_description,
                    system_prompt=system_prompt,
                    tool_spec=tool_spec,
                    action_spec=action_spec,
                    scene_image=scene_image,
                    n_tools=n_tools,
                    n_actions=n_actions,
                    previous_designs=previous_designs,
                    evolution_prompt=evolution_prompt,
                )
                all_designs.extend(designs)
            except Exception as e:
                print(f"Agent {i} failed: {e}")
                continue
        
        return all_designs
