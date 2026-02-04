"""
VLM client for VLMgineer.

Interfaces with Google Gemini for tool and action generation.
Uses the new google.genai SDK with support for Gemini Robotics-ER model.
"""

import json
import base64
from io import BytesIO
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from PIL import Image
import numpy as np

# Use google.genai SDK (new SDK for Gemini)
try:
    from google import genai
    from google.genai import types
except ImportError:
    raise ImportError(
        "google-genai is required. Install with: uv add google-genai"
    )

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
    """Client for Google Gemini VLM (supports Robotics-ER model)."""
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """
        Initialize the Gemini client.
        
        Args:
            config: VLM configuration
        """
        self.config = config or VLMConfig()
        self.config.validate()
        
        # Initialize google.genai client
        self.client = genai.Client(api_key=self.config.api_key)
        self.model_name = self.config.model_name
    
    def _image_to_bytes(self, image: Image.Image) -> bytes:
        """Convert PIL Image to bytes."""
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()
    
    def _image_to_base64(self, image: Image.Image) -> str:
        """Convert PIL Image to base64 string."""
        return base64.b64encode(self._image_to_bytes(image)).decode()
    
    def generate_content(
        self,
        prompt: str,
        image: Optional[Image.Image] = None,
    ) -> str:
        """
        Generate content from the model.
        
        Args:
            prompt: Text prompt
            image: Optional image input
            
        Returns:
            Generated text response
        """
        contents = []
        
        if image is not None:
            image_bytes = self._image_to_bytes(image)
            contents.append(
                types.Part.from_bytes(data=image_bytes, mime_type='image/png')
            )
        
        contents.append(prompt)
        
        # Configure generation
        gen_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            max_output_tokens=self.config.max_output_tokens,
        )
        
        # For robotics model, use minimal thinking for speed
        if "robotics" in self.model_name.lower():
            gen_config = types.GenerateContentConfig(
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                thinking_config=types.ThinkingConfig(thinking_budget=256)
            )
        
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=contents,
            config=gen_config,
        )
        
        return response.text
    
    def generate_tool_designs(
        self,
        task_description: str,
        environment_code: str,
        scene_image: Optional[Image.Image] = None,
        n_tools: int = 3,
        n_actions: int = 3,
        previous_designs: Optional[List[Dict]] = None,
    ) -> List[ToolActionDesign]:
        """
        Generate tool and action designs for a task.
        
        Args:
            task_description: Description of the task
            environment_code: Code showing environment setup
            scene_image: Image of the scene
            n_tools: Number of tool designs to generate
            n_actions: Number of action sequences per tool
            previous_designs: Previous designs for evolution
            
        Returns:
            List of ToolActionDesign objects
        """
        prompt = self._build_prompt(
            task_description=task_description,
            environment_code=environment_code,
            n_tools=n_tools,
            n_actions=n_actions,
            previous_designs=previous_designs,
        )
        
        # Generate response
        response_text = self.generate_content(prompt, scene_image)
        
        # Parse response into designs
        designs = self._parse_response(response_text, n_tools, n_actions)
        
        return designs
    
    def _build_prompt(
        self,
        task_description: str,
        environment_code: str,
        n_tools: int,
        n_actions: int,
        previous_designs: Optional[List[Dict]] = None,
    ) -> str:
        """Build the full prompt for tool generation."""
        from .prompts.system import INITIAL_MISSION_PROMPT, EVOLUTION_MISSION_PROMPT, PROCEDURE_PROMPT
        from .prompts.tool_spec import TOOL_SPECIFICATION_PROMPT
        from .prompts.action_spec import ACTION_SPECIFICATION_PROMPT, FRAME_CLARIFICATION_PROMPT
        
        # Choose mission prompt based on whether we have previous designs
        if previous_designs:
            mission = EVOLUTION_MISSION_PROMPT
            evolution_context = self._format_previous_designs(previous_designs)
        else:
            mission = INITIAL_MISSION_PROMPT
            evolution_context = ""
        
        prompt = f"""{mission}

{PROCEDURE_PROMPT}

## Task Description
{task_description}

## Environment Code
```python
{environment_code}
```

{TOOL_SPECIFICATION_PROMPT}

{ACTION_SPECIFICATION_PROMPT}

{FRAME_CLARIFICATION_PROMPT}

{evolution_context}

## Your Task

Generate {n_tools} different tool designs. For each tool, generate {n_actions} different action sequences.

For each tool design:
1. First describe your strategy and the tool design
2. Output the tool URDF in a ```xml code block
3. For each action sequence, describe the approach then output the waypoints as a numpy array in a ```python code block

Be creative and diverse in your designs! Each tool should have a different approach to solving the task.
"""
        return prompt
    
    def _format_previous_designs(self, designs: List[Dict]) -> str:
        """Format previous designs for evolution prompt."""
        if not designs:
            return ""
        
        text = "\n## Previous Designs (for evolution)\n\n"
        text += "Learn from these designs. Mutate or combine them to create better solutions.\n\n"
        
        for i, design in enumerate(designs):
            reward = design.get("reward", 0)
            desc = design.get("description", "")[:200]
            text += f"### Design {i+1} (reward: {reward:.3f})\n"
            text += f"{desc}\n\n"
        
        return text
    
    def _parse_response(
        self,
        response: str,
        n_tools: int,
        n_actions: int
    ) -> List[ToolActionDesign]:
        """Parse the VLM response into tool-action designs."""
        designs = []
        
        # Extract all tools from response
        tools = extract_tool_from_response(response)
        
        # Extract all action sequences
        all_actions = extract_actions_from_response(response)
        
        # Match tools with actions
        for i, tool_urdf in enumerate(tools):
            if not validate_tool_urdf(tool_urdf):
                continue
            
            # Get corresponding actions
            start_idx = i * n_actions
            end_idx = min(start_idx + n_actions, len(all_actions))
            tool_actions = all_actions[start_idx:end_idx] if start_idx < len(all_actions) else []
            
            # Create design for each action sequence
            for actions in tool_actions:
                if actions is not None and len(actions) > 0:
                    # Convert to numpy array
                    actions_array = np.array(actions)
                    
                    designs.append(ToolActionDesign(
                        tool_urdf=tool_urdf,
                        actions=actions_array,
                        raw_response=response[:500],
                    ))
        
        # If no proper matches, try to create designs from whatever we found
        if not designs:
            for tool_urdf in tools:
                if validate_tool_urdf(tool_urdf):
                    for actions in all_actions:
                        if actions is not None and len(actions) > 0:
                            # Convert to numpy array
                            actions_array = np.array(actions)
                            
                            designs.append(ToolActionDesign(
                                tool_urdf=tool_urdf,
                                actions=actions_array,
                                raw_response=response[:500],
                            ))
        
        return designs


class VLMClient:
    """
    Main VLM client interface for VLMgineer.
    """
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """Initialize the VLM client."""
        self.config = config or VLMConfig()
        self.client = GeminiClient(config=self.config)
    
    def sample_population(
        self,
        task_description: str,
        environment_code: str,
        frame_description: str,
        system_prompt: str,
        tool_spec: str,
        action_spec: str,
        scene_image: Optional[Image.Image] = None,
        n_agents: int = 3,
        n_tools: int = 3,
        n_actions: int = 3,
        previous_designs: Optional[List[Dict]] = None,
        evolution_prompt: Optional[str] = None,
    ) -> List[ToolActionDesign]:
        """
        Sample a population of tool-action designs.
        
        This is the main interface called by the evolution engine.
        
        Args:
            task_description: Description of the manipulation task
            environment_code: Python code showing environment setup
            frame_description: Description of coordinate frames
            system_prompt: System prompt for the VLM
            tool_spec: Tool specification prompt
            action_spec: Action specification prompt
            scene_image: PIL Image of the current scene
            n_agents: Number of parallel agents
            n_tools: Number of tool designs per agent
            n_actions: Number of action sequences per tool
            previous_designs: Previous designs for evolutionary refinement
            evolution_prompt: Additional prompt for evolution
            
        Returns:
            List of ToolActionDesign objects
        """
        all_designs = []
        
        # Each agent generates designs in parallel (conceptually)
        for agent_idx in range(n_agents):
            # Build the full prompt
            prompt = self._build_full_prompt(
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
                agent_idx=agent_idx,
            )
            
            try:
                # Generate response
                response_text = self.client.generate_content(prompt, scene_image)
                
                # Parse response into designs
                designs = self._parse_response(response_text, n_tools, n_actions)
                all_designs.extend(designs)
                
            except Exception as e:
                print(f"  Agent {agent_idx + 1} failed: {e}")
                continue
        
        return all_designs
    
    def _build_full_prompt(
        self,
        task_description: str,
        environment_code: str,
        frame_description: str,
        system_prompt: str,
        tool_spec: str,
        action_spec: str,
        n_tools: int,
        n_actions: int,
        previous_designs: Optional[List[Dict]],
        evolution_prompt: Optional[str],
        agent_idx: int,
    ) -> str:
        """Build the full prompt following the paper's format (Appendix D)."""
        from .prompts import ACTION_DIVERSITY_PROMPT, FRAME_CLARIFICATION_PROMPT, PROCEDURE_PROMPT
        
        # Format procedure prompt with n_tools and n_actions
        procedure = PROCEDURE_PROMPT.format(n_tools=n_tools, n_actions=n_actions)
        
        # Evolution context with full design details
        evolution_context = ""
        if previous_designs and evolution_prompt:
            evolution_context = "\n## Previous Elite Designs\n\n"
            evolution_context += "Learn from these high-performing designs:\n\n"
            for i, d in enumerate(previous_designs[:5]):  # Top 5
                reward = d.get('reward', 0)
                urdf = d.get('tool_urdf', '')[:500]  # First 500 chars of URDF
                evolution_context += f"### Elite Design {i+1} (reward: {reward:.3f})\n"
                evolution_context += f"```xml\n{urdf}\n```\n\n"
            
            evolution_context += f"\n{evolution_prompt}\n"
        
        # Build full prompt following paper structure
        prompt = f"""{system_prompt}

{procedure}

## Task Description
{task_description}

## Environment Code
```python
{environment_code}
```

{tool_spec}

{action_spec}

{ACTION_DIVERSITY_PROMPT}

{FRAME_CLARIFICATION_PROMPT}

{evolution_context}

## Output Format

For EACH of the {n_tools} tools you design:

1. Briefly describe your tool strategy (1-2 sentences)

2. Output the tool URDF in an ```xml code block:
```xml
<link name="...">...</link>
<joint name="..." type="fixed">...</joint>
```

3. For EACH of the {n_actions} action sequences for this tool, output waypoints as a numpy array in a ```python code block:
```python
action = np.array([
    [x, y, z, roll, pitch, yaw, gripper],  # waypoint 1
    [x, y, z, roll, pitch, yaw, gripper],  # waypoint 2
    ...
])
```

Be creative and diverse! Each tool should take a different approach to solving the task."""
        return prompt
    
    def _parse_response(
        self,
        response: str,
        n_tools: int,
        n_actions: int
    ) -> List[ToolActionDesign]:
        """Parse the VLM response into tool-action designs."""
        designs = []
        
        # Extract all tools from response
        tools = extract_tool_from_response(response)
        
        # Extract all action sequences
        all_actions = extract_actions_from_response(response)
        
        # Match tools with actions
        for i, tool_urdf in enumerate(tools):
            if not validate_tool_urdf(tool_urdf):
                continue
            
            # Get corresponding actions
            start_idx = i * n_actions
            end_idx = min(start_idx + n_actions, len(all_actions))
            tool_actions = all_actions[start_idx:end_idx] if start_idx < len(all_actions) else []
            
            # Create design for each action sequence
            for actions in tool_actions:
                if actions is not None and len(actions) > 0:
                    # Convert to numpy array
                    actions_array = np.array(actions)
                    
                    designs.append(ToolActionDesign(
                        tool_urdf=tool_urdf,
                        actions=actions_array,
                        raw_response=response[:500],
                    ))
        
        # If no proper matches, try to create designs from whatever we found
        if not designs:
            for tool_urdf in tools:
                if validate_tool_urdf(tool_urdf):
                    for actions in all_actions:
                        if actions is not None and len(actions) > 0:
                            # Convert to numpy array
                            actions_array = np.array(actions)
                            
                            designs.append(ToolActionDesign(
                                tool_urdf=tool_urdf,
                                actions=actions_array,
                                raw_response=response[:500],
                            ))
        
        return designs
    
    def generate_designs(
        self,
        task_description: str,
        environment_code: str,
        scene_image: Optional[Image.Image] = None,
        n_tools: int = 3,
        n_actions: int = 3,
        previous_designs: Optional[List[Dict]] = None,
    ) -> List[ToolActionDesign]:
        """
        Generate tool and action designs (legacy interface).
        
        Args:
            task_description: Description of the manipulation task
            environment_code: Python code showing environment setup
            scene_image: PIL Image of the current scene
            n_tools: Number of tool designs to generate
            n_actions: Number of action sequences per tool
            previous_designs: Previous designs for evolutionary refinement
            
        Returns:
            List of ToolActionDesign objects
        """
        return self.client.generate_tool_designs(
            task_description=task_description,
            environment_code=environment_code,
            scene_image=scene_image,
            n_tools=n_tools,
            n_actions=n_actions,
            previous_designs=previous_designs,
        )
