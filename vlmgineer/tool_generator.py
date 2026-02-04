"""
Tool generator for VLMgineer.

Combines VLM client with prompts to generate tool designs.
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass
from PIL import Image
import numpy as np

from .vlm_client import VLMClient, ToolActionDesign
from .config import VLMConfig
from .prompts import (
    INITIAL_MISSION_PROMPT,
    EVOLUTION_MISSION_PROMPT,
    PROCEDURE_PROMPT,
    TOOL_SPECIFICATION_PROMPT,
    FRAME_CLARIFICATION_PROMPT,
)


@dataclass
class GenerationContext:
    """Context for tool generation."""
    
    task_description: str
    environment_code: str
    scene_image: Optional[Image.Image] = None
    
    # Optional custom prompts
    custom_system_prompt: Optional[str] = None
    custom_tool_spec: Optional[str] = None


class ToolGenerator:
    """
    High-level tool generator that wraps VLM client.
    
    Provides convenient methods for generating tool designs
    with appropriate prompts and context.
    """
    
    def __init__(self, config: Optional[VLMConfig] = None):
        """
        Initialize the tool generator.
        
        Args:
            config: VLM configuration
        """
        self.config = config or VLMConfig()
        self.vlm_client = VLMClient(self.config)
    
    def generate_initial_population(
        self,
        context: GenerationContext,
        action_spec: str,
        n_agents: int = 3,
        n_tools_per_agent: int = 2,
        n_actions_per_tool: int = 2,
    ) -> List[ToolActionDesign]:
        """
        Generate an initial population of tool-action designs.
        
        Args:
            context: Generation context with task info
            action_spec: Action specification prompt
            n_agents: Number of parallel VLM agents
            n_tools_per_agent: Tools per agent
            n_actions_per_tool: Actions per tool
            
        Returns:
            List of generated designs
        """
        # Build system prompt
        system_prompt = context.custom_system_prompt or (
            INITIAL_MISSION_PROMPT + "\n" + PROCEDURE_PROMPT
        )
        
        # Build tool spec
        tool_spec = context.custom_tool_spec or TOOL_SPECIFICATION_PROMPT
        
        # Generate designs
        designs = self.vlm_client.sample_population(
            task_description=context.task_description,
            environment_code=context.environment_code,
            frame_description=FRAME_CLARIFICATION_PROMPT,
            system_prompt=system_prompt,
            tool_spec=tool_spec,
            action_spec=action_spec,
            scene_image=context.scene_image,
            n_agents=n_agents,
            n_tools=n_tools_per_agent,
            n_actions=n_actions_per_tool,
        )
        
        return designs
    
    def evolve_population(
        self,
        context: GenerationContext,
        action_spec: str,
        previous_designs: List[Dict[str, Any]],
        evolution_prompt: str,
        n_agents: int = 2,
        n_tools_per_agent: int = 2,
        n_actions_per_tool: int = 2,
    ) -> List[ToolActionDesign]:
        """
        Evolve a population of designs based on previous results.
        
        Args:
            context: Generation context with task info
            action_spec: Action specification prompt
            previous_designs: List of previous designs with rewards
            evolution_prompt: Evolution instruction prompt
            n_agents: Number of parallel VLM agents
            n_tools_per_agent: Tools per agent
            n_actions_per_tool: Actions per tool
            
        Returns:
            List of evolved designs
        """
        # Build system prompt for evolution
        system_prompt = EVOLUTION_MISSION_PROMPT + "\n" + PROCEDURE_PROMPT
        
        # Build tool spec
        tool_spec = context.custom_tool_spec or TOOL_SPECIFICATION_PROMPT
        
        # Generate evolved designs
        designs = self.vlm_client.sample_population(
            task_description=context.task_description,
            environment_code=context.environment_code,
            frame_description=FRAME_CLARIFICATION_PROMPT,
            system_prompt=system_prompt,
            tool_spec=tool_spec,
            action_spec=action_spec,
            scene_image=context.scene_image,
            n_agents=n_agents,
            n_tools=n_tools_per_agent,
            n_actions=n_actions_per_tool,
            previous_designs=previous_designs,
            evolution_prompt=evolution_prompt,
        )
        
        return designs


def create_example_tool() -> str:
    """
    Create an example tool URDF for testing.
    
    Returns:
        URDF XML string for a simple L-shaped pusher tool
    """
    return """
  <link name="pusher_vertical">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0 0 0.075"/>
      <geometry>
        <box size="0.02 0.02 0.15"/>
      </geometry>
      <material name="tool_blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0 0 0.075"/>
      <geometry>
        <box size="0.02 0.02 0.15"/>
      </geometry>
    </collision>
  </link>
  <joint name="pusher_vertical_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0"/>
    <parent link="tool_mount"/>
    <child link="pusher_vertical"/>
  </joint>
  <link name="pusher_horizontal">
    <inertial>
      <origin rpy="0 0 0" xyz="0 0 0"/>
      <mass value="0.001"/>
      <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/>
    </inertial>
    <visual>
      <origin rpy="0 0 0" xyz="0.05 0 0"/>
      <geometry>
        <box size="0.1 0.02 0.02"/>
      </geometry>
      <material name="tool_blue">
        <color rgba="0.2 0.2 0.8 1"/>
      </material>
    </visual>
    <collision>
      <origin rpy="0 0 0" xyz="0.05 0 0"/>
      <geometry>
        <box size="0.1 0.02 0.02"/>
      </geometry>
    </collision>
  </link>
  <joint name="pusher_horizontal_joint" type="fixed">
    <origin rpy="0 0 0" xyz="0 0 0.15"/>
    <parent link="pusher_vertical"/>
    <child link="pusher_horizontal"/>
  </joint>
"""


def create_example_actions() -> np.ndarray:
    """
    Create example action waypoints for testing.
    
    Returns:
        Nx7 numpy array of waypoints
    """
    return np.array([
        # Start above table
        [0.3, 0.0, 0.4, 0.0, 0.0, 0.0, 0],
        # Move forward
        [0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 0],
        # Lower towards table
        [0.5, 0.0, 0.2, 0.0, 0.5, 0.0, 0],
        # Push forward
        [0.6, 0.0, 0.15, 0.0, 0.5, 0.0, 0],
        # Pull back
        [0.4, 0.0, 0.15, 0.0, 0.5, 0.0, 0],
        # Lift up
        [0.4, 0.0, 0.4, 0.0, 0.0, 0.0, 0],
    ])
