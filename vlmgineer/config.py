"""
Configuration for VLMgineer.
"""

import os
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


def _load_api_key() -> Optional[str]:
    """Load API key from environment or .env file."""
    # First check environment variable
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("API_KEY")
    if key:
        return key
    
    # Try to load from .env file in project root
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        if '=' in line:
                            k, v = line.split('=', 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k in ("API_KEY", "GOOGLE_API_KEY"):
                                return v
        except Exception:
            pass
    
    return None


@dataclass
class VLMConfig:
    """Configuration for the VLM client."""
    
    # API configuration
    api_key: Optional[str] = field(default_factory=_load_api_key)
    model_name: str = "gemini-3-flash-preview"
    
    # Generation parameters
    temperature: float = 0.8
    max_output_tokens: int = 16384
    
    def validate(self) -> None:
        """Validate the configuration."""
        if not self.api_key:
            raise ValueError(
                "GOOGLE_API_KEY environment variable not set. "
                "Please set it with your Gemini API key."
            )


@dataclass
class EvolutionConfig:
    """Configuration for the evolutionary search."""
    
    # Population parameters
    n_agents: int = 5  # Number of parallel VLM agents
    n_tools: int = 3   # Tools per agent
    n_actions: int = 3  # Actions per tool
    
    # Selection parameters
    top_k: int = 3  # Number of elite designs to keep
    reward_threshold: float = 0.3  # Minimum reward to be considered
    
    # Iteration parameters
    n_iterations: int = 3  # Number of evolution cycles
    
    @property
    def total_samples_per_iteration(self) -> int:
        """Total tool-action pairs per iteration."""
        return self.n_agents * self.n_tools * self.n_actions


@dataclass  
class SimulationConfig:
    """Configuration for the PyBullet simulation."""
    
    # Simulation parameters
    time_step: float = 1 / 240
    max_steps: int = 1000
    
    # Parallel evaluation
    n_parallel_sims: int = 4
    
    # GUI settings
    gui: bool = False
    
    # Robot configuration (Franka Panda)
    robot_urdf: str = "robot_descriptions/franka_panda/panda_with_tool_mount.urdf"
    tool_mount_link: str = "tool_mount"
    
    # Workspace bounds (Franka Panda has ~855mm reach)
    workspace_center: tuple = (0.5, 0.0, 0.3)
    workspace_radius: float = 0.7


@dataclass
class VLMgineerConfig:
    """Main configuration combining all sub-configs."""
    
    vlm: VLMConfig = field(default_factory=VLMConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    
    # Output directory for results
    output_dir: str = "vlmgineer_results"
    
    # Logging
    verbose: bool = True
    save_all_designs: bool = True
    
    def validate(self) -> None:
        """Validate all configurations."""
        self.vlm.validate()


# Default configuration instance
default_config = VLMgineerConfig()
