"""
Internal configuration classes for VLMgineer.
Settings are loaded from the root config.py file.
"""

import os
import sys
from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path


def _load_api_key() -> Optional[str]:
    """Load API key from environment or .env file."""
    key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("API_KEY")
    if key:
        return key
    
    # Try .env file in project root
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    
    if env_file.exists():
        try:
            with open(env_file) as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#') and '=' in line:
                        k, v = line.split('=', 1)
                        k, v = k.strip(), v.strip().strip('"').strip("'")
                        if k in ("API_KEY", "GOOGLE_API_KEY"):
                            return v
        except Exception:
            pass
    return None


def _load_user_config():
    """Load settings from root config.py."""
    try:
        # Add project root to path
        project_root = Path(__file__).parent.parent
        if str(project_root) not in sys.path:
            sys.path.insert(0, str(project_root))
        
        import config as user_config
        return user_config
    except ImportError:
        return None


@dataclass
class VLMConfig:
    """VLM client configuration."""
    api_key: Optional[str] = field(default_factory=_load_api_key)
    model_name: str = "gemini-robotics-er-1.5-preview"  # Specialized for robotics
    temperature: float = 0.5  # Lower for robotics tasks
    max_output_tokens: int = 4096  # Reduced for faster generation
    
    def validate(self) -> None:
        if not self.api_key:
            raise ValueError("No API key found. Set API_KEY in .env file.")


@dataclass
class EvolutionConfig:
    """Evolutionary search configuration."""
    n_agents: int = 3
    n_tools: int = 3
    n_actions: int = 3
    top_k: int = 3
    reward_threshold: float = 0.3
    n_iterations: int = 3
    
    @property
    def total_samples_per_iteration(self) -> int:
        return self.n_agents * self.n_tools * self.n_actions


@dataclass  
class SimulationConfig:
    """PyBullet simulation configuration."""
    time_step: float = 1 / 240
    max_steps: int = 1000
    n_parallel_sims: int = 4
    gui: bool = False
    robot_urdf: str = "robot_descriptions/franka_panda/panda_with_tool_mount.urdf"
    tool_mount_link: str = "tool_mount"
    workspace_center: tuple = (0.5, 0.0, 0.3)
    workspace_radius: float = 0.7


@dataclass
class VLMgineerConfig:
    """Main configuration combining all sub-configs."""
    vlm: VLMConfig = field(default_factory=VLMConfig)
    evolution: EvolutionConfig = field(default_factory=EvolutionConfig)
    simulation: SimulationConfig = field(default_factory=SimulationConfig)
    output_dir: str = "vlmgineer_results"
    verbose: bool = True
    save_all_designs: bool = True
    
    def validate(self) -> None:
        self.vlm.validate()


def get_config() -> VLMgineerConfig:
    """Get configuration from root config.py file."""
    user = _load_user_config()
    
    if user is None:
        return VLMgineerConfig()
    
    vlm = VLMConfig(
        api_key=_load_api_key(),
        model_name=getattr(user, 'MODEL', 'gemini-3-pro-preview'),
        temperature=getattr(user, 'TEMPERATURE', 0.8),
        max_output_tokens=getattr(user, 'MAX_TOKENS', 16384),
    )
    
    evolution = EvolutionConfig(
        n_agents=getattr(user, 'AGENTS', 3),
        n_tools=getattr(user, 'TOOLS_PER_AGENT', 3),
        n_actions=getattr(user, 'ACTIONS_PER_TOOL', 3),
        n_iterations=getattr(user, 'ITERATIONS', 3),
        top_k=getattr(user, 'TOP_K', 5),
        reward_threshold=getattr(user, 'REWARD_THRESHOLD', 0.3),
    )
    
    simulation = SimulationConfig(
        gui=getattr(user, 'SHOW_GUI', False),
    )
    
    return VLMgineerConfig(
        vlm=vlm,
        evolution=evolution,
        simulation=simulation,
        output_dir=getattr(user, 'OUTPUT_DIR', 'vlmgineer_results'),
        verbose=getattr(user, 'VERBOSE', True),
        save_all_designs=getattr(user, 'SAVE_ALL_DESIGNS', True),
    )


# Default config loaded from root config.py
default_config = get_config()
