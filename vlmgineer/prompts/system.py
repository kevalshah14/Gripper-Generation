"""
System prompts for VLMgineer (minimal version for speed).
"""

INITIAL_MISSION_PROMPT = """You design robotic tools and actions for a Franka Panda arm.
Generate diverse tool-action pairs. Output URDF and waypoints directly - minimal explanation."""

EVOLUTION_MISSION_PROMPT = """Evolve the previous designs by mutation (change one aspect) or crossover (combine two designs).
Learn from rewards. Output improved URDF and waypoints directly."""

PROCEDURE_PROMPT = """Output format for EACH design:
1. Tool URDF in ```xml block
2. Action waypoints as np.array in ```python block

Be concise. Generate code directly."""
