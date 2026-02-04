"""
Action specification prompts (minimal version).
"""

ACTION_SPECIFICATION_PROMPT = """## Action Format
Nx7 numpy array: [x, y, z, roll, pitch, yaw, gripper]
- Position in meters, orientation in radians
- gripper: 0=open, 1=closed
- Account for tool length when setting Z height

Example:
```python
action = np.array([
    [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 0],
    [0.5, 0.0, 0.15, 0.0, 0.5, 0.0, 0],
    [0.4, 0.0, 0.15, 0.0, 0.5, 0.0, 0],
])
```

Workspace: radius ~0.8m, table Z≈0.04m, robot at origin."""

ACTION_DIVERSITY_PROMPT = ""  # Not needed

FRAME_CLARIFICATION_PROMPT = """Coordinates: +X=front, +Y=left, +Z=up. Robot at origin."""
