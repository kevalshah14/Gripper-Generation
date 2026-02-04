# Tool Visualizer - Simple Tool Generation

Generate a custom robot gripper/tool for any task and see it on the robot in 3D simulation.

## Quick Start

```bash
# Generate a tool to push a cube
uv run visualize_tool.py --task "push a cube closer" --object "red cube at position (0.7, 0, 0.06)"

# Generate a tool to scoop objects
uv run visualize_tool.py --task "scoop small balls" --object "purple spheres in a container"

# Generate a tool to lift something
uv run visualize_tool.py --task "lift a heavy box" --object "cardboard box (0.2m × 0.2m × 0.3m)"

# Generate a hook tool
uv run visualize_tool.py --task "pull an object closer" --object "cube behind an obstacle"
```

## What It Does

1. **Sends your task to AI**: Describes what you want to accomplish
2. **AI designs a tool**: Creates a custom gripper/tool geometry in URDF format
3. **AI plans actions**: Generates waypoints for how the robot should move
4. **Shows it in 3D**: Opens PyBullet simulation with the tool attached to the robot

## Controls (in simulation)

- **SPACE**: Start/pause the action execution
- **R**: Reset to initial pose
- **Q**: Quit

## Parameters

```bash
--task "description"    # What should the robot do? (required)
--object "description"  # What object(s) are involved? (required)
--model "model-name"    # Which AI model to use (default: gemini-2.5-flash)
```

## Examples with Different Tasks

### Push/Drag Tasks
```bash
uv run visualize_tool.py \
  --task "push a cube to the left side of the table" \
  --object "red cube at (0.7, 0, 0.06), target at (0.3, 0, 0.06)"
```

### Scoop/Gather Tasks
```bash
uv run visualize_tool.py \
  --task "scoop multiple small balls into a container" \
  --object "5 small spheres scattered on table"
```

### Lift/Elevate Tasks
```bash
uv run visualize_tool.py \
  --task "lift a plate off a table" \
  --object "flat plate (20cm diameter) at table height"
```

### Pull/Hook Tasks
```bash
uv run visualize_tool.py \
  --task "hook and pull an object from behind an obstacle" \
  --object "cube behind a wall at (0.8, 0, 0.06)"
```

## How It Works

```
┌─────────────────────────────────────────┐
│ 1. You describe the task and object     │
│                                         │
│ 2. AI (Gemini) generates:              │
│    • Tool geometry (URDF)               │
│    • Motion plan (waypoints)            │
│                                         │
│ 3. PyBullet loads:                      │
│    • Franka Panda robot                 │
│    • Your custom tool                   │
│    • Table environment                  │
│                                         │
│ 4. You see the tool in action!         │
└─────────────────────────────────────────┘
```

## Coordinate System

- **Origin (0, 0, 0)**: Robot base
- **+X**: Forward (away from robot)
- **+Y**: Left
- **+Z**: Up
- **Table**: At Z = 0.04m
- **Workspace**: ~0.8m radius

## Tips for Good Results

1. **Be specific** about object location:
   - Good: "red cube at (0.7, 0, 0.06)"
   - Bad: "a cube somewhere"

2. **Describe the goal clearly**:
   - Good: "push the cube from X=0.7 to X=0.3"
   - Bad: "move the cube"

3. **Mention object properties** if relevant:
   - Size: "small sphere (5cm diameter)"
   - Weight: "heavy box"
   - Shape: "flat plate", "cylindrical can"

4. **Specify constraints**:
   - "without touching the walls"
   - "from above"
   - "using a hook motion"

## Troubleshooting

**"Could not parse tool or actions"**
- The AI response wasn't formatted correctly
- Try rephrasing your task description
- Try a different model (--model gemini-2.5-pro)

**Tool looks weird**
- AI might have misunderstood the task
- Be more specific about dimensions and locations
- Try adding more context about the object

**Robot doesn't move**
- Check if waypoints are within workspace (~0.8m radius)
- Press SPACE to start execution
- Check terminal for errors

**Rate limit error**
- You've hit API quota
- Wait for reset or use a different model
- Consider upgrading to paid tier
