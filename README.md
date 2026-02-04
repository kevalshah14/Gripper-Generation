# VLMgineer

**Vision Language Models as Robotic Toolsmiths**

VLMgineer uses Google's Gemini Robotics-ER model to automatically design custom gripper tools and action plans for robotic manipulation tasks.

## How It Works

1. **Task Description** - You specify a manipulation task (e.g., "bring cube to target")
2. **VLM Generation** - Gemini Robotics-ER generates custom tool URDFs and action waypoints
3. **Simulation** - Designs are evaluated in PyBullet physics simulation
4. **Evolution** - Best designs are refined through evolutionary search

## Requirements

- Python 3.9+
- [uv](https://docs.astral.sh/uv/) package manager
- Google AI API key (for Gemini)

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd Gripper-Generation

# Install dependencies with uv
uv sync
```

## Configuration

### 1. Set your API key

Create a `.env` file in the project root:

```bash
API_KEY=your_google_ai_api_key_here
```

Get your API key from [Google AI Studio](https://aistudio.google.com/apikey).

### 2. Edit config.py (optional)

All settings are in `config.py`:

```python
# Task to solve
TASK = "bring_cube"        # Options: bring_cube, drill_press, lift_box, clean_table

# Evolution parameters
ITERATIONS = 3             # Number of evolution iterations
AGENTS = 3                 # Parallel design agents
TOOLS_PER_AGENT = 3        # Tool designs per agent
ACTIONS_PER_TOOL = 3       # Action sequences per tool

# Model settings
MODEL = "gemini-robotics-er-1.5-preview"  # Robotics-specialized model
TEMPERATURE = 0.5          # Generation temperature

# Simulation
SHOW_GUI = False           # Show PyBullet visualization
VERBOSE = True             # Print detailed output
```

## Usage

### Basic Run

```bash
uv run main.py
```

This runs evolutionary search using settings from `config.py`.

### Command Line Options

Override any config setting from the command line:

```bash
# Different task
uv run main.py --task drill_press

# More iterations
uv run main.py --iterations 5

# Show visualization
uv run main.py --gui

# Minimal test run
uv run main.py --iterations 1 --agents 1 --tools 1 --actions 1
```

### Demo Mode (No API Key Needed)

Test the simulation with a pre-built example tool:

```bash
uv run main.py --demo
uv run main.py --demo --gui  # With visualization
```

### Evaluate a Saved Result

```bash
uv run main.py --evaluate vlmgineer_results/best_design.json --gui
```

## Available Tasks

| Task | Description |
|------|-------------|
| `bring_cube` | Move a cube to a target location |
| `drill_press` | Operate a drill press spindle handle |
| `lift_box` | Lift a box off the table |
| `clean_table` | Push objects off a table |

## Output

Results are saved to `vlmgineer_results/`:

- `best_design.json` - Best tool URDF and action waypoints
- `evolution_log.json` - Full evolution history

## Project Structure

```
Gripper-Generation/
├── config.py              # ← Edit this to configure
├── main.py                # ← Run this
├── .env                   # ← Your API key
├── vlmgineer/             # Internal code
│   ├── vlm_client.py      # Gemini API client
│   ├── evolution.py       # Evolutionary search
│   ├── envs/              # Task environments
│   └── prompts/           # VLM prompts
├── robot_descriptions/    # Robot URDFs
└── vlmgineer_results/     # Output directory
```

## Troubleshooting

### API Quota Exceeded

If you see `429 Quota Exceeded`, either:
- Wait and retry later
- Use `--demo` mode to test without API
- Switch to a different model in `config.py`

### PyBullet Issues on M1 Mac

The project uses `pybullet-mm` which has better M1 support. If you have issues:

```bash
uv add pybullet-mm --upgrade
```

### Python Version Warnings

Warnings about Python 3.9 EOL are safe to ignore. For cleaner output, upgrade to Python 3.10+.

## License

MIT
