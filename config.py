"""
=============================================================================
VLMgineer Configuration
=============================================================================

Edit this file to configure your run, then execute:
    uv run main.py

"""

# =============================================================================
# TASK CONFIGURATION
# =============================================================================

# Which task to run? Options:
#   - "bring_cube"   : Push a distant cube to a target zone
#   - "drill_press"  : Turn a spindle handle to operate drill press
#   - "lift_box"     : Lift a heavy box
#   - "move_ball"    : Move a ball across the table
#   - "clean_table"  : Push debris off the table edge
TASK = "bring_cube"

# =============================================================================
# EVOLUTION PARAMETERS
# =============================================================================

# Number of evolution iterations (more = better results, slower)
ITERATIONS = 1

# Number of parallel VLM agents per iteration
AGENTS = 1

# Tools generated per agent
TOOLS_PER_AGENT = 1

# Action sequences per tool
ACTIONS_PER_TOOL = 1

# =============================================================================
# VLM CONFIGURATION
# =============================================================================

# Gemini model to use
# "gemini-robotics-er-1.5-preview" - Specialized for robotics (best for tool design)
# "gemini-2.0-flash" - Fast general model
MODEL = "gemini-robotics-er-1.5-preview"

# Temperature (0.0 = deterministic, 1.0 = creative)
TEMPERATURE = 0.8

# Max output tokens (increase if responses are truncated)
MAX_TOKENS = 16384

# =============================================================================
# DISPLAY OPTIONS
# =============================================================================

# Show PyBullet GUI window during evaluation?
SHOW_GUI = False

# Print detailed progress information?
VERBOSE = True

# =============================================================================
# OUTPUT
# =============================================================================

# Directory to save results
OUTPUT_DIR = "vlmgineer_results"
