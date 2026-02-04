"""
=============================================================================
VLMgineer Configuration
=============================================================================

Edit this file to configure your run, then execute:
    uv run main.py

Based on the VLMgineer paper (Gao et al. 2025):
- Paper uses nagents=20, ntool=10, naction=10 (2000 samples/iter) for best results
- We use smaller defaults for faster iteration during development

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
TASK = "lift_box"

# =============================================================================
# EVOLUTION PARAMETERS (Paper: nagents=20, ntool=10, naction=10, niter=3)
# =============================================================================

# Number of evolution iterations (paper uses 3)
ITERATIONS = 1

# Number of parallel VLM agents per iteration (paper uses 20)
# Each agent generates TOOLS_PER_AGENT * ACTIONS_PER_TOOL designs
AGENTS = 1

# Tools generated per agent (paper uses 10)
TOOLS_PER_AGENT = 1

# Action sequences per tool (paper uses 10)
ACTIONS_PER_TOOL = 1

# Total samples per iteration = AGENTS * TOOLS_PER_AGENT * ACTIONS_PER_TOOL
# Paper: 20 * 10 * 10 = 2000 samples/iteration
# Default: 3 * 3 * 3 = 27 samples/iteration

# Selection parameters for evolution
TOP_K = 5  # Number of elite designs to keep (paper uses 5)
REWARD_THRESHOLD = 0.3  # Minimum reward to be considered elite (paper uses 0.6 for BringCube)

# =============================================================================
# VLM CONFIGURATION
# =============================================================================

# Gemini model to use
# "gemini-2.5-flash" - Fast and capable
# "gemini-2.5-pro" - Best for complex reasoning
# "gemini-robotics-er-1.5-preview" - Specialized for robotics (low quota: 20/day)
MODEL = "gemini-2.5-flash"

# Temperature (0.0 = deterministic, 1.0 = creative)
# Paper doesn't specify, but moderate creativity helps diversity
TEMPERATURE = 0.7

# Max output tokens (must be high enough for multiple tools + actions)
MAX_TOKENS = 65536

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

# Save all designs (not just best)? Useful for analysis
SAVE_ALL_DESIGNS = True
