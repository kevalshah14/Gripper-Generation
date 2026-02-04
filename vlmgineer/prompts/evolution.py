"""
Evolution instruction prompts for VLMgineer - matching the paper exactly (Appendix D.7).
"""

# Evolutionary instructions (from paper Appendix D.7)
EVOLUTION_INSTRUCTION_PROMPT = """(Evolutionary Process) Your design decision is a part of a tool design genetic algorithm. For each of the tool designs, you can choose to either mutate or crossover.

Specifically, tool mutation is defined as one change to a single randomly selected previous tool design. Mutation changes include:
(1) Changing the dimension, location, or orientation of a single component of the tool.
(2) Adding, removing, or replacing a single component of the tool.

Crossover is defined as the process of combining two randomly selected previous tool designs to create a new tool design. Combination is defined as:
(1) Selecting components from two previous tool designs and combining them to form a new tool design.

All mutation and crossover decisions must potentially increase the likelihood of task success, yet all decisions must be different and diverse.

## Previous Elite Designs (Learn from these!)

The following designs achieved the highest rewards. Study their tool geometries and action patterns:

{previous_designs}

For each new design:
1. Decide: MUTATION (modify one elite) or CROSSOVER (combine two elites)
2. Explain your reasoning briefly
3. Generate the new tool URDF and action waypoints"""

SELECTION_CRITERIA_PROMPT = """
## Selection Criteria

Designs are evaluated based on:

1. **Task Reward** (0-1): Primary metric - did the tool help accomplish the task?
   - 1.0 = Task fully completed
   - 0.5 = Partial progress
   - 0.0 = No progress or failure

2. **Efficiency** (secondary): Given similar rewards, prefer:
   - Shorter trajectories (less distance traveled)
   - Simpler tools (fewer components)
   - More robust designs (works from multiple starting positions)

High-reward designs (above threshold) are selected as elites for evolution.
Low-reward designs are discarded.
"""
