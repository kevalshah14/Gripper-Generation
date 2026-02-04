"""
Evolution instruction prompts for VLMgineer.
"""

EVOLUTION_INSTRUCTION_PROMPT = """
## Evolutionary Process

Your design is part of a genetic algorithm for tool creation. You must either MUTATE or perform CROSSOVER:

### Mutation (change one tool)
Pick ONE previous tool and make exactly ONE change:
1. **Dimension change**: Modify the size of one component (make longer, wider, thinner)
2. **Position change**: Move one component's attachment point
3. **Orientation change**: Rotate one component
4. **Add component**: Add a new box part to the tool
5. **Remove component**: Remove one part (if multiple exist)
6. **Replace component**: Swap one component for a different shape/size

### Crossover (combine two tools)
Pick TWO previous tools and combine their elements:
1. Take components from both designs
2. Create a new tool that incorporates ideas from each
3. The result should be a coherent, functional design

### Guidelines
- Learn from rewards: High-reward designs have good ideas to keep
- Learn from failures: Low-reward designs show what to avoid
- Stay diverse: Don't just make small tweaks to the best design
- Be creative: Sometimes a big change leads to breakthrough
- Physical intuition: Changes should make physical sense

For each design you create:
1. State whether it's a mutation or crossover
2. Explain what you changed and why
3. Predict how this change might improve performance
"""

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

High-reward designs (>0.6) are candidates for evolution.
Low-reward designs (<0.3) should be significantly modified or abandoned.
"""
