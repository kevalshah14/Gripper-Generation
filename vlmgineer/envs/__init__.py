"""Task environments for VLMgineer."""

from .base_env import BaseEnv, ToolActionPair, EvaluationResult
from .bring_cube import BringCubeEnv
from .lift_box import LiftBoxEnv
from .move_ball import MoveBallEnv
from .clean_table import CleanTableEnv
from .drill_press import DrillPressEnv

__all__ = [
    "BaseEnv",
    "ToolActionPair",
    "EvaluationResult",
    "BringCubeEnv",
    "LiftBoxEnv",
    "MoveBallEnv",
    "CleanTableEnv",
    "DrillPressEnv",
]

# Task registry for easy lookup
TASK_REGISTRY = {
    "bring_cube": BringCubeEnv,
    "lift_box": LiftBoxEnv,
    "move_ball": MoveBallEnv,
    "clean_table": CleanTableEnv,
    "drill_press": DrillPressEnv,
}


def get_task_env(task_name: str, **kwargs):
    """
    Get a task environment by name.
    
    Args:
        task_name: Name of the task
        **kwargs: Additional arguments for the environment
        
    Returns:
        Task environment instance
    """
    if task_name not in TASK_REGISTRY:
        raise ValueError(
            f"Unknown task: {task_name}. "
            f"Available tasks: {list(TASK_REGISTRY.keys())}"
        )
    return TASK_REGISTRY[task_name](**kwargs)
