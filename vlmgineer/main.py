"""
Main entry point for VLMgineer.

Usage:
    uv run python -m vlmgineer.main --task bring_cube --iterations 3
    uv run python -m vlmgineer.main --task lift_box --gui
    uv run python -m vlmgineer.main --task move_ball --evaluate tool.json
"""

import argparse
import os
import json
import sys
from typing import Optional

from .config import VLMgineerConfig, VLMConfig, EvolutionConfig, SimulationConfig
from .envs import get_task_env, TASK_REGISTRY
from .evolution import VLMgineerEvolution
from .envs.base_env import ToolActionPair
from .runner import SimulationRunner
from .tool_generator import create_example_tool, create_example_actions
import numpy as np


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="VLMgineer: Vision Language Models as Robotic Toolsmiths"
    )
    
    # Task selection
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(TASK_REGISTRY.keys()),
        help="Task to run"
    )
    
    # Evolution parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=3,
        help="Number of evolution iterations (default: 3)"
    )
    parser.add_argument(
        "--agents",
        type=int,
        default=3,
        help="Number of VLM agents (default: 3)"
    )
    parser.add_argument(
        "--tools",
        type=int,
        default=2,
        help="Tools per agent (default: 2)"
    )
    parser.add_argument(
        "--actions",
        type=int,
        default=2,
        help="Actions per tool (default: 2)"
    )
    
    # Simulation parameters
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Show PyBullet GUI during evaluation"
    )
    
    # Output
    parser.add_argument(
        "--output",
        type=str,
        default="vlmgineer_results",
        help="Output directory for results"
    )
    
    # Evaluation mode
    parser.add_argument(
        "--evaluate",
        type=str,
        default=None,
        help="Path to a saved design JSON to evaluate (skips evolution)"
    )
    
    # Demo mode
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo with example tool (no VLM needed)"
    )
    
    # Verbose
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    return parser.parse_args()


def run_evolution(args) -> None:
    """Run the evolutionary search."""
    print("Initializing VLMgineer...")
    
    # Create config
    config = VLMgineerConfig(
        vlm=VLMConfig(),
        evolution=EvolutionConfig(
            n_agents=args.agents,
            n_tools=args.tools,
            n_actions=args.actions,
            n_iterations=args.iterations,
        ),
        simulation=SimulationConfig(
            gui=args.gui,
        ),
        output_dir=args.output,
        verbose=args.verbose,
    )
    
    # Validate config
    try:
        config.validate()
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("\nPlease set the GOOGLE_API_KEY environment variable:")
        print("  export GOOGLE_API_KEY='your-api-key'")
        sys.exit(1)
    
    # Create environment
    env = get_task_env(args.task, gui=args.gui)
    
    # Create and run evolution
    evolution = VLMgineerEvolution(env, config)
    best_design = evolution.run()
    
    # Save results
    evolution.save_results(args.output)
    
    print("\nDone!")


def run_evaluation(args) -> None:
    """Evaluate a saved design."""
    import pybullet as p
    import time
    
    print(f"Loading design from {args.evaluate}...")
    
    with open(args.evaluate, 'r') as f:
        design_data = json.load(f)
    
    print(f"\nTool: {design_data.get('tool_description', 'Custom tool')}\n")
    
    # Create tool-action pair
    tool_action = ToolActionPair(
        tool_urdf=design_data["tool_urdf"],
        actions=np.array(design_data["actions"]),
        tool_description=design_data.get("tool_description", ""),
    )
    
    # Create environment and runner
    env = get_task_env(args.task, gui=args.gui)
    
    if args.gui:
        print("Opening PyBullet GUI...")
        print("Watch the VLM-generated tool in action!")
        print("Close the window when done.\n")
    
    runner = SimulationRunner(env)
    
    print("Evaluating design...")
    result = runner.evaluate_single(tool_action, verbose=args.verbose, keep_open=args.gui)
    
    print(f"\nResults:")
    print(f"  Reward: {result.reward:.3f}")
    print(f"  Success: {result.success}")
    print(f"  Distance traversed: {result.distance:.3f}m")
    
    if result.error:
        print(f"  Error: {result.error}")
    
    # Keep GUI open if requested
    if args.gui:
        print("\nSimulation complete. Close the PyBullet window to exit.")
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1/60)
        except:
            pass
        finally:
            env.disconnect()


def run_demo(args) -> None:
    """Run demo with example tool (no VLM needed)."""
    import pybullet as p
    import time
    
    print("Running demo mode with example tool...")
    print("(No VLM API key needed)\n")
    
    # Create example tool and actions
    tool_urdf = create_example_tool()
    actions = create_example_actions()
    
    print("Example tool: L-shaped pusher")
    print(f"Action waypoints: {len(actions)} steps\n")
    
    # Create tool-action pair
    tool_action = ToolActionPair(
        tool_urdf=tool_urdf,
        actions=actions,
        tool_description="L-shaped pusher tool",
    )
    
    # Create environment and runner
    env = get_task_env(args.task, gui=args.gui)
    
    if args.gui:
        print("Opening PyBullet GUI...")
        print("Watch the robot execute the action sequence.")
        print("Close the window when done.\n")
    
    runner = SimulationRunner(env)
    
    print("Evaluating...")
    result = runner.evaluate_single(tool_action, verbose=True, keep_open=args.gui)
    
    print(f"\nDemo Results:")
    print(f"  Task: {args.task}")
    print(f"  Reward: {result.reward:.3f}")
    print(f"  Success: {result.success}")
    print(f"  Distance traversed: {result.distance:.3f}m")
    
    if result.error:
        print(f"  Error: {result.error}")
    
    # Keep GUI open if requested
    if args.gui:
        print("\nSimulation complete. Close the PyBullet window to exit.")
        try:
            while p.isConnected():
                p.stepSimulation()
                time.sleep(1/60)
        except:
            pass
        finally:
            env.disconnect()


def main():
    """Main entry point."""
    args = parse_args()
    
    print("=" * 60)
    print("VLMgineer: Vision Language Models as Robotic Toolsmiths")
    print("=" * 60)
    print(f"Task: {args.task}")
    print()
    
    if args.demo:
        run_demo(args)
    elif args.evaluate:
        run_evaluation(args)
    else:
        run_evolution(args)


if __name__ == "__main__":
    main()
