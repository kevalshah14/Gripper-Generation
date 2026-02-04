#!/usr/bin/env python3
"""
=============================================================================
VLMgineer - Vision Language Models as Robotic Toolsmiths
=============================================================================

Edit config.py to configure your run, then execute:
    uv run main.py

Or run with command-line overrides:
    uv run main.py --task drill_press --iterations 5 --gui
"""

import argparse
import sys
import os

# Import config
import config


def main():
    parser = argparse.ArgumentParser(
        description="VLMgineer: AI-powered robotic tool design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py                          # Use settings from config.py
  uv run main.py --task drill_press       # Override task
  uv run main.py --iterations 5 --gui     # More iterations with GUI
  uv run main.py --demo                   # Quick demo (no API needed)
  uv run main.py --evaluate results.json  # Visualize a saved design
        """
    )
    
    # Task selection
    parser.add_argument(
        "--task", "-t",
        type=str,
        default=config.TASK,
        choices=["bring_cube", "drill_press", "lift_box", "move_ball", "clean_table"],
        help=f"Task to solve (default: {config.TASK})"
    )
    
    # Evolution parameters
    parser.add_argument(
        "--iterations", "-i",
        type=int,
        default=config.ITERATIONS,
        help=f"Number of evolution iterations (default: {config.ITERATIONS})"
    )
    parser.add_argument(
        "--agents", "-a",
        type=int,
        default=config.AGENTS,
        help=f"Number of parallel VLM agents (default: {config.AGENTS})"
    )
    parser.add_argument(
        "--tools",
        type=int,
        default=config.TOOLS_PER_AGENT,
        help=f"Tools per agent (default: {config.TOOLS_PER_AGENT})"
    )
    parser.add_argument(
        "--actions",
        type=int,
        default=config.ACTIONS_PER_TOOL,
        help=f"Actions per tool (default: {config.ACTIONS_PER_TOOL})"
    )
    
    # Display options
    parser.add_argument(
        "--gui", "-g",
        action="store_true",
        default=config.SHOW_GUI,
        help="Show PyBullet GUI window"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=config.VERBOSE,
        help="Print detailed progress"
    )
    
    # Special modes
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run demo mode with example tool (no API needed)"
    )
    parser.add_argument(
        "--evaluate", "-e",
        type=str,
        help="Evaluate a saved design file (JSON)"
    )
    
    args = parser.parse_args()
    
    # Update vlmgineer config from our settings
    from vlmgineer.config import VLMConfig, EvolutionConfig, SimulationConfig, VLMgineerConfig
    
    # Load API key
    api_key = config.API_KEY
    if api_key is None:
        # Try to load from environment or .env
        api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("API_KEY")
        if api_key is None:
            env_file = os.path.join(os.path.dirname(__file__), ".env")
            if os.path.exists(env_file):
                with open(env_file) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            k, v = line.split('=', 1)
                            k = k.strip()
                            v = v.strip().strip('"').strip("'")
                            if k in ("API_KEY", "GOOGLE_API_KEY"):
                                api_key = v
                                break
    
    # Create config
    vlm_config = VLMConfig(
        api_key=api_key,
        model_name=config.MODEL,
        temperature=config.TEMPERATURE,
        max_output_tokens=config.MAX_TOKENS,
    )
    
    evolution_config = EvolutionConfig(
        n_agents=args.agents,
        n_tools=args.tools,
        n_actions=args.actions,
        n_iterations=args.iterations,
    )
    
    sim_config = SimulationConfig(
        gui=args.gui,
    )
    
    full_config = VLMgineerConfig(
        vlm=vlm_config,
        evolution=evolution_config,
        simulation=sim_config,
        output_dir=config.OUTPUT_DIR,
        verbose=args.verbose,
    )
    
    # Import here to avoid slow startup for --help
    from vlmgineer.envs import get_task_env, ToolActionPair
    from vlmgineer.evolution import VLMgineerEvolution
    from vlmgineer.runner import SimulationRunner
    import numpy as np
    import json
    import pybullet as p
    import time
    
    print("=" * 60)
    print("VLMgineer: Vision Language Models as Robotic Toolsmiths")
    print("=" * 60)
    print(f"Task: {args.task}")
    print()
    
    # === DEMO MODE ===
    if args.demo:
        print("Running demo mode with example tool...")
        print("(No API key needed)\n")
        
        env = get_task_env(args.task, gui=args.gui)
        
        # Simple example tool
        example_tool = """
<link name="pusher_arm">
  <inertial><origin xyz="0 0 0"/><mass value="0.005"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0.12 0 0"/>
    <geometry><box size="0.24 0.02 0.02"/></geometry>
    <material name="grey"><color rgba="0.5 0.5 0.55 1"/></material></visual>
  <collision><origin xyz="0.12 0 0"/>
    <geometry><box size="0.24 0.02 0.02"/></geometry></collision>
</link>
<link name="pusher_blade">
  <inertial><origin xyz="0 0 0"/><mass value="0.003"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0 0 -0.03"/>
    <geometry><box size="0.02 0.08 0.06"/></geometry>
    <material name="blue"><color rgba="0.2 0.2 0.8 1"/></material></visual>
  <collision><origin xyz="0 0 -0.03"/>
    <geometry><box size="0.02 0.08 0.06"/></geometry></collision>
</link>
<joint name="arm_joint" type="fixed">
  <origin xyz="0 0 0"/><parent link="tool_mount"/><child link="pusher_arm"/></joint>
<joint name="blade_joint" type="fixed">
  <origin xyz="0.24 0 0"/><parent link="pusher_arm"/><child link="pusher_blade"/></joint>
"""
        
        example_actions = np.array([
            [0.35, 0.0, 0.30, 0.0, 0.3, 0.0, 0.0],
            [0.50, 0.0, 0.20, 0.0, 0.6, 0.0, 0.0],
            [0.60, 0.0, 0.12, 0.0, 0.8, 0.0, 0.0],
            [0.45, 0.0, 0.12, 0.0, 0.8, 0.0, 0.0],
            [0.35, 0.0, 0.25, 0.0, 0.3, 0.0, 0.0],
        ])
        
        env.connect()
        env.load_robot(example_tool)
        env.setup_environment()
        
        tool_action = ToolActionPair(tool_urdf=example_tool, actions=example_actions)
        result = env.evaluate(tool_action, max_steps=600)
        
        print(f"Demo Results:")
        print(f"  Reward: {result.reward:.3f}")
        print(f"  Success: {result.success}")
        
        if args.gui:
            print("\nGUI window open. Close it to exit.")
            try:
                while p.isConnected():
                    p.stepSimulation()
                    time.sleep(1/60)
            except:
                pass
        
        env.disconnect()
        return
    
    # === EVALUATE MODE ===
    if args.evaluate:
        print(f"Evaluating saved design: {args.evaluate}\n")
        
        with open(args.evaluate, 'r') as f:
            design = json.load(f)
        
        print(f"Tool: {design.get('tool_description', 'Custom tool')}")
        print()
        
        env = get_task_env(args.task, gui=args.gui)
        env.connect()
        env.load_robot(design["tool_urdf"])
        env.setup_environment()
        
        tool_action = ToolActionPair(
            tool_urdf=design["tool_urdf"],
            actions=np.array(design["actions"]),
        )
        
        result = env.evaluate(tool_action, max_steps=800)
        
        print(f"Results:")
        print(f"  Reward: {result.reward:.3f}")
        print(f"  Success: {result.success}")
        
        if args.gui:
            print("\nGUI window open. Close it to exit.")
            try:
                while p.isConnected():
                    p.stepSimulation()
                    time.sleep(1/60)
            except:
                pass
        
        env.disconnect()
        return
    
    # === EVOLUTION MODE (main) ===
    if not api_key:
        print("ERROR: No API key found!")
        print("Set API_KEY in config.py or .env file")
        sys.exit(1)
    
    print("Initializing VLMgineer...")
    
    env = get_task_env(args.task, gui=args.gui)
    
    engine = VLMgineerEvolution(
        env=env,
        config=full_config,
    )
    
    print()
    print("Starting evolutionary search...")
    print(f"  Iterations: {args.iterations}")
    print(f"  Agents: {args.agents}")
    print(f"  Tools/agent: {args.tools}")
    print(f"  Actions/tool: {args.actions}")
    print(f"  Total samples/iteration: {args.agents * args.tools * args.actions}")
    print()
    
    try:
        best_design = engine.run()
        
        print()
        print("=" * 60)
        print("Evolution Complete!")
        print("=" * 60)
        
        if best_design:
            print(f"Best reward: {best_design.reward:.3f}")
            print(f"Best tool: {best_design.tool_description[:100]}...")
            
            # Save result
            output_file = os.path.join(
                config.OUTPUT_DIR,
                f"best_{args.task}.json"
            )
            engine.save_results(output_file)
            print(f"\nSaved to: {output_file}")
            print(f"\nTo visualize: uv run main.py --task {args.task} --evaluate {output_file} --gui")
        else:
            print("No successful designs found. Try increasing iterations or check API quota.")
            
    except KeyboardInterrupt:
        print("\n\nInterrupted by user.")
    finally:
        env.disconnect()


if __name__ == "__main__":
    main()
