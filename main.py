#!/usr/bin/env python3
"""
=============================================================================
VLMgineer - Vision Language Models as Robotic Toolsmiths
=============================================================================

1. Edit config.py to set your preferences
2. Run: uv run main.py

Or use command-line overrides:
    uv run main.py --task drill_press --gui
"""

import argparse
import sys
import os


def main():
    # Import user config
    import config
    
    parser = argparse.ArgumentParser(
        description="VLMgineer: AI-powered robotic tool design",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run main.py                          # Use settings from config.py
  uv run main.py --task drill_press       # Override task
  uv run main.py --gui                    # Show PyBullet GUI
  uv run main.py --demo                   # Quick demo (no API needed)
  uv run main.py --evaluate result.json   # Visualize a saved design
        """
    )
    
    parser.add_argument("--task", "-t", type=str, default=config.TASK,
        choices=["bring_cube", "drill_press", "lift_box", "move_ball", "clean_table"])
    parser.add_argument("--iterations", "-i", type=int, default=config.ITERATIONS)
    parser.add_argument("--agents", "-a", type=int, default=config.AGENTS)
    parser.add_argument("--tools", type=int, default=config.TOOLS_PER_AGENT)
    parser.add_argument("--actions", type=int, default=config.ACTIONS_PER_TOOL)
    parser.add_argument("--gui", "-g", action="store_true", default=config.SHOW_GUI)
    parser.add_argument("--verbose", "-v", action="store_true", default=config.VERBOSE)
    parser.add_argument("--demo", action="store_true", help="Demo mode (no API needed)")
    parser.add_argument("--evaluate", "-e", type=str, help="Evaluate a saved design")
    
    args = parser.parse_args()
    
    # Imports (slow, so do after arg parsing)
    from vlmgineer.config import VLMConfig, EvolutionConfig, SimulationConfig, VLMgineerConfig, _load_api_key
    from vlmgineer.envs import get_task_env, ToolActionPair
    from vlmgineer.evolution import VLMgineerEvolution
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
        print("Demo mode - example tool (no API needed)\n")
        
        env = get_task_env(args.task, gui=args.gui)
        
        tool = """
<link name="arm"><inertial><origin xyz="0 0 0"/><mass value="0.005"/>
  <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0.12 0 0"/><geometry><box size="0.24 0.02 0.02"/></geometry>
  <material name="grey"><color rgba="0.5 0.5 0.55 1"/></material></visual>
  <collision><origin xyz="0.12 0 0"/><geometry><box size="0.24 0.02 0.02"/></geometry></collision></link>
<link name="blade"><inertial><origin xyz="0 0 0"/><mass value="0.003"/>
  <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0 0 -0.03"/><geometry><box size="0.02 0.08 0.06"/></geometry>
  <material name="blue"><color rgba="0.2 0.2 0.8 1"/></material></visual>
  <collision><origin xyz="0 0 -0.03"/><geometry><box size="0.02 0.08 0.06"/></geometry></collision></link>
<joint name="arm_j" type="fixed"><origin xyz="0 0 0"/><parent link="tool_mount"/><child link="arm"/></joint>
<joint name="blade_j" type="fixed"><origin xyz="0.24 0 0"/><parent link="arm"/><child link="blade"/></joint>
"""
        actions = np.array([
            [0.35, 0.0, 0.30, 0, 0.3, 0, 0],
            [0.50, 0.0, 0.20, 0, 0.6, 0, 0],
            [0.60, 0.0, 0.12, 0, 0.8, 0, 0],
            [0.45, 0.0, 0.12, 0, 0.8, 0, 0],
            [0.35, 0.0, 0.25, 0, 0.3, 0, 0],
        ])
        
        env.connect()
        env.load_robot(tool)
        env.setup_environment()
        
        result = env.evaluate(ToolActionPair(tool_urdf=tool, actions=actions), max_steps=600)
        print(f"Result: reward={result.reward:.3f}, success={result.success}")
        
        if args.gui:
            print("\nClose PyBullet window to exit.")
            try:
                while p.isConnected():
                    p.stepSimulation()
                    time.sleep(1/60)
            except: pass
        env.disconnect()
        return
    
    # === EVALUATE MODE ===
    if args.evaluate:
        print(f"Evaluating: {args.evaluate}\n")
        
        with open(args.evaluate) as f:
            design = json.load(f)
        
        print(f"Tool: {design.get('tool_description', 'Custom')}\n")
        
        env = get_task_env(args.task, gui=args.gui)
        env.connect()
        env.load_robot(design["tool_urdf"])
        env.setup_environment()
        
        result = env.evaluate(
            ToolActionPair(tool_urdf=design["tool_urdf"], actions=np.array(design["actions"])),
            max_steps=800
        )
        print(f"Result: reward={result.reward:.3f}, success={result.success}")
        
        if args.gui:
            print("\nClose PyBullet window to exit.")
            try:
                while p.isConnected():
                    p.stepSimulation()
                    time.sleep(1/60)
            except: pass
        env.disconnect()
        return
    
    # === EVOLUTION MODE ===
    api_key = _load_api_key()
    if not api_key:
        print("ERROR: No API key found!")
        print("Add API_KEY=your_key to .env file")
        sys.exit(1)
    
    # Build config from settings
    full_config = VLMgineerConfig(
        vlm=VLMConfig(
            api_key=api_key,
            model_name=config.MODEL,
            temperature=config.TEMPERATURE,
            max_output_tokens=config.MAX_TOKENS,
        ),
        evolution=EvolutionConfig(
            n_agents=args.agents,
            n_tools=args.tools,
            n_actions=args.actions,
            n_iterations=args.iterations,
        ),
        simulation=SimulationConfig(gui=args.gui),
        output_dir=config.OUTPUT_DIR,
        verbose=args.verbose,
    )
    
    print(f"Model: {config.MODEL}")
    print(f"Iterations: {args.iterations}")
    print(f"Samples/iteration: {args.agents * args.tools * args.actions}")
    print()
    
    env = get_task_env(args.task, gui=args.gui)
    engine = VLMgineerEvolution(env=env, config=full_config)
    
    try:
        best = engine.run()
        
        print("\n" + "=" * 60)
        print("Complete!")
        print("=" * 60)
        
        # Always save results (even if all rewards are 0)
        engine.save_results(config.OUTPUT_DIR)
        output = os.path.join(config.OUTPUT_DIR, f"best_{args.task}.json")
        
        if best:
            print(f"\nBest reward: {best.reward:.3f}")
            print(f"Best design: {best.design.tool_description}")
            
            # Show the generated URDF and actions
            print("\n" + "-" * 40)
            print("Generated Tool URDF:")
            print("-" * 40)
            print(best.design.tool_urdf[:1500] + "..." if len(best.design.tool_urdf) > 1500 else best.design.tool_urdf)
            
            print("\n" + "-" * 40)
            print("Generated Actions (waypoints):")
            print("-" * 40)
            print(f"Shape: {best.design.actions.shape}")
            print(best.design.actions)
            
            # Save to the named file so the example command works
            actions_list = best.design.actions
            if hasattr(actions_list, 'tolist'):
                actions_list = actions_list.tolist()
                
            with open(output, 'w') as f:
                json.dump({
                    "reward": float(best.reward),
                    "success": bool(best.success),
                    "tool_urdf": best.design.tool_urdf,
                    "actions": actions_list,
                    "tool_description": best.design.tool_description,
                }, f, indent=2)
            
            print(f"\nSaved to: {output}")
            print(f"Visualize: uv run main.py --task {args.task} --evaluate {output} --gui")
        else:
            print("No valid designs generated. Try more iterations or check API quota.")
            
    except KeyboardInterrupt:
        print("\n\nStopped.")
    finally:
        env.disconnect()


if __name__ == "__main__":
    main()
