"""
Simple RL-based policy trainer for VLMgineer tasks.
Uses evolutionary strategy with random mutations - no API required.
"""

import numpy as np
import json
import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional
import pybullet as p

from .envs import get_task_env, ToolActionPair


@dataclass
class Individual:
    """An individual in the population (tool + actions)."""
    tool_urdf: str
    actions: np.ndarray
    reward: float = 0.0
    description: str = ""


class RLPolicyTrainer:
    """
    Reinforcement Learning trainer using evolutionary strategies.
    
    Uses (1+λ)-ES: Generate λ offspring from the best individual,
    keep the best one for the next generation.
    """
    
    def __init__(
        self,
        task_name: str,
        population_size: int = 20,
        n_generations: int = 50,
        mutation_rate: float = 0.3,
        action_noise: float = 0.05,
        gui: bool = False,
        verbose: bool = True,
    ):
        self.task_name = task_name
        self.population_size = population_size
        self.n_generations = n_generations
        self.mutation_rate = mutation_rate
        self.action_noise = action_noise
        self.gui = gui
        self.verbose = verbose
        
        # Tool templates for the drill press task
        self.tool_templates = self._get_tool_templates()
        
        # Best individual found
        self.best_individual: Optional[Individual] = None
        self.history: List[dict] = []
        
    def _get_tool_templates(self) -> List[dict]:
        """Get parameterized tool templates."""
        return [
            {
                "name": "hook",
                "description": "Hook gripper for pulling handle",
                "params": {
                    "arm_length": (0.10, 0.20),
                    "hook_depth": (0.04, 0.10),
                    "hook_width": (0.03, 0.06),
                }
            },
            {
                "name": "fork",
                "description": "Fork gripper for capturing handle",
                "params": {
                    "arm_length": (0.10, 0.18),
                    "prong_length": (0.05, 0.10),
                    "prong_spacing": (0.025, 0.045),
                }
            },
            {
                "name": "clamp",
                "description": "C-clamp gripper for squeezing handle",
                "params": {
                    "arm_length": (0.08, 0.15),
                    "jaw_length": (0.04, 0.08),
                    "jaw_spacing": (0.015, 0.030),
                }
            },
            {
                "name": "paddle",
                "description": "Paddle pusher for pushing handle",
                "params": {
                    "arm_length": (0.12, 0.20),
                    "paddle_width": (0.04, 0.08),
                    "paddle_height": (0.03, 0.06),
                }
            },
        ]
    
    def _generate_tool_urdf(self, template_name: str, params: dict) -> str:
        """Generate URDF for a tool given parameters."""
        
        if template_name == "hook":
            arm_len = params.get("arm_length", 0.15)
            hook_depth = params.get("hook_depth", 0.07)
            hook_width = params.get("hook_width", 0.04)
            
            return f"""
<link name="arm">
  <inertial><origin xyz="0 0 0"/><mass value="0.005"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="{arm_len/2} 0 0"/>
    <geometry><box size="{arm_len} 0.02 0.02"/></geometry>
    <material name="grey"><color rgba="0.5 0.5 0.55 1"/></material></visual>
  <collision><origin xyz="{arm_len/2} 0 0"/>
    <geometry><box size="{arm_len} 0.02 0.02"/></geometry></collision>
</link>
<link name="hook_stem">
  <inertial><origin xyz="0 0 0"/><mass value="0.003"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0 0 {-hook_depth/2}"/>
    <geometry><box size="0.02 0.02 {hook_depth}"/></geometry>
    <material name="orange"><color rgba="0.9 0.5 0.1 1"/></material></visual>
  <collision><origin xyz="0 0 {-hook_depth/2}"/>
    <geometry><box size="0.02 0.02 {hook_depth}"/></geometry></collision>
</link>
<link name="hook_tip">
  <inertial><origin xyz="0 0 0"/><mass value="0.002"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="{-hook_width/2} 0 0"/>
    <geometry><box size="{hook_width} 0.02 0.02"/></geometry>
    <material name="orange"><color rgba="0.9 0.5 0.1 1"/></material></visual>
  <collision><origin xyz="{-hook_width/2} 0 0"/>
    <geometry><box size="{hook_width} 0.02 0.02"/></geometry></collision>
</link>
<joint name="arm_joint" type="fixed">
  <origin xyz="0 0 0"/><parent link="tool_mount"/><child link="arm"/></joint>
<joint name="stem_joint" type="fixed">
  <origin xyz="{arm_len} 0 0"/><parent link="arm"/><child link="hook_stem"/></joint>
<joint name="tip_joint" type="fixed">
  <origin xyz="0 0 {-hook_depth}"/><parent link="hook_stem"/><child link="hook_tip"/></joint>
"""
        
        elif template_name == "fork":
            arm_len = params.get("arm_length", 0.14)
            prong_len = params.get("prong_length", 0.07)
            spacing = params.get("prong_spacing", 0.035)
            
            return f"""
<link name="arm">
  <inertial><origin xyz="0 0 0"/><mass value="0.005"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="{arm_len/2} 0 0"/>
    <geometry><box size="{arm_len} 0.025 0.02"/></geometry>
    <material name="grey"><color rgba="0.5 0.5 0.55 1"/></material></visual>
  <collision><origin xyz="{arm_len/2} 0 0"/>
    <geometry><box size="{arm_len} 0.025 0.02"/></geometry></collision>
</link>
<link name="left_prong">
  <inertial><origin xyz="0 0 0"/><mass value="0.002"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0 0 {-prong_len/2}"/>
    <geometry><box size="0.015 0.012 {prong_len}"/></geometry>
    <material name="yellow"><color rgba="0.9 0.75 0.1 1"/></material></visual>
  <collision><origin xyz="0 0 {-prong_len/2}"/>
    <geometry><box size="0.015 0.012 {prong_len}"/></geometry></collision>
</link>
<link name="right_prong">
  <inertial><origin xyz="0 0 0"/><mass value="0.002"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0 0 {-prong_len/2}"/>
    <geometry><box size="0.015 0.012 {prong_len}"/></geometry>
    <material name="yellow"><color rgba="0.9 0.75 0.1 1"/></material></visual>
  <collision><origin xyz="0 0 {-prong_len/2}"/>
    <geometry><box size="0.015 0.012 {prong_len}"/></geometry></collision>
</link>
<joint name="arm_joint" type="fixed">
  <origin xyz="0 0 0"/><parent link="tool_mount"/><child link="arm"/></joint>
<joint name="left_joint" type="fixed">
  <origin xyz="{arm_len} {spacing} -0.01"/><parent link="arm"/><child link="left_prong"/></joint>
<joint name="right_joint" type="fixed">
  <origin xyz="{arm_len} {-spacing} -0.01"/><parent link="arm"/><child link="right_prong"/></joint>
"""
        
        elif template_name == "clamp":
            arm_len = params.get("arm_length", 0.12)
            jaw_len = params.get("jaw_length", 0.06)
            spacing = params.get("jaw_spacing", 0.022)
            
            return f"""
<link name="arm">
  <inertial><origin xyz="0 0 0"/><mass value="0.005"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="{arm_len/2} 0 0"/>
    <geometry><box size="{arm_len} 0.025 0.025"/></geometry>
    <material name="grey"><color rgba="0.5 0.5 0.55 1"/></material></visual>
  <collision><origin xyz="{arm_len/2} 0 0"/>
    <geometry><box size="{arm_len} 0.025 0.025"/></geometry></collision>
</link>
<link name="left_jaw">
  <inertial><origin xyz="0 0 0"/><mass value="0.003"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0 {spacing/2} {-jaw_len/2}"/>
    <geometry><box size="0.02 0.035 {jaw_len}"/></geometry>
    <material name="blue"><color rgba="0.2 0.2 0.8 1"/></material></visual>
  <collision><origin xyz="0 {spacing/2} {-jaw_len/2}"/>
    <geometry><box size="0.02 0.035 {jaw_len}"/></geometry></collision>
</link>
<link name="right_jaw">
  <inertial><origin xyz="0 0 0"/><mass value="0.003"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0 {-spacing/2} {-jaw_len/2}"/>
    <geometry><box size="0.02 0.035 {jaw_len}"/></geometry>
    <material name="blue"><color rgba="0.2 0.2 0.8 1"/></material></visual>
  <collision><origin xyz="0 {-spacing/2} {-jaw_len/2}"/>
    <geometry><box size="0.02 0.035 {jaw_len}"/></geometry></collision>
</link>
<joint name="arm_joint" type="fixed">
  <origin xyz="0 0 0"/><parent link="tool_mount"/><child link="arm"/></joint>
<joint name="left_joint" type="fixed">
  <origin xyz="{arm_len} 0 0"/><parent link="arm"/><child link="left_jaw"/></joint>
<joint name="right_joint" type="fixed">
  <origin xyz="{arm_len} 0 0"/><parent link="arm"/><child link="right_jaw"/></joint>
"""
        
        elif template_name == "paddle":
            arm_len = params.get("arm_length", 0.16)
            paddle_w = params.get("paddle_width", 0.06)
            paddle_h = params.get("paddle_height", 0.04)
            
            return f"""
<link name="arm">
  <inertial><origin xyz="0 0 0"/><mass value="0.005"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="{arm_len/2} 0 0"/>
    <geometry><box size="{arm_len} 0.02 0.02"/></geometry>
    <material name="grey"><color rgba="0.5 0.5 0.55 1"/></material></visual>
  <collision><origin xyz="{arm_len/2} 0 0"/>
    <geometry><box size="{arm_len} 0.02 0.02"/></geometry></collision>
</link>
<link name="paddle">
  <inertial><origin xyz="0 0 0"/><mass value="0.004"/>
    <inertia ixx="0.0001" ixy="0" ixz="0" iyy="0.0001" iyz="0" izz="0.0001"/></inertial>
  <visual><origin xyz="0 0 {-paddle_h/2}"/>
    <geometry><box size="0.015 {paddle_w} {paddle_h}"/></geometry>
    <material name="green"><color rgba="0.2 0.7 0.3 1"/></material></visual>
  <collision><origin xyz="0 0 {-paddle_h/2}"/>
    <geometry><box size="0.015 {paddle_w} {paddle_h}"/></geometry></collision>
</link>
<joint name="arm_joint" type="fixed">
  <origin xyz="0 0 0"/><parent link="tool_mount"/><child link="arm"/></joint>
<joint name="paddle_joint" type="fixed">
  <origin xyz="{arm_len} 0 0"/><parent link="arm"/><child link="paddle"/></joint>
"""
        
        return ""
    
    def _generate_random_actions(self, n_waypoints: int = 8) -> np.ndarray:
        """Generate random action waypoints for drill press task."""
        # Start position (approach)
        actions = []
        
        # Handle approximate position: (0.63, -0.03, 0.515)
        handle_x, handle_y, handle_z = 0.63, -0.03, 0.515
        
        # Approach from above/behind
        actions.append([
            np.random.uniform(0.35, 0.45),  # x
            np.random.uniform(-0.15, 0.0),   # y
            np.random.uniform(0.55, 0.65),   # z
            0.0, np.random.uniform(0.3, 0.8), 0.0, 0.0  # orientation + gripper
        ])
        
        # Move towards handle
        actions.append([
            np.random.uniform(0.50, 0.58),
            np.random.uniform(-0.08, 0.0),
            np.random.uniform(0.50, 0.58),
            0.0, np.random.uniform(1.0, 1.5), 0.0, 0.0
        ])
        
        # Engage handle
        actions.append([
            np.random.uniform(0.58, 0.66),
            np.random.uniform(-0.05, 0.02),
            np.random.uniform(0.48, 0.54),
            0.0, np.random.uniform(1.3, 1.7), 0.0, 1.0
        ])
        
        # Arc motion (turning the spindle)
        for i in range(3):
            angle = (i + 1) * np.random.uniform(0.2, 0.5)
            actions.append([
                handle_x + np.random.uniform(-0.08, 0.02),
                handle_y + np.random.uniform(0.03, 0.12) * (i + 1),
                handle_z + np.random.uniform(-0.06, -0.02),
                np.random.uniform(0.0, 0.4),
                np.random.uniform(1.2, 1.7),
                np.random.uniform(0.1, 0.6) * (i + 1),
                1.0
            ])
        
        # Release and retract
        actions.append([
            np.random.uniform(0.45, 0.55),
            np.random.uniform(0.0, 0.10),
            np.random.uniform(0.52, 0.60),
            0.0, np.random.uniform(0.8, 1.2), 0.0, 0.0
        ])
        
        actions.append([
            np.random.uniform(0.35, 0.45),
            np.random.uniform(-0.10, 0.0),
            np.random.uniform(0.58, 0.65),
            0.0, np.random.uniform(0.3, 0.6), 0.0, 0.0
        ])
        
        return np.array(actions)
    
    def _mutate_actions(self, actions: np.ndarray) -> np.ndarray:
        """Mutate action waypoints."""
        mutated = actions.copy()
        
        for i in range(len(mutated)):
            if np.random.random() < self.mutation_rate:
                # Mutate position
                mutated[i, :3] += np.random.normal(0, self.action_noise, 3)
                # Mutate orientation
                mutated[i, 3:6] += np.random.normal(0, self.action_noise * 2, 3)
                # Clip gripper to 0 or 1
                mutated[i, 6] = np.clip(mutated[i, 6] + np.random.normal(0, 0.3), 0, 1)
                mutated[i, 6] = 1.0 if mutated[i, 6] > 0.5 else 0.0
        
        # Ensure reasonable bounds
        mutated[:, 0] = np.clip(mutated[:, 0], 0.3, 0.7)  # x
        mutated[:, 1] = np.clip(mutated[:, 1], -0.2, 0.2)  # y
        mutated[:, 2] = np.clip(mutated[:, 2], 0.1, 0.7)   # z
        
        return mutated
    
    def _mutate_tool_params(self, template: dict, params: dict) -> dict:
        """Mutate tool parameters."""
        mutated = params.copy()
        
        for key, (min_val, max_val) in template["params"].items():
            if np.random.random() < self.mutation_rate:
                current = mutated.get(key, (min_val + max_val) / 2)
                noise = (max_val - min_val) * 0.1
                mutated[key] = np.clip(
                    current + np.random.normal(0, noise),
                    min_val, max_val
                )
        
        return mutated
    
    def _evaluate(self, individual: Individual) -> float:
        """Evaluate an individual in simulation."""
        env = get_task_env(self.task_name, gui=False)
        
        try:
            env.connect()
            env.load_robot(individual.tool_urdf)
            env.setup_environment()
            
            tool_action = ToolActionPair(
                tool_urdf=individual.tool_urdf,
                actions=individual.actions,
            )
            
            result = env.evaluate(tool_action, max_steps=800)
            return float(result.reward)
            
        except Exception as e:
            if self.verbose:
                print(f"    Evaluation error: {e}")
            return 0.0
        finally:
            env.disconnect()
    
    def _create_random_individual(self) -> Individual:
        """Create a random individual."""
        template = np.random.choice(self.tool_templates)
        
        # Random parameters within bounds
        params = {}
        for key, (min_val, max_val) in template["params"].items():
            params[key] = np.random.uniform(min_val, max_val)
        
        tool_urdf = self._generate_tool_urdf(template["name"], params)
        actions = self._generate_random_actions()
        
        return Individual(
            tool_urdf=tool_urdf,
            actions=actions,
            description=f"{template['name']} - {template['description']}",
        )
    
    def train(self) -> Individual:
        """Run the evolutionary training."""
        print("=" * 60)
        print("RL Policy Trainer - Evolutionary Strategy")
        print("=" * 60)
        print(f"Task: {self.task_name}")
        print(f"Population: {self.population_size}")
        print(f"Generations: {self.n_generations}")
        print("=" * 60)
        
        # Initialize population
        print("\nGenerating initial population...")
        population = [self._create_random_individual() for _ in range(self.population_size)]
        
        # Evaluate initial population
        print("Evaluating initial population...")
        for i, ind in enumerate(population):
            ind.reward = self._evaluate(ind)
            if self.verbose:
                print(f"  [{i+1}/{self.population_size}] Reward: {ind.reward:.4f}")
        
        # Sort by reward
        population.sort(key=lambda x: x.reward, reverse=True)
        self.best_individual = population[0]
        
        print(f"\nInitial best reward: {self.best_individual.reward:.4f}")
        print(f"Initial best tool: {self.best_individual.description}")
        
        # Evolution loop
        for gen in range(self.n_generations):
            print(f"\n--- Generation {gen + 1}/{self.n_generations} ---")
            
            # Keep top 20% as elites
            n_elites = max(2, self.population_size // 5)
            elites = population[:n_elites]
            
            # Generate offspring from elites
            offspring = []
            for _ in range(self.population_size - n_elites):
                parent = np.random.choice(elites)
                
                # Mutate actions
                new_actions = self._mutate_actions(parent.actions)
                
                # Occasionally mutate tool
                if np.random.random() < 0.2:
                    child = self._create_random_individual()
                    child.actions = new_actions
                else:
                    child = Individual(
                        tool_urdf=parent.tool_urdf,
                        actions=new_actions,
                        description=parent.description,
                    )
                
                offspring.append(child)
            
            # Evaluate offspring
            for i, ind in enumerate(offspring):
                ind.reward = self._evaluate(ind)
                if self.verbose and (i + 1) % 5 == 0:
                    print(f"  Evaluated {i+1}/{len(offspring)}, best so far: {max(ind.reward for ind in offspring[:i+1]):.4f}")
            
            # Combine and select
            population = elites + offspring
            population.sort(key=lambda x: x.reward, reverse=True)
            population = population[:self.population_size]
            
            # Update best
            if population[0].reward > self.best_individual.reward:
                self.best_individual = population[0]
                print(f"  ★ New best! Reward: {self.best_individual.reward:.4f}")
            
            # Log progress
            gen_best = population[0].reward
            gen_avg = np.mean([ind.reward for ind in population])
            print(f"  Gen {gen+1}: Best={gen_best:.4f}, Avg={gen_avg:.4f}")
            
            self.history.append({
                "generation": gen + 1,
                "best_reward": gen_best,
                "avg_reward": gen_avg,
            })
            
            # Early stopping if solved
            if self.best_individual.reward >= 0.95:
                print("\n✓ Task solved!")
                break
        
        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)
        print(f"Best reward: {self.best_individual.reward:.4f}")
        print(f"Best tool: {self.best_individual.description}")
        
        return self.best_individual
    
    def save_best(self, filepath: str) -> None:
        """Save the best individual to a JSON file."""
        data = {
            "reward": self.best_individual.reward,
            "success": self.best_individual.reward >= 0.8,
            "tool_description": self.best_individual.description,
            "tool_urdf": self.best_individual.tool_urdf,
            "actions": self.best_individual.actions.tolist(),
            "training_history": self.history,
        }
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved best policy to: {filepath}")


def main():
    """Run RL training for drill press task."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RL Policy Trainer")
    parser.add_argument("--task", type=str, default="drill_press", help="Task name")
    parser.add_argument("--population", type=int, default=15, help="Population size")
    parser.add_argument("--generations", type=int, default=30, help="Number of generations")
    parser.add_argument("--output", type=str, default="vlmgineer_results/rl_trained_policy.json")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    trainer = RLPolicyTrainer(
        task_name=args.task,
        population_size=args.population,
        n_generations=args.generations,
        verbose=args.verbose,
    )
    
    best = trainer.train()
    trainer.save_best(args.output)


if __name__ == "__main__":
    main()
