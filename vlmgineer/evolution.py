"""
Evolutionary search algorithm for VLMgineer.

Implements Algorithm 1 from the paper: evolutionary co-design of tools and actions.
"""

import os
import json
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from PIL import Image

from .config import VLMgineerConfig, EvolutionConfig
from .vlm_client import VLMClient, ToolActionDesign
from .tool_generator import ToolGenerator, GenerationContext
from .envs.base_env import BaseEnv, ToolActionPair, EvaluationResult
from .runner import SimulationRunner, RunResult
from .prompts import (
    INITIAL_MISSION_PROMPT,
    EVOLUTION_MISSION_PROMPT,
    PROCEDURE_PROMPT,
    TOOL_SPECIFICATION_PROMPT,
    ACTION_SPECIFICATION_PROMPT,
    ACTION_DIVERSITY_PROMPT,
    FRAME_CLARIFICATION_PROMPT,
    EVOLUTION_INSTRUCTION_PROMPT,
)


@dataclass
class DesignWithReward:
    """A design paired with its evaluation results."""
    
    design: ToolActionDesign
    reward: float
    success: bool
    distance: float
    iteration: int
    design_id: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for VLM context."""
        return {
            "tool_urdf": self.design.tool_urdf,
            "tool_description": self.design.tool_description,
            "reward": self.reward,
            "success": self.success,
            "iteration": self.iteration,
        }


@dataclass
class EvolutionState:
    """State of the evolutionary search."""
    
    iteration: int = 0
    all_designs: List[DesignWithReward] = field(default_factory=list)
    elite_designs: List[DesignWithReward] = field(default_factory=list)
    best_design: Optional[DesignWithReward] = None
    best_reward: float = 0.0
    
    # Statistics
    rewards_per_iteration: List[List[float]] = field(default_factory=list)
    best_rewards_per_iteration: List[float] = field(default_factory=list)
    
    def add_iteration_results(
        self,
        designs: List[DesignWithReward],
        top_k: int,
        reward_threshold: float
    ) -> None:
        """
        Add results from an iteration and update state.
        
        Args:
            designs: Evaluated designs from this iteration
            top_k: Number of elite designs to keep
            reward_threshold: Minimum reward to be considered elite
        """
        self.all_designs.extend(designs)
        
        # Record rewards
        rewards = [d.reward for d in designs]
        self.rewards_per_iteration.append(rewards)
        
        # Update best
        for design in designs:
            if design.reward > self.best_reward:
                self.best_reward = design.reward
                self.best_design = design
        
        self.best_rewards_per_iteration.append(self.best_reward)
        
        # Select elite designs
        # First filter by threshold, then take top-k
        qualified = [d for d in self.all_designs if d.reward >= reward_threshold]
        qualified.sort(key=lambda x: x.reward, reverse=True)
        self.elite_designs = qualified[:top_k]
        
        self.iteration += 1


class VLMgineerEvolution:
    """
    Main evolutionary search class.
    
    Implements the VLMgineer algorithm:
    1. Generate initial population with VLM
    2. Evaluate in simulation
    3. Select top-k designs
    4. Evolve via VLM-guided mutation/crossover
    5. Repeat
    """
    
    def __init__(
        self,
        env: BaseEnv,
        config: Optional[VLMgineerConfig] = None,
    ):
        """
        Initialize the evolutionary search.
        
        Args:
            env: Task environment
            config: VLMgineer configuration
        """
        self.env = env
        self.config = config or VLMgineerConfig()
        
        # Initialize components
        self.vlm_client = VLMClient(self.config.vlm)
        self.runner = SimulationRunner(env, self.config.simulation)
        
        # State
        self.state = EvolutionState()
        
        # Cache scene image
        self._scene_image: Optional[Image.Image] = None
    
    def _capture_scene_image(self) -> Optional[Image.Image]:
        """Capture initial scene image for VLM context."""
        try:
            # Connect and setup environment temporarily
            self.env.connect()
            self.env.load_robot()
            self.env.setup_environment()
            image = self.env.capture_image()
            self.env.disconnect()
            return image
        except Exception as e:
            print(f"Warning: Could not capture scene image: {e}")
            return None
    
    def _build_action_spec(self) -> str:
        """Build the action specification prompt."""
        return (
            ACTION_SPECIFICATION_PROMPT + 
            "\n" + 
            ACTION_DIVERSITY_PROMPT + 
            "\n" + 
            FRAME_CLARIFICATION_PROMPT
        )
    
    def _generate_initial_population(self) -> List[ToolActionDesign]:
        """Generate the initial population of designs."""
        evo_config = self.config.evolution
        
        # Build system prompt
        system_prompt = INITIAL_MISSION_PROMPT + "\n" + PROCEDURE_PROMPT
        
        # Generate designs
        designs = self.vlm_client.sample_population(
            task_description=self.env.get_task_description(),
            environment_code=self.env.get_environment_code(),
            frame_description=self.env.get_frame_description(),
            system_prompt=system_prompt,
            tool_spec=TOOL_SPECIFICATION_PROMPT,
            action_spec=self._build_action_spec(),
            scene_image=self._scene_image,
            n_agents=evo_config.n_agents,
            n_tools=evo_config.n_tools,
            n_actions=evo_config.n_actions,
        )
        
        return designs
    
    def _generate_evolved_population(
        self,
        elite_designs: List[DesignWithReward]
    ) -> List[ToolActionDesign]:
        """Generate evolved population from elite designs."""
        evo_config = self.config.evolution
        
        # Build system prompt for evolution
        system_prompt = EVOLUTION_MISSION_PROMPT + "\n" + PROCEDURE_PROMPT
        
        # Format previous designs for context
        previous = [d.to_dict() for d in elite_designs]
        
        # Generate evolved designs
        designs = self.vlm_client.sample_population(
            task_description=self.env.get_task_description(),
            environment_code=self.env.get_environment_code(),
            frame_description=self.env.get_frame_description(),
            system_prompt=system_prompt,
            tool_spec=TOOL_SPECIFICATION_PROMPT,
            action_spec=self._build_action_spec(),
            scene_image=self._scene_image,
            n_agents=evo_config.n_agents,
            n_tools=evo_config.n_tools,
            n_actions=evo_config.n_actions,
            previous_designs=previous,
            evolution_prompt=EVOLUTION_INSTRUCTION_PROMPT,
        )
        
        return designs
    
    def _evaluate_designs(
        self,
        designs: List[ToolActionDesign],
        iteration: int,
    ) -> List[DesignWithReward]:
        """Evaluate designs in simulation."""
        results = []
        
        for i, design in enumerate(designs):
            if not design.is_valid:
                continue
            
            # Create tool-action pair
            tool_action = ToolActionPair(
                tool_urdf=design.tool_urdf,
                actions=design.actions,
                tool_description=design.tool_description,
                action_description=design.action_description,
            )
            
            # Evaluate
            run_result = self.runner.evaluate_single(
                tool_action,
                design_id=i,
                verbose=self.config.verbose
            )
            
            # Create result
            results.append(DesignWithReward(
                design=design,
                reward=run_result.reward,
                success=run_result.success,
                distance=run_result.distance,
                iteration=iteration,
                design_id=i,
            ))
            
            if self.config.verbose:
                print(f"  Design {i}: reward={run_result.reward:.3f}")
        
        return results
    
    def run(self) -> DesignWithReward:
        """
        Run the evolutionary search.
        
        Returns:
            Best design found
        """
        evo_config = self.config.evolution
        
        print("=" * 60)
        print("VLMgineer Evolutionary Search")
        print("=" * 60)
        print(f"Task: {self.env.__class__.__name__}")
        print(f"Iterations: {evo_config.n_iterations}")
        print(f"Population: {evo_config.n_agents} agents × {evo_config.n_tools} tools × {evo_config.n_actions} actions")
        print("=" * 60)
        
        # Capture scene image
        print("\nCapturing scene image...")
        self._scene_image = self._capture_scene_image()
        
        for iteration in range(evo_config.n_iterations):
            print(f"\n{'='*60}")
            print(f"Iteration {iteration + 1}/{evo_config.n_iterations}")
            print("=" * 60)
            
            # Generate designs
            if iteration == 0:
                print("\nGenerating initial population...")
                designs = self._generate_initial_population()
            else:
                print(f"\nEvolving from {len(self.state.elite_designs)} elite designs...")
                designs = self._generate_evolved_population(self.state.elite_designs)
            
            print(f"Generated {len(designs)} valid designs")
            
            if len(designs) == 0:
                print("Warning: No valid designs generated, skipping iteration")
                continue
            
            # Evaluate
            print("\nEvaluating designs...")
            evaluated = self._evaluate_designs(designs, iteration)
            
            # Update state
            self.state.add_iteration_results(
                evaluated,
                top_k=evo_config.top_k,
                reward_threshold=evo_config.reward_threshold
            )
            
            # Print statistics
            rewards = [d.reward for d in evaluated]
            print(f"\nIteration {iteration + 1} results:")
            print(f"  Designs evaluated: {len(evaluated)}")
            print(f"  Mean reward: {np.mean(rewards):.3f}")
            print(f"  Max reward: {np.max(rewards):.3f}")
            print(f"  Best overall: {self.state.best_reward:.3f}")
            print(f"  Elite designs: {len(self.state.elite_designs)}")
            
            # Early stopping if task solved
            if self.state.best_design and self.state.best_design.success:
                print("\nTask solved! Stopping early.")
                break
        
        print("\n" + "=" * 60)
        print("Evolution Complete")
        print("=" * 60)
        print(f"Best reward: {self.state.best_reward:.3f}")
        if self.state.best_design:
            print(f"Best design: {self.state.best_design.design.tool_description}")
        
        return self.state.best_design
    
    def save_results(self, output_dir: str) -> None:
        """
        Save evolution results to disk.
        
        Args:
            output_dir: Directory to save results
        """
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save best design
        if self.state.best_design:
            best_path = os.path.join(output_dir, f"best_design_{timestamp}.json")
            with open(best_path, 'w') as f:
                json.dump({
                    "reward": float(self.state.best_design.reward),
                    "success": bool(self.state.best_design.success),
                    "tool_urdf": self.state.best_design.design.tool_urdf,
                    "actions": self.state.best_design.design.actions.tolist(),
                    "tool_description": self.state.best_design.design.tool_description,
                }, f, indent=2)
            print(f"Saved best design to {best_path}")
        
        # Save statistics
        stats_path = os.path.join(output_dir, f"evolution_stats_{timestamp}.json")
        with open(stats_path, 'w') as f:
            json.dump({
                "n_iterations": self.state.iteration,
                "best_reward": self.state.best_reward,
                "rewards_per_iteration": self.state.rewards_per_iteration,
                "best_rewards_per_iteration": self.state.best_rewards_per_iteration,
                "total_designs_evaluated": len(self.state.all_designs),
            }, f, indent=2)
        print(f"Saved statistics to {stats_path}")
        
        # Save all designs if configured
        if self.config.save_all_designs:
            all_path = os.path.join(output_dir, f"all_designs_{timestamp}.json")
            all_designs_data = []
            for d in self.state.all_designs:
                all_designs_data.append({
                    "iteration": int(d.iteration),
                    "design_id": int(d.design_id),
                    "reward": float(d.reward),
                    "success": bool(d.success),
                    "tool_urdf": d.design.tool_urdf,
                    "actions": d.design.actions.tolist(),
                })
            with open(all_path, 'w') as f:
                json.dump(all_designs_data, f, indent=2)
            print(f"Saved all designs to {all_path}")
