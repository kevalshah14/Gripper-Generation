"""
Simulation runner for VLMgineer.

Executes tool-action pairs in PyBullet and collects rewards.
"""

import pybullet as p
import pybullet_data
import numpy as np
import os
import tempfile
import shutil
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

from .config import SimulationConfig
from .envs.base_env import BaseEnv, ToolActionPair, EvaluationResult
from .action_generator import interpolate_waypoints, validate_actions


@dataclass
class RunResult:
    """Result from a single simulation run."""
    
    design_id: int
    reward: float
    success: bool
    distance: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "design_id": self.design_id,
            "reward": self.reward,
            "success": self.success,
            "distance": self.distance,
            "error": self.error,
        }


class SimulationRunner:
    """
    Runs tool-action evaluations in PyBullet.
    
    Supports both single and parallel evaluation.
    """
    
    def __init__(
        self,
        env: BaseEnv,
        config: Optional[SimulationConfig] = None,
    ):
        """
        Initialize the simulation runner.
        
        Args:
            env: The task environment to use
            config: Simulation configuration
        """
        self.env = env
        self.config = config or SimulationConfig()
    
    def evaluate_single(
        self,
        tool_action: ToolActionPair,
        design_id: int = 0,
        verbose: bool = False,
        keep_open: bool = False,
    ) -> RunResult:
        """
        Evaluate a single tool-action pair.
        
        Args:
            tool_action: The tool design and actions to evaluate
            design_id: ID for this design
            verbose: Whether to print progress
            keep_open: Whether to keep the simulation open (for GUI viewing)
            
        Returns:
            RunResult with evaluation metrics
        """
        try:
            # Validate actions
            valid, errors = validate_actions(tool_action.actions)
            if not valid:
                return RunResult(
                    design_id=design_id,
                    reward=0.0,
                    success=False,
                    distance=0.0,
                    error=f"Invalid actions: {errors}"
                )
            
            # Connect to physics server
            if not p.isConnected():
                self.env.connect()
            
            # Load robot with tool
            self.env.load_robot(tool_action.tool_urdf)
            
            # Setup environment
            self.env.setup_environment()
            
            # Run evaluation
            result = self._run_simulation(tool_action, verbose)
            
            return RunResult(
                design_id=design_id,
                reward=result.reward,
                success=result.success,
                distance=result.distance_traversed,
            )
            
        except Exception as e:
            return RunResult(
                design_id=design_id,
                reward=0.0,
                success=False,
                distance=0.0,
                error=str(e)
            )
        finally:
            # Disconnect unless keeping open
            if not keep_open:
                self.env.disconnect()
    
    def _run_simulation(
        self,
        tool_action: ToolActionPair,
        verbose: bool = False,
    ) -> EvaluationResult:
        """
        Run the actual simulation.
        
        Args:
            tool_action: Tool and actions to execute
            verbose: Whether to print progress
            
        Returns:
            EvaluationResult with metrics
        """
        # Interpolate waypoints for smooth motion
        interpolated = interpolate_waypoints(
            tool_action.actions,
            steps_between=20
        )
        
        # Track metrics
        trajectory = []
        total_distance = 0.0
        last_pos = None
        
        # Execute interpolated waypoints
        for i, waypoint in enumerate(interpolated):
            position = waypoint[:3]
            orientation_euler = waypoint[3:6]
            
            # Convert to quaternion
            orientation = p.getQuaternionFromEuler(orientation_euler)
            
            # Set end effector pose via IK
            self.env.set_end_effector_pose(position, orientation)
            
            # Step simulation
            for _ in range(5):  # Multiple steps per waypoint
                self.env.step()
            
            # Track trajectory
            current_pos, _ = self.env.get_tool_mount_pose()
            trajectory.append(current_pos.copy())
            
            if last_pos is not None:
                total_distance += np.linalg.norm(current_pos - last_pos)
            last_pos = current_pos.copy()
            
            if verbose and i % 50 == 0:
                print(f"  Step {i}/{len(interpolated)}")
        
        # Let simulation settle
        for _ in range(100):
            self.env.step()
        
        # Compute reward
        reward, success, info = self.env._compute_reward()
        
        return EvaluationResult(
            reward=reward,
            success=success,
            distance_traversed=total_distance,
            final_state=info,
            trajectory=np.array(trajectory)
        )
    
    def evaluate_batch(
        self,
        tool_actions: List[ToolActionPair],
        verbose: bool = False,
    ) -> List[RunResult]:
        """
        Evaluate a batch of tool-action pairs sequentially.
        
        Args:
            tool_actions: List of designs to evaluate
            verbose: Whether to print progress
            
        Returns:
            List of RunResults
        """
        results = []
        
        for i, ta in enumerate(tool_actions):
            if verbose:
                print(f"Evaluating design {i+1}/{len(tool_actions)}...")
            
            result = self.evaluate_single(ta, design_id=i, verbose=verbose)
            results.append(result)
            
            if verbose:
                print(f"  Reward: {result.reward:.3f}, Success: {result.success}")
        
        return results


def evaluate_design_worker(args: Tuple) -> Dict[str, Any]:
    """
    Worker function for parallel evaluation.
    
    Args:
        args: Tuple of (env_class, env_kwargs, tool_urdf, actions, design_id)
        
    Returns:
        Dict with evaluation results
    """
    env_class, env_kwargs, tool_urdf, actions, design_id = args
    
    try:
        # Create environment
        env = env_class(**env_kwargs)
        env.connect()
        
        # Create tool-action pair
        tool_action = ToolActionPair(
            tool_urdf=tool_urdf,
            actions=np.array(actions)
        )
        
        # Load robot with tool
        env.load_robot(tool_action.tool_urdf)
        env.setup_environment()
        
        # Interpolate waypoints
        interpolated = interpolate_waypoints(
            tool_action.actions,
            steps_between=20
        )
        
        # Execute
        total_distance = 0.0
        last_pos = None
        
        for waypoint in interpolated:
            position = waypoint[:3]
            orientation = p.getQuaternionFromEuler(waypoint[3:6])
            env.set_end_effector_pose(position, orientation)
            
            for _ in range(5):
                env.step()
            
            current_pos, _ = env.get_tool_mount_pose()
            if last_pos is not None:
                total_distance += np.linalg.norm(current_pos - last_pos)
            last_pos = current_pos.copy()
        
        # Settle
        for _ in range(100):
            env.step()
        
        # Get reward
        reward, success, info = env._compute_reward()
        
        return {
            "design_id": design_id,
            "reward": reward,
            "success": success,
            "distance": total_distance,
            "error": None
        }
        
    except Exception as e:
        return {
            "design_id": design_id,
            "reward": 0.0,
            "success": False,
            "distance": 0.0,
            "error": str(e)
        }
    finally:
        try:
            env.disconnect()
        except:
            pass


class ParallelRunner:
    """
    Runs evaluations in parallel using multiple processes.
    
    Each process runs its own PyBullet simulation.
    """
    
    def __init__(
        self,
        env_class: type,
        env_kwargs: Dict[str, Any],
        n_workers: int = 4,
    ):
        """
        Initialize the parallel runner.
        
        Args:
            env_class: Class of environment to instantiate
            env_kwargs: Keyword arguments for environment constructor
            n_workers: Number of parallel workers
        """
        self.env_class = env_class
        self.env_kwargs = env_kwargs
        self.n_workers = n_workers
    
    def evaluate_batch(
        self,
        tool_actions: List[ToolActionPair],
        verbose: bool = False,
    ) -> List[RunResult]:
        """
        Evaluate designs in parallel.
        
        Args:
            tool_actions: List of designs to evaluate
            verbose: Whether to print progress
            
        Returns:
            List of RunResults
        """
        # Prepare work items
        work_items = [
            (
                self.env_class,
                self.env_kwargs,
                ta.tool_urdf,
                ta.actions.tolist(),
                i
            )
            for i, ta in enumerate(tool_actions)
        ]
        
        results = []
        
        # Use process pool
        with ProcessPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(evaluate_design_worker, item): item[4]
                for item in work_items
            }
            
            for future in as_completed(futures):
                design_id = futures[future]
                try:
                    result_dict = future.result()
                    results.append(RunResult(**result_dict))
                    
                    if verbose:
                        print(f"Design {design_id}: reward={result_dict['reward']:.3f}")
                        
                except Exception as e:
                    results.append(RunResult(
                        design_id=design_id,
                        reward=0.0,
                        success=False,
                        distance=0.0,
                        error=str(e)
                    ))
        
        # Sort by design_id to maintain order
        results.sort(key=lambda r: r.design_id)
        
        return results
