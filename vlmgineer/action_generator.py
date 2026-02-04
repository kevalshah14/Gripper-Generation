"""
Action generator utilities for VLMgineer.

Provides utilities for creating, validating, and interpolating action sequences.
"""

import numpy as np
from typing import List, Tuple, Optional
from scipy.spatial.transform import Rotation, Slerp


def validate_actions(actions: np.ndarray) -> Tuple[bool, List[str]]:
    """
    Validate an action waypoint array.
    
    Args:
        actions: Nx7 or Nx6 array of waypoints
        
    Returns:
        (valid, errors): Tuple of validity and list of error messages
    """
    errors = []
    
    # Check shape
    if actions.ndim != 2:
        errors.append(f"Actions must be 2D array, got {actions.ndim}D")
        return False, errors
    
    if actions.shape[1] < 6:
        errors.append(f"Actions must have at least 6 columns, got {actions.shape[1]}")
        return False, errors
    
    if actions.shape[0] < 2:
        errors.append(f"Actions must have at least 2 waypoints, got {actions.shape[0]}")
        return False, errors
    
    # Check for NaN/Inf
    if np.any(np.isnan(actions)):
        errors.append("Actions contain NaN values")
    if np.any(np.isinf(actions)):
        errors.append("Actions contain Inf values")
    
    # Check position bounds (reasonable workspace)
    positions = actions[:, :3]
    if np.any(np.abs(positions) > 2.0):
        errors.append("Positions exceed reasonable workspace bounds (>2m)")
    
    # Check orientations are reasonable (within [-2π, 2π])
    orientations = actions[:, 3:6]
    if np.any(np.abs(orientations) > 2 * np.pi):
        errors.append("Orientations exceed [-2π, 2π] range")
    
    return len(errors) == 0, errors


def interpolate_waypoints(
    waypoints: np.ndarray,
    steps_between: int = 50,
    use_slerp: bool = True
) -> np.ndarray:
    """
    Interpolate between waypoints for smooth motion.
    
    Args:
        waypoints: Nx7 array of waypoints [x,y,z,roll,pitch,yaw,gripper]
        steps_between: Number of interpolation steps between waypoints
        use_slerp: Whether to use SLERP for orientation (vs linear)
        
    Returns:
        Interpolated waypoints array
    """
    n_waypoints = len(waypoints)
    if n_waypoints < 2:
        return waypoints
    
    interpolated = []
    
    for i in range(n_waypoints - 1):
        start = waypoints[i]
        end = waypoints[i + 1]
        
        for t in np.linspace(0, 1, steps_between, endpoint=False):
            # Linear interpolation for position
            pos = start[:3] + t * (end[:3] - start[:3])
            
            # Orientation interpolation
            if use_slerp:
                # Convert euler to quaternion for SLERP
                r_start = Rotation.from_euler('xyz', start[3:6])
                r_end = Rotation.from_euler('xyz', end[3:6])
                
                # Create SLERP interpolator
                key_times = [0, 1]
                key_rots = Rotation.concatenate([r_start, r_end])
                slerp = Slerp(key_times, key_rots)
                
                # Interpolate
                r_interp = slerp(t)
                orn = r_interp.as_euler('xyz')
            else:
                # Linear interpolation
                orn = start[3:6] + t * (end[3:6] - start[3:6])
            
            # Gripper state (step function - use start value until halfway)
            gripper = start[6] if len(start) > 6 else 0
            if t >= 0.5 and len(end) > 6:
                gripper = end[6]
            
            interpolated.append(np.concatenate([pos, orn, [gripper]]))
    
    # Add final waypoint
    interpolated.append(waypoints[-1])
    
    return np.array(interpolated)


def create_approach_waypoints(
    target_pos: np.ndarray,
    target_orn: np.ndarray = None,
    approach_height: float = 0.15,
    start_height: float = 0.4,
) -> np.ndarray:
    """
    Create waypoints for approaching a target position.
    
    Args:
        target_pos: Target position [x, y, z]
        target_orn: Target orientation [roll, pitch, yaw] or None for default
        approach_height: Height above target to approach from
        start_height: Starting height
        
    Returns:
        Nx7 array of approach waypoints
    """
    if target_orn is None:
        target_orn = np.array([0, 0, 0])
    
    waypoints = [
        # Start position (above target)
        [target_pos[0], target_pos[1], start_height, 0, 0, 0, 0],
        # Move to approach position
        [target_pos[0], target_pos[1], target_pos[2] + approach_height, 
         target_orn[0], target_orn[1], target_orn[2], 0],
        # Lower to target
        [target_pos[0], target_pos[1], target_pos[2],
         target_orn[0], target_orn[1], target_orn[2], 0],
    ]
    
    return np.array(waypoints)


def create_push_waypoints(
    start_pos: np.ndarray,
    end_pos: np.ndarray,
    push_height: float = 0.1,
    approach_height: float = 0.2,
    tool_pitch: float = 0.3,
) -> np.ndarray:
    """
    Create waypoints for a pushing action.
    
    Args:
        start_pos: Starting position of push [x, y, z]
        end_pos: Ending position of push [x, y, z]
        push_height: Height during push
        approach_height: Height for approach/retreat
        tool_pitch: Tool pitch angle during push (radians)
        
    Returns:
        Nx7 array of push waypoints
    """
    waypoints = [
        # Start above
        [start_pos[0] - 0.1, start_pos[1], approach_height, 0, tool_pitch, 0, 0],
        # Lower to push height
        [start_pos[0], start_pos[1], push_height, 0, tool_pitch, 0, 0],
        # Push to end
        [end_pos[0], end_pos[1], push_height, 0, tool_pitch, 0, 0],
        # Lift up
        [end_pos[0], end_pos[1], approach_height, 0, 0, 0, 0],
    ]
    
    return np.array(waypoints)


def create_scoop_waypoints(
    target_pos: np.ndarray,
    scoop_direction: np.ndarray = None,
    scoop_depth: float = 0.02,
    lift_height: float = 0.3,
) -> np.ndarray:
    """
    Create waypoints for a scooping action.
    
    Args:
        target_pos: Position of object to scoop
        scoop_direction: Direction to scoop from (normalized) or None for default
        scoop_depth: How deep to scoop below object
        lift_height: Height to lift after scooping
        
    Returns:
        Nx7 array of scoop waypoints
    """
    if scoop_direction is None:
        scoop_direction = np.array([-1, 0, 0])  # Scoop from behind
    
    scoop_direction = scoop_direction / np.linalg.norm(scoop_direction)
    approach_offset = scoop_direction * 0.1
    
    waypoints = [
        # Start position
        [target_pos[0] + approach_offset[0], 
         target_pos[1] + approach_offset[1], 
         0.3, 0, 0.5, 0, 0],
        # Lower and approach
        [target_pos[0] + approach_offset[0] * 0.5,
         target_pos[1] + approach_offset[1] * 0.5,
         target_pos[2] - scoop_depth, 0, 0.5, 0, 0],
        # Scoop under
        [target_pos[0] - approach_offset[0] * 0.5,
         target_pos[1] - approach_offset[1] * 0.5,
         target_pos[2], 0, 0.3, 0, 0],
        # Lift
        [target_pos[0], target_pos[1], lift_height, 0, 0, 0, 0],
    ]
    
    return np.array(waypoints)


def compute_trajectory_length(waypoints: np.ndarray) -> float:
    """
    Compute the total length of a trajectory.
    
    Args:
        waypoints: Nx7 array of waypoints
        
    Returns:
        Total distance traveled (positions only)
    """
    if len(waypoints) < 2:
        return 0.0
    
    positions = waypoints[:, :3]
    distances = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    return float(np.sum(distances))
