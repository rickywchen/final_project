"""
Affordance Mapping for Robotic Grasping
Tests TOP-DOWN grasp success with different gripper orientations
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import mujoco
import json
from datetime import datetime
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Import your working grasp functions
import sys
sys.path.append(str(Path(__file__).parent))
from pickup_clean import (
    solve_ik_hybrid,
    compute_downward_orientation,
    smooth_trajectory,
    check_contacts,
    quaternion_multiply
)

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"
OUTPUT_DIR = ROOT / "results" / "affordance_maps"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def axis_angle_to_quaternion(axis, angle):
    """Convert axis-angle rotation to quaternion"""
    axis = axis / np.linalg.norm(axis)
    half_angle = angle / 2
    s = np.sin(half_angle)
    
    return np.array([
        np.cos(half_angle),
        axis[0] * s,
        axis[1] * s,
        axis[2] * s
    ])


def sample_grasp_poses(cube_pos, num_rotations=24, 
                       approach_distance=0.15, grasp_distance=0.10):
    """
    Sample TOP-DOWN grasp poses with different gripper rotations
    
    Tests how gripper orientation affects grasp success for symmetric cube
    """
    grasp_samples = []
    
    # Sample gripper rotations around Z-axis
    rotation_angles = np.linspace(0, 2*np.pi, num_rotations, endpoint=False)
    
    for i, rotation in enumerate(rotation_angles):
        # Positions (always from above)
        approach_pos = cube_pos + np.array([0, 0, approach_distance])
        grasp_pos = cube_pos + np.array([0, 0, grasp_distance])
        lift_pos = cube_pos + np.array([0, 0, approach_distance + 0.10])
        
        # Orientation: downward grasp, rotated around Z
        base_quat = compute_downward_orientation()
        z_rotation_quat = axis_angle_to_quaternion(np.array([0, 0, 1]), rotation)
        orientation = quaternion_multiply(z_rotation_quat, base_quat)
        
        grasp_samples.append({
            'id': len(grasp_samples),
            'rotation_deg': np.degrees(rotation),
            'rotation_rad': rotation,
            'approach_pos': approach_pos,
            'grasp_pos': grasp_pos,
            'lift_pos': lift_pos,
            'orientation': orientation
        })
    
    return grasp_samples


def test_single_grasp(model, data, grasp_params, hand_id, cube_id, 
                      cube_geom_id, left_finger_id, right_finger_id,
                      timeout=10.0):
    """Test a single grasp attempt"""
    start_time = time.time()
    
    # Reset
    mujoco.mj_resetData(model, data)
    home_q = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    data.qpos[:7] = home_q
    data.qpos[7:9] = 0.04
    mujoco.mj_forward(model, data)
    data.ctrl[:7] = home_q
    data.ctrl[7] = 255.0
    
    # Settle
    for _ in range(500):
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
    
    initial_cube_z = data.xpos[cube_id][2]
    
    if initial_cube_z < 0.42 or initial_cube_z > 0.47:
        return {
            'success': False,
            'reason': 'cube_not_on_table',
            'lift_distance': 0.0,
            'contact_forces': (0.0, 0.0),
            'both_contacts': False
        }
    
    # IK solve
    try:
        q_approach = solve_ik_hybrid(model, data, hand_id, 
                                    grasp_params['approach_pos'], 
                                    grasp_params['orientation'], 
                                    home_q, max_steps=1000)
        
        q_grasp = solve_ik_hybrid(model, data, hand_id, 
                                 grasp_params['grasp_pos'], 
                                 grasp_params['orientation'], 
                                 q_approach, max_steps=1000)
        
        q_lift = solve_ik_hybrid(model, data, hand_id, 
                                grasp_params['lift_pos'], 
                                grasp_params['orientation'], 
                                q_grasp, max_steps=1000)
    except Exception as e:
        return {
            'success': False,
            'reason': f'ik_failed',
            'lift_distance': 0.0,
            'contact_forces': (0.0, 0.0),
            'both_contacts': False
        }
    
    # Approach
    for q in smooth_trajectory(home_q, q_approach, 1000):
        data.ctrl[:7] = q
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if time.time() - start_time > timeout:
            return {'success': False, 'reason': 'timeout', 'lift_distance': 0.0, 'contact_forces': (0.0, 0.0), 'both_contacts': False}
    
    for _ in range(200):
        data.ctrl[:7] = q_approach
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
    
    # Grasp
    for q in smooth_trajectory(q_approach, q_grasp, 800):
        data.ctrl[:7] = q
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
    
    for _ in range(200):
        data.ctrl[:7] = q_grasp
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
    
    # Close gripper
    for i in range(1000):
        t = (i + 1) / 1000
        data.ctrl[:7] = q_grasp
        data.ctrl[7] = (1 - t) * 255.0 + t * (-100.0)
        mujoco.mj_step(model, data)
    
    for _ in range(500):
        data.ctrl[:7] = q_grasp
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
    
    # Check contacts
    left_ok, right_ok, left_f, right_f = check_contacts(
        model, data, cube_geom_id, left_finger_id, right_finger_id
    )
    
    # Lift
    for q in smooth_trajectory(q_grasp, q_lift, 1200):
        data.ctrl[:7] = q
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
    
    for _ in range(500):
        data.ctrl[:7] = q_lift
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
    
    # Evaluate
    final_cube_z = data.xpos[cube_id][2]
    lift_distance = final_cube_z - initial_cube_z
    success = lift_distance >= 0.08
    
    return {
        'success': bool(success),
        'reason': 'success' if success else 'insufficient_lift',
        'lift_distance': float(lift_distance),
        'contact_forces': (float(left_f), float(right_f)),
        'both_contacts': bool(left_ok and right_ok)
    }


def run_affordance_mapping(model, data, num_rotations=24):
    """Run affordance mapping for gripper rotations"""
    
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    
    # Get cube position
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    cube_pos = data.xpos[cube_id].copy()
    
    print(f"\n{'='*70}")
    print(f"AFFORDANCE MAPPING - Gripper Rotation Study")
    print(f"{'='*70}")
    print(f"Cube position: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    print(f"Testing {num_rotations} gripper orientations (top-down approach)")
    print(f"{'='*70}\n")
    
    grasp_samples = sample_grasp_poses(cube_pos, num_rotations)
    results = []
    
    for i, grasp_params in enumerate(grasp_samples):
        print(f"[{i+1}/{len(grasp_samples)}] Testing rotation: "
              f"{grasp_params['rotation_deg']:.1f}°", end=" ... ")
        
        result = test_single_grasp(
            model, data, grasp_params,
            hand_id, cube_id, cube_geom_id, 
            left_finger_id, right_finger_id
        )
        
        full_result = {**grasp_params, **result}
        results.append(full_result)
        
        status = "✓ SUCCESS" if result['success'] else f"✗ FAILED"
        lift_cm = result['lift_distance'] * 100
        print(f"{status} (lift: {lift_cm:.1f}cm)")
    
    num_success = sum(1 for r in results if r['success'])
    success_rate = 100 * num_success / len(results)
    
    print(f"\n{'='*70}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*70}")
    print(f"Total grasps: {len(results)}")
    print(f"Successful:   {num_success}")
    print(f"Failed:       {len(results) - num_success}")
    print(f"Success rate: {success_rate:.1f}%")
    print(f"{'='*70}\n")
    
    return results


def visualize_affordance_map(results, save_path=None):
    """Visualize rotation-based affordance"""
    
    rotations = np.array([r['rotation_deg'] for r in results])
    successes = np.array([1.0 if r['success'] else 0.0 for r in results])
    lifts = np.array([r['lift_distance'] * 100 for r in results])
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Success/Failure
    colors = ['green' if s else 'red' for s in successes]
    ax1.bar(rotations, successes, width=360/len(results)*0.8, color=colors, edgecolor='black')
    ax1.set_xlabel('Gripper Rotation (degrees)', fontsize=12)
    ax1.set_ylabel('Success (1) / Failure (0)', fontsize=12)
    ax1.set_title('Grasp Success vs Gripper Rotation', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1.2])
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Lift Distance
    ax2.plot(rotations, lifts, 'o-', linewidth=2, markersize=8, color='blue')
    ax2.axhline(y=8.0, color='red', linestyle='--', label='Success threshold (8cm)')
    ax2.set_xlabel('Gripper Rotation (degrees)', fontsize=12)
    ax2.set_ylabel('Lift Distance (cm)', fontsize=12)
    ax2.set_title('Lift Performance vs Gripper Rotation', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✓ Visualization saved: {save_path}")
    
    plt.show()


def save_results(results, output_dir):
    """Save results to JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"affordance_results_{timestamp}.json"
    
    # Convert all numpy types
    results_serializable = []
    for r in results:
        r_copy = {}
        for key, value in r.items():
            if isinstance(value, np.ndarray):
                r_copy[key] = value.tolist()
            elif isinstance(value, (np.bool_, bool)):
                r_copy[key] = bool(value)
            elif isinstance(value, (np.integer, np.floating)):
                r_copy[key] = float(value)
            elif isinstance(value, tuple):
                r_copy[key] = list(value)
            else:
                r_copy[key] = value
        results_serializable.append(r_copy)
    
    with open(output_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'total_grasps': int(len(results)),
            'success_count': int(sum(1 for r in results if r['success'])),
            'results': results_serializable
        }, f, indent=2)
    
    print(f"✓ Results saved: {output_file}")
    return output_file


if __name__ == "__main__":
    if not XML_PATH.exists():
        print(f"ERROR: {XML_PATH} not found")
        exit(1)
    
    print("Loading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    print("✓ Model loaded\n")
    
    # Test gripper rotations (symmetric cube should work for all angles)
    results = run_affordance_mapping(model, data, num_rotations=24)
    
    save_results(results, OUTPUT_DIR)
    
    viz_path = OUTPUT_DIR / f"affordance_map_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    visualize_affordance_map(results, save_path=viz_path)
    
    print("\n✓ Affordance mapping complete!")
    print(f"Results: {OUTPUT_DIR}")