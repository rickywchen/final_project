"""
Affordance Pipeline with Live Visualization - FAST VERSION
Focus on antipodal grasps with 5x speed
"""

from pathlib import Path
import sys
import time
import numpy as np
import mujoco
import mujoco.viewer

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from affordance_pipeline import AffordancePipeline
from object_loader import ObjectLoader
from grasp_sampler import GraspSampler

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"


def execute_with_viewer(grasp, pipeline, viewer, speed=0.0002):
    """Execute grasp with live visualization - FAST MOTION"""
    
    # Reset
    mujoco.mj_resetData(pipeline.model, pipeline.data)
    pipeline.data.qpos[:7] = pipeline.home_q
    pipeline.data.qpos[7:9] = 0.04
    mujoco.mj_forward(pipeline.model, pipeline.data)
    
    for _ in range(50):  # Reduced from 100
        mujoco.mj_step(pipeline.model, pipeline.data)
        viewer.sync()
        time.sleep(speed)
    
    initial_cube_z = pipeline.data.xpos[pipeline.cube_id][2]
    
    # Compute waypoints
    approach_distance = 0.15
    approach_pos = grasp.position - approach_distance * grasp.approach_direction
    grasp_pos = grasp.position
    lift_pos = grasp.position + np.array([0, 0, 0.20])
    
    from pickup_clean import solve_ik_hybrid, smooth_trajectory
    
    try:
        q_approach = solve_ik_hybrid(
            pipeline.model, pipeline.data, pipeline.hand_id,
            approach_pos, grasp.orientation, pipeline.home_q,
            max_steps=1000
        )
        
        q_grasp = solve_ik_hybrid(
            pipeline.model, pipeline.data, pipeline.hand_id,
            grasp_pos, grasp.orientation, q_approach,
            max_steps=1000
        )
        
        q_lift = solve_ik_hybrid(
            pipeline.model, pipeline.data, pipeline.hand_id,
            lift_pos, grasp.orientation, q_grasp,
            max_steps=1000
        )
    except:
        print("  IK failed!")
        return False
    
    # Phase 1: Approach (FAST - 200 steps)
    print("  → Approaching...")
    for q in smooth_trajectory(pipeline.home_q, q_approach, 200):
        pipeline.data.ctrl[:7] = q
        pipeline.data.ctrl[7] = 255.0
        mujoco.mj_step(pipeline.model, pipeline.data)
        viewer.sync()
        time.sleep(speed)
    
    # Phase 2: Move to grasp (FAST - 150 steps)
    print("  → Grasping...")
    for q in smooth_trajectory(q_approach, q_grasp, 150):
        pipeline.data.ctrl[:7] = q
        pipeline.data.ctrl[7] = 255.0
        mujoco.mj_step(pipeline.model, pipeline.data)
        viewer.sync()
        time.sleep(speed)
    
    # Phase 3: Close gripper (FAST - 300 steps)
    print("  → Closing...")
    for i in range(300):
        t = (i + 1) / 300
        pipeline.data.ctrl[:7] = q_grasp
        pipeline.data.ctrl[7] = (1 - t) * 255.0 + t * (-100.0)
        mujoco.mj_step(pipeline.model, pipeline.data)
        viewer.sync()
        time.sleep(speed)
    
    for _ in range(100):
        pipeline.data.ctrl[:7] = q_grasp
        pipeline.data.ctrl[7] = -100.0
        mujoco.mj_step(pipeline.model, pipeline.data)
        viewer.sync()
        time.sleep(speed)
    
    # Phase 4: Lift (FAST - 200 steps)
    print("  → Lifting...")
    for q in smooth_trajectory(q_grasp, q_lift, 200):
        pipeline.data.ctrl[:7] = q
        pipeline.data.ctrl[7] = -100.0
        mujoco.mj_step(pipeline.model, pipeline.data)
        viewer.sync()
        time.sleep(speed)
    
    for _ in range(100):
        pipeline.data.ctrl[:7] = q_lift
        pipeline.data.ctrl[7] = -100.0
        mujoco.mj_step(pipeline.model, pipeline.data)
        viewer.sync()
        time.sleep(speed)
    
    # Evaluate
    final_cube_z = pipeline.data.xpos[pipeline.cube_id][2]
    lift_distance = final_cube_z - initial_cube_z
    success = lift_distance >= 0.08
    
    result = 'SUCCESS ✓' if success else 'FAILED ✗'
    print(f"  → Lift: {lift_distance*100:.1f}cm - {result}")
    
    time.sleep(0.3)  # Shorter pause (was 0.5)
    
    return success

if __name__ == "__main__":
    print("\n" + "="*70)
    print("AFFORDANCE PIPELINE - FAST MODE (ANTIPODAL FOCUS)")
    print("="*70)
    
    # Load model
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    
    # Create viewer
    print("\nOpening 3D viewer...")
    print("Controls:")
    print("  - Left mouse: Rotate view")
    print("  - Right mouse: Pan view")
    print("  - Scroll: Zoom")
    
    viewer = mujoco.viewer.launch_passive(model, data)
    
    # Create object
    loader = ObjectLoader(ROOT)
    mesh, props = loader.create_simple_cube(size=0.05)
    
    # Sample grasps
    sampler = GraspSampler()
    all_grasps = sampler.sample_grasps(mesh, props, num_samples=40)
    
    # Get cube position
    mujoco.mj_resetData(model, data)
    for _ in range(1000):
        mujoco.mj_step(model, data)
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_pos = data.xpos[cube_id].copy()
    
    # Transform to world
    for g in all_grasps:
        g.position += cube_pos
    
    # Filter
    workspace = {'x': (0.3, 0.9), 'y': (-0.3, 0.6), 'z': (0.4, 0.8)}
    feasible_grasps = sampler.filter_feasible_grasps(all_grasps, workspace_bounds=workspace)
    
    # Separate by strategy
    antipodal_grasps = [g for g in feasible_grasps if g.strategy == 'antipodal']
    top_down_grasps = [g for g in feasible_grasps if g.strategy == 'top_down']
    lateral_grasps = [g for g in feasible_grasps if g.strategy == 'lateral']
    
    print(f"\nAvailable grasps:")
    print(f"  Top-down: {len(top_down_grasps)}")
    print(f"  Antipodal: {len(antipodal_grasps)}")
    print(f"  Lateral: {len(lateral_grasps)}")
    
    # Test: 2 top-down + 10 antipodal
    test_grasps = top_down_grasps[:2] + antipodal_grasps[:10]
    
    print(f"\nTesting {len(test_grasps)} grasps (2 top-down, 10 antipodal)")
    print("="*70 + "\n")
    
    pipeline = AffordancePipeline(model, data)
    
    successes_by_strategy = {'top_down': 0, 'antipodal': 0, 'lateral': 0}
    total_by_strategy = {'top_down': 0, 'antipodal': 0, 'lateral': 0}
    
    for i, grasp in enumerate(test_grasps):
        print(f"[{i+1}/{len(test_grasps)}] {grasp.strategy.upper()} grasp:")
        success = execute_with_viewer(grasp, pipeline, viewer, speed=0.0002)  # Fast!
        
        total_by_strategy[grasp.strategy] += 1
        if success:
            successes_by_strategy[grasp.strategy] += 1
        print()
    
    print("="*70)
    print("RESULTS:")
    print("="*70)
    for strategy in ['top_down', 'antipodal', 'lateral']:
        if total_by_strategy[strategy] > 0:
            rate = 100 * successes_by_strategy[strategy] / total_by_strategy[strategy]
            print(f"{strategy.capitalize()}: {successes_by_strategy[strategy]}/{total_by_strategy[strategy]} ({rate:.0f}%)")
    
    total_success = sum(successes_by_strategy.values())
    total_tested = sum(total_by_strategy.values())
    print(f"\nOverall: {total_success}/{total_tested} ({100*total_success/total_tested:.0f}%)")
    print("="*70)
    
    print("\nPress ENTER to close...")
    input()
    viewer.close()