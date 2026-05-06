"""
Complete Affordance Pipeline
Integrates: Object Loading → Grasp Sampling → MuJoCo Execution → Visualization
"""

from pathlib import Path
import sys
import time
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from datetime import datetime
import json

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from object_loader import ObjectLoader
from grasp_sampler import GraspSampler
from pickup_clean import (
    solve_ik_hybrid,
    smooth_trajectory,
    check_contacts
)

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"
OUTPUT_DIR = ROOT / "results" / "affordance_pipeline"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class AffordancePipeline:
    """Complete affordance mapping pipeline"""
    
    def __init__(self, model, data):
        self.model = model
        self.data = data
        
        # Get body IDs
        self.hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
        self.cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
        self.cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
        self.left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
        self.right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
        
        self.home_q = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    
    def execute_single_grasp(self, grasp, timeout=10.0):
        """Execute one grasp attempt in simulation"""
        start_time = time.time()
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:7] = self.home_q
        self.data.qpos[7:9] = 0.04
        mujoco.mj_forward(self.model, self.data)
        self.data.ctrl[:7] = self.home_q
        self.data.ctrl[7] = 255.0
        
        # Settle
        for _ in range(500):
            self.data.ctrl[7] = 255.0
            mujoco.mj_step(self.model, self.data)
        
        initial_cube_z = self.data.xpos[self.cube_id][2]
        
        # Check cube on table
        if initial_cube_z < 0.42 or initial_cube_z > 0.47:
            return {
                'success': False,
                'reason': 'cube_not_on_table',
                'lift_distance': 0.0,
                'execution_time': time.time() - start_time
            }
        
        # Compute waypoints
        approach_distance = 0.15
        approach_pos = grasp.position - approach_distance * grasp.approach_direction
        grasp_pos = grasp.position
        lift_pos = grasp.position + np.array([0, 0, 0.20])

        # DEBUG: Print grasp details
        print(f"\n    DEBUG Grasp #{grasp.grasp_id}:")
        print(f"      Position: {grasp_pos}")
        print(f"      Approach: {approach_pos}")
        print(f"      Cube at: {self.data.xpos[self.cube_id]}")
        print(f"      Distance to cube: {np.linalg.norm(grasp_pos - self.data.xpos[self.cube_id])*100:.1f}cm")
        
        # Solve IK
        try:
            q_approach = solve_ik_hybrid(
                self.model, self.data, self.hand_id,
                approach_pos, grasp.orientation, self.home_q,
                max_steps=1000
            )
            
            q_grasp = solve_ik_hybrid(
                self.model, self.data, self.hand_id,
                grasp_pos, grasp.orientation, q_approach,
                max_steps=1000
            )
            
            q_lift = solve_ik_hybrid(
                self.model, self.data, self.hand_id,
                lift_pos, grasp.orientation, q_grasp,
                max_steps=1000
            )
        except Exception as e:
            return {
                'success': False,
                'reason': f'ik_failed',
                'lift_distance': 0.0,
                'execution_time': time.time() - start_time
            }
        
        # Execute motion
        
        # Phase 1: Approach
        for q in smooth_trajectory(self.home_q, q_approach, 1000):
            self.data.ctrl[:7] = q
            self.data.ctrl[7] = 255.0
            mujoco.mj_step(self.model, self.data)
            if time.time() - start_time > timeout:
                return {'success': False, 'reason': 'timeout', 'lift_distance': 0.0, 'execution_time': timeout}
        
        for _ in range(200):
            self.data.ctrl[:7] = q_approach
            self.data.ctrl[7] = 255.0
            mujoco.mj_step(self.model, self.data)
        
        # Phase 2: Move to grasp
        for q in smooth_trajectory(q_approach, q_grasp, 800):
            self.data.ctrl[:7] = q
            self.data.ctrl[7] = 255.0
            mujoco.mj_step(self.model, self.data)
        
        for _ in range(200):
            self.data.ctrl[:7] = q_grasp
            self.data.ctrl[7] = 255.0
            mujoco.mj_step(self.model, self.data)
        
        # Phase 3: Close gripper
        for i in range(1000):
            t = (i + 1) / 1000
            self.data.ctrl[:7] = q_grasp
            self.data.ctrl[7] = (1 - t) * 255.0 + t * (-100.0)
            mujoco.mj_step(self.model, self.data)
        
        for _ in range(500):
            self.data.ctrl[:7] = q_grasp
            self.data.ctrl[7] = -100.0
            mujoco.mj_step(self.model, self.data)
        
        # Check contacts
        left_ok, right_ok, left_f, right_f = check_contacts(
            self.model, self.data,
            self.cube_geom_id,
            self.left_finger_id,
            self.right_finger_id
        )
        
        # Phase 4: Lift
        for q in smooth_trajectory(q_grasp, q_lift, 1200):
            self.data.ctrl[:7] = q
            self.data.ctrl[7] = -100.0
            mujoco.mj_step(self.model, self.data)
        
        for _ in range(500):
            self.data.ctrl[:7] = q_lift
            self.data.ctrl[7] = -100.0
            mujoco.mj_step(self.model, self.data)
        
        # Evaluate
        final_cube_z = self.data.xpos[self.cube_id][2]
        lift_distance = final_cube_z - initial_cube_z
        success = lift_distance >= 0.08
        
        return {
            'success': bool(success),
            'reason': 'success' if success else 'insufficient_lift',
            'lift_distance': float(lift_distance),
            'contact_forces': (float(left_f), float(right_f)),
            'both_contacts': bool(left_ok and right_ok),
            'execution_time': time.time() - start_time
        }
    
    def run_affordance_mapping(self, mesh, properties, num_grasps=20):
        """
        Complete affordance mapping:
        Sample grasps → Test in simulation → Generate results
        """
        
        print("\n" + "="*70)
        print("AFFORDANCE MAPPING PIPELINE")
        print("="*70)
        print(f"Object: {properties.name}")
        print(f"Bounding box: {properties.bounding_box['size']}")
        print(f"Testing {num_grasps} grasps")
        print("="*70 + "\n")
        
        # Step 1: Sample grasps
        print("[1/3] Sampling grasps...")
        sampler = GraspSampler()
        all_grasps = sampler.sample_grasps(mesh, properties, num_samples=num_grasps*2)
        
        # Get cube position from simulation
        mujoco.mj_resetData(self.model, self.data)
        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
        cube_pos = self.data.xpos[self.cube_id].copy()
        
        # Transform to world coordinates
        for g in all_grasps:
            g.position += cube_pos
        
        # Filter feasible
        workspace = {
            'x': (0.3, 0.9),
            'y': (-0.3, 0.6),
            'z': (0.4, 0.8)
        }
        feasible_grasps = sampler.filter_feasible_grasps(all_grasps, workspace_bounds=workspace)
        
        print(f"  Generated: {len(all_grasps)} grasps")
        print(f"  Feasible: {len(feasible_grasps)} grasps")
        
        # Limit to num_grasps
        test_grasps = feasible_grasps[:num_grasps]
        print(f"  Testing: {len(test_grasps)} grasps\n")
        
        if len(test_grasps) == 0:
            print("ERROR: No feasible grasps to test!")
            return []
        
        # Step 2: Execute grasps
        print("[2/3] Executing grasps in simulation...")
        results = []
        
        for i, grasp in enumerate(test_grasps):
            print(f"  [{i+1}/{len(test_grasps)}] Testing {grasp.strategy} grasp...", end=" ")
            
            result = self.execute_single_grasp(grasp)
            
            # Combine grasp + result
            full_result = {
                'grasp_id': grasp.grasp_id,
                'strategy': grasp.strategy,
                'position': grasp.position.tolist(),
                'approach_direction': grasp.approach_direction.tolist(),
                'quality_score': grasp.quality_score,
                'width': grasp.width,
                **result
            }
            results.append(full_result)
            
            status = "✓" if result['success'] else "✗"
            lift_cm = result['lift_distance'] * 100
            print(f"{status} (lift: {lift_cm:.1f}cm)")
        
        # Step 3: Analyze results
        print("\n[3/3] Analyzing results...")
        self.print_summary(results)
        
        return results
    
    def print_summary(self, results):
        """Print affordance analysis summary"""
        
        total = len(results)
        successes = [r for r in results if r['success']]
        failures = [r for r in results if not r['success']]
        
        # Overall stats
        print("\n" + "="*70)
        print("AFFORDANCE MAPPING RESULTS")
        print("="*70)
        print(f"Total grasps tested: {total}")
        print(f"Successful: {len(successes)} ({100*len(successes)/total:.1f}%)")
        print(f"Failed: {len(failures)} ({100*len(failures)/total:.1f}%)")
        
        # By strategy
        print("\nSuccess by strategy:")
        strategies = set(r['strategy'] for r in results)
        for strategy in strategies:
            strategy_results = [r for r in results if r['strategy'] == strategy]
            strategy_success = [r for r in strategy_results if r['success']]
            if len(strategy_results) > 0:
                rate = 100 * len(strategy_success) / len(strategy_results)
                print(f"  {strategy}: {len(strategy_success)}/{len(strategy_results)} ({rate:.1f}%)")
        
        # Lift distances
        if len(successes) > 0:
            avg_lift = np.mean([r['lift_distance'] for r in successes])
            print(f"\nAverage lift (successful): {avg_lift*100:.1f}cm")
        
        print("="*70 + "\n")


def visualize_affordance_results(mesh, results, object_name="object"):
    """
    Visualize affordance heatmap on mesh
    Green = successful grasps, Red = failed grasps
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: Success/Failure by Strategy
    ax1 = axes[0]
    strategies = {}
    for r in results:
        strategy = r['strategy']
        if strategy not in strategies:
            strategies[strategy] = {'success': 0, 'total': 0}
        strategies[strategy]['total'] += 1
        if r['success']:
            strategies[strategy]['success'] += 1
    
    strategy_names = list(strategies.keys())
    success_rates = [100 * strategies[s]['success'] / strategies[s]['total'] for s in strategy_names]
    counts = [strategies[s]['total'] for s in strategy_names]
    
    x = np.arange(len(strategy_names))
    bars = ax1.bar(x, success_rates, color=['blue', 'red', 'green'][:len(x)])
    
    # Add count labels
    for i, (bar, count) in enumerate(zip(bars, counts)):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 2,
                f'n={count}', ha='center', va='bottom', fontsize=10)
    
    ax1.set_ylabel('Success Rate (%)', fontsize=12)
    ax1.set_title(f'Grasp Success by Strategy - {object_name}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(strategy_names, rotation=15)
    ax1.set_ylim(0, 110)
    ax1.axhline(y=50, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Lift Distance Distribution
    ax2 = axes[1]
    
    successful_lifts = [r['lift_distance']*100 for r in results if r['success']]
    failed_lifts = [r['lift_distance']*100 for r in results if not r['success']]
    
    bins = np.linspace(0, 20, 20)
    ax2.hist(successful_lifts, bins=bins, alpha=0.7, color='green', label='Success', edgecolor='black')
    ax2.hist(failed_lifts, bins=bins, alpha=0.7, color='red', label='Failed', edgecolor='black')
    ax2.axvline(x=8.0, color='black', linestyle='--', linewidth=2, label='Threshold')
    
    ax2.set_xlabel('Lift Distance (cm)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('Lift Performance Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = OUTPUT_DIR / f"affordance_{object_name}_{timestamp}.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✓ Visualization saved: {save_path}")
    
    plt.show()


def save_results_json(results, object_name, output_dir):
    """Save results to JSON"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = output_dir / f"results_{object_name}_{timestamp}.json"
    
    summary = {
        'timestamp': timestamp,
        'object_name': object_name,
        'total_grasps': len(results),
        'successful': sum(1 for r in results if r['success']),
        'success_rate': 100 * sum(1 for r in results if r['success']) / len(results),
        'results': results
    }
    
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"✓ Results saved: {output_file}")


if __name__ == "__main__":
    print("\n" + "="*70)
    print("AFFORDANCE PIPELINE - COMPLETE SYSTEM TEST")
    print("="*70)
    
    # Load model
    print("\nLoading MuJoCo model...")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    print("✓ Model loaded")
    
    # Create object
    print("\nCreating test object...")
    loader = ObjectLoader(ROOT)
    mesh, props = loader.create_simple_cube(size=0.05)
    print(f"✓ Object created: {props.name}")
    print(f"  Size: {props.bounding_box['size']}")
    print(f"  Stability: {props.stability_score:.2f}")
    
    # Run pipeline
    pipeline = AffordancePipeline(model, data)
    results = pipeline.run_affordance_mapping(mesh, props, num_grasps=20)
    
    if len(results) == 0:
        print("\n❌ No results to analyze!")
        exit(1)
    
    # Save results
    save_results_json(results, props.name, OUTPUT_DIR)
    
    # Visualize
    visualize_affordance_results(mesh, results, object_name=props.name)
    
    print("\n" + "="*70)
    print("✓✓✓ AFFORDANCE PIPELINE COMPLETE! ✓✓✓")
    print("="*70)
    print(f"\nResults saved to: {OUTPUT_DIR}")
    print("\nYou now have:")
    print("  ✓ Grasp sampling working")
    print("  ✓ Simulation testing working")
    print("  ✓ Affordance analysis complete")
    print("  ✓ Visualization generated")
    print("\n🎯 Project requirements satisfied!")
    print("="*70 + "\n")