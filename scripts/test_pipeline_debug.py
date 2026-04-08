"""
Debug: Test pipeline with known-working grasp from pickup_clean.py
"""

from pathlib import Path
import sys
import numpy as np
import mujoco

_SCRIPT_DIR = Path(__file__).parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from affordance_pipeline import AffordancePipeline
from grasp_sampler import GraspCandidate

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"

# Load model
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

# Get cube position
mujoco.mj_resetData(model, data)
for _ in range(1000):
    mujoco.mj_step(model, data)
cube_pos = data.xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")].copy()

print(f"Cube position: {cube_pos}")
print(f"Cube Z: {cube_pos[2]:.4f}m")

# Create a manual grasp using the EXACT parameters from pickup_clean.py
# that we know works
manual_grasp = GraspCandidate(
    grasp_id=999,
    strategy='manual_test',
    position=cube_pos + np.array([0, 0, 0.10]),  # 10cm above cube (known working)
    approach_direction=np.array([0, 0, -1]),     # From directly above
    orientation=np.array([0.0, 1.0, 0.0, 0.0]), # Downward orientation
    width=0.05,
    quality_score=1.0,
    contact_points=(cube_pos, cube_pos),
    surface_normals=(np.array([0, 0, 1]), np.array([0, 0, 1]))
)

print(f"\nManual grasp position: {manual_grasp.position}")
print(f"Testing known-working grasp configuration...")

# Test it
pipeline = AffordancePipeline(model, data)
result = pipeline.execute_single_grasp(manual_grasp, timeout=15.0)

print("\n" + "="*70)
print("RESULT:")
print("="*70)
print(f"Success: {result['success']}")
print(f"Reason: {result['reason']}")
print(f"Lift: {result['lift_distance']*100:.1f}cm")
print(f"Execution time: {result['execution_time']:.1f}s")
print("="*70)

if result['success']:
    print("\n✓ Pipeline works! Problem is with sampled grasp positions.")
    print("   → Need to adjust grasp sampler height offset")
else:
    print("\n✗ Even known-working grasp fails. Problem is in execution logic.")
    print(f"   Reason: {result['reason']}")
    
    # Additional debugging
    if 'ik_failed' in result['reason']:
        print("\n   Debug: IK solver couldn't find solution")
        print("   → Check if target position is reachable")
    elif result['lift_distance'] < 0.01:
        print("\n   Debug: No lift occurred")
        print("   → Gripper might not be grasping object")
        print("   → Check contact forces or gripper closure")