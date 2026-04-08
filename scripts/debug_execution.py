"""
Debug why grasps fail during execution
Compare working manual grasp vs failing sampled grasp
"""

from pathlib import Path
import sys
import numpy as np
import mujoco

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from affordance_pipeline import AffordancePipeline
from grasp_sampler import GraspCandidate
from object_loader import ObjectLoader
from grasp_sampler import GraspSampler

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"

model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

# Get cube position
mujoco.mj_resetData(model, data)
for _ in range(1000):
    mujoco.mj_step(model, data)
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
cube_pos = data.xpos[cube_id].copy()

print("="*70)
print("EXECUTION DEBUG: Manual vs Sampled Grasp")
print("="*70)

# Test 1: Manual grasp (KNOWN TO WORK)
print("\n[Test 1] Manual grasp (working reference):")
manual_grasp = GraspCandidate(
    grasp_id=999,
    strategy='manual',
    position=cube_pos + np.array([0, 0, 0.10]),
    approach_direction=np.array([0, 0, -1]),
    orientation=np.array([0.0, 1.0, 0.0, 0.0]),
    width=0.05,
    quality_score=1.0,
    contact_points=(cube_pos, cube_pos),
    surface_normals=(np.array([0, 0, 1]), np.array([0, 0, 1]))
)

pipeline = AffordancePipeline(model, data)
result1 = pipeline.execute_single_grasp(manual_grasp, timeout=15.0)

print(f"  Position: {manual_grasp.position}")
print(f"  Orientation: {manual_grasp.orientation}")
print(f"  Success: {result1['success']}")
print(f"  Lift: {result1['lift_distance']*100:.1f}cm")

# Test 2: Sampled grasp (CURRENTLY FAILING)
print("\n[Test 2] Sampled grasp (should work but doesn't):")
loader = ObjectLoader(ROOT)
mesh, props = loader.create_simple_cube(size=0.05)
sampler = GraspSampler()

grasps = sampler.sample_grasps(mesh, props, num_samples=10)
for g in grasps:
    g.position += cube_pos

sampled_grasp = grasps[0]  # First top-down grasp

result2 = pipeline.execute_single_grasp(sampled_grasp, timeout=15.0)

print(f"  Position: {sampled_grasp.position}")
print(f"  Orientation: {sampled_grasp.orientation}")
print(f"  Success: {result2['success']}")
print(f"  Lift: {result2['lift_distance']*100:.1f}cm")

# Compare
print("\n" + "="*70)
print("COMPARISON:")
print("="*70)
print(f"Position diff: {np.linalg.norm(manual_grasp.position - sampled_grasp.position)*100:.2f}cm")
print(f"Orientation diff:")
print(f"  Manual:  {manual_grasp.orientation}")
print(f"  Sampled: {sampled_grasp.orientation}")

print("\nOrientation quaternions:")
print(f"  Manual points {'DOWN' if manual_grasp.orientation[2] == 0 else 'TILTED'}")
print(f"  Sampled points {'DOWN' if abs(sampled_grasp.orientation[2]) < 0.01 else 'TILTED'}")

if result1['success'] and not result2['success']:
    print("\n❌ ISSUE FOUND: Sampled grasp fails despite similar position!")
    print("   Likely cause: ORIENTATION or IK solver difference")
else:
    print("\n✓ Both work or both fail - different issue")