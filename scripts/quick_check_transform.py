from pathlib import Path
import sys
import numpy as np
import mujoco

_SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(_SCRIPT_DIR))

from object_loader import ObjectLoader
from grasp_sampler import GraspSampler

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"

# Load MuJoCo
model = mujoco.MjModel.from_xml_path(str(XML_PATH))
data = mujoco.MjData(model)

# Get actual cube position in sim
mujoco.mj_resetData(model, data)
for _ in range(1000):
    mujoco.mj_step(model, data)
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
cube_world_pos = data.xpos[cube_id].copy()

print(f"Cube world position: {cube_world_pos}")
print(f"Cube Z: {cube_world_pos[2]:.4f}m")
print(f"Table Z: 0.42m")
print(f"Cube height above table: {(cube_world_pos[2] - 0.42)*100:.2f}cm")

# Create object and sample
loader = ObjectLoader(ROOT)
mesh, props = loader.create_simple_cube(size=0.05)
sampler = GraspSampler()

grasps = sampler.sample_grasps(mesh, props, num_samples=10)

print(f"\nGenerated {len(grasps)} grasps")
print(f"First grasp (LOCAL coords):")
print(f"  Position: {grasps[0].position}")
print(f"  Object top (local): {props.bounding_box['max'][2]:.4f}m")

# Transform to world (like pipeline does)
for g in grasps:
    g.position += cube_world_pos

print(f"\nFirst grasp (WORLD coords after transform):")
print(f"  Position: {grasps[0].position}")
print(f"  Height above table: {(grasps[0].position[2] - 0.42)*100:.1f}cm")
print(f"  Height above cube: {(grasps[0].position[2] - cube_world_pos[2])*100:.1f}cm")
print(f"  Should be ~10.5cm above cube (8cm offset + 2.5cm half-height)")

expected_height = cube_world_pos[2] + 0.025 + 0.08  # cube_z + half_cube + offset
print(f"\nExpected grasp height: {expected_height:.4f}m")
print(f"Actual grasp height: {grasps[0].position[2]:.4f}m")
print(f"Difference: {(grasps[0].position[2] - expected_height)*100:.1f}cm")