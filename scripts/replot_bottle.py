"""
Quick re-plot — reloads existing trial JSON and regenerates all heatmap images.
Run this instead of run_bottle_heatmap.py when you only changed empirical_heatmap.py.
Takes ~10 seconds instead of 15 minutes (no physics trials).
"""
import sys
import numpy as np
import mujoco
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from empirical_grasp_sampler import get_object_info
from empirical_heatmap import EmpiricalHeatmap, ObjInfo

ROOT       = Path(__file__).resolve().parent.parent
SCENES_DIR = ROOT / "assets" / "scenes"
OUTPUT_DIR = ROOT / "outputs" / "empirical"

# The actual file gets saved to assets/outputs by run_bottle_heatmap.py
# Check both locations so it works regardless
JSON_PATHS = [
    ROOT / "assets" / "outputs" / "empirical" / "empirical_trials_scene_bottle.json",
    ROOT / "outputs" / "empirical" / "empirical_trials_scene_bottle.json",
]
JSON_PATH = next((p for p in JSON_PATHS if p.exists()), None)

if JSON_PATH is None:
    print("ERROR: Could not find empirical_trials_scene_bottle.json")
    print("  Checked:")
    for p in JSON_PATHS:
        print(f"    {p}")
    print("Run run_bottle_heatmap.py first!")
    sys.exit(1)

print(f"Found trial data at: {JSON_PATH}")

# Load geometry from scene (no simulation needed, just model load)
model = mujoco.MjModel.from_xml_path(str(SCENES_DIR / "scene_bottle.xml"))
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)
for _ in range(300):
    mujoco.mj_step(model, data)

raw = get_object_info(model, data, "cube")

BOTTLE_SECTIONS = [
    # (z_world_bottom, z_world_top, radius)
    (0.420, 0.440, 0.030),   # base
    (0.440, 0.480, 0.030),   # body_lower
    (0.480, 0.520, 0.030),   # slippery_band
    (0.520, 0.550, 0.022),   # shoulder
    (0.550, 0.590, 0.015),   # neck
    (0.590, 0.600, 0.018),   # cap
]

obj = ObjInfo(
    body_name       = "bottle",
    com             = raw.com,
    geom_centre     = raw.geom_centre,
    height          = raw.height,
    width           = raw.width,
    depth           = raw.depth,
    geom_type       = raw.geom_type,
    table_z         = raw.table_z,
    top_z           = raw.top_z,
    bottle_sections = BOTTLE_SECTIONS,
)

# Save outputs to the same folder the JSON came from
out_dir = str(JSON_PATH.parent)
print(f"Saving plots to: {out_dir}")
heatmap = EmpiricalHeatmap.from_json(str(JSON_PATH), obj)
paths   = heatmap.save_all(out_dir, tag="bottle")

print(f"\nDone! Regenerated {len(paths)} files:")
for p in paths:
    print(f"  {p}")