"""
run_bottle_heatmap.py
Standalone script: run the empirical affordance trial loop on the bottle
scene and immediately generate the heatmap visualisation.

Usage:
    cd scripts
    python run_bottle_heatmap.py

Outputs → outputs/empirical/
    trial_outcomes_scene_bottle.json
    empirical_2d_bottle.png
    empirical_3d_bottle.png
    empirical_mesh_bottle.png
    empirical_report_bottle.txt
"""

import sys
import json
import numpy as np
from pathlib import Path

# Make scripts/ importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from empirical_grasp_sampler import run_trials, get_object_info, ObjInfo
from empirical_heatmap import EmpiricalHeatmap, TrialResult, ObjInfo as HeatmapObjInfo

ROOT       = Path(__file__).resolve().parent.parent
SCENES_DIR = ROOT / "assets" / "scenes"
OUTPUT_DIR = ROOT / "outputs" / "empirical"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

XML_PATH = str(SCENES_DIR / "scene_bottle.xml")
OBJ_NAME = "cube"   # bottle body is named "cube" in the XML
OBJ_POS  = [0.55, 0.00, 0.47]   # correct Z: bottom flush with table

# ── Step 1: Run physical trials ───────────────────────────────────────────────
print("=" * 60)
print("  BOTTLE AFFORDANCE TRIAL RUN")
print("=" * 60)
print(f"  Scene:   {XML_PATH}")
print(f"  Output:  {OUTPUT_DIR}")
print()

results = run_trials(
    xml_path       = XML_PATH,
    obj_name       = OBJ_NAME,
    n_top          = 20,          # 20 top-down contact samples
    n_side_heights = 10,          # 10 height bands on the side
    n_side_angles  = 24,          # 24 angles = every 15 degrees
    show_viewer    = True,        # set False to run headless
    verbose        = True,
    rng_seed       = 42,
)

# ── Step 2: Load geometry for heatmap ─────────────────────────────────────────
import mujoco
model = mujoco.MjModel.from_xml_path(XML_PATH)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)
for _ in range(300):
    mujoco.mj_step(model, data)

raw = get_object_info(model, data, OBJ_NAME)

# Convert ObjInfo from sampler to heatmap's ObjInfo
obj = HeatmapObjInfo(
    body_name   = "bottle",
    com         = raw.com,
    geom_centre = raw.geom_centre,
    height      = raw.height,
    width       = raw.width,
    depth       = raw.depth,
    geom_type   = raw.geom_type,
    table_z     = raw.table_z,
    top_z       = raw.top_z,
)

# ── Step 3: Build and save empirical heatmap ──────────────────────────────────
heatmap = EmpiricalHeatmap(results, obj)
paths   = heatmap.save_all(str(OUTPUT_DIR), tag="bottle")

print(f"\n  Saved {len(paths)} output files:")
for p in paths:
    print(f"    {p}")

print("\n  Done! Open the PNG files in outputs/empirical/ to see the affordance map.")
print("""
  What to expect in the bottle affordance map:
    GREEN  (high lift)  → base section (dark blue in XML) — heavy, grippy
    YELLOW (partial)    → body lower section — normal friction
    RED    (fail/slip)  → slippery band (orange in XML) — µ=0.15
    RED    (fail/tip)   → neck/cap — CoM is far below, object tips sideways
    ASYMMETRY in XY     → real CoM is +Y offset → +Y side grasps succeed more
""")