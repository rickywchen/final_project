"""
Viewer script - run this separately to visualize
"""
import mujoco
import mujoco.viewer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"

if __name__ == "__main__":
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    
    # Set to interesting pose (after grasp)
    data.qpos[:7] = [0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0]
    data.qpos[7:9] = 0.0  # Gripper closed
    data.qpos[9:12] = [0.6, 0.15, 0.6]  # Cube lifted
    data.qpos[12:16] = [1.0, 0.0, 0.0, 0.0]
    
    mujoco.mj_forward(model, data)
    
    print("Opening viewer (this should work)...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()