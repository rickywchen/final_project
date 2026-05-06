from pathlib import Path
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
MODEL_XML = ROOT / "third_party/mujoco_menagerie/franka_emika_panda/panda.xml"

m = mujoco.MjModel.from_xml_path(str(MODEL_XML))
d = mujoco.MjData(m)

with mujoco.viewer.launch_passive(m, d) as viewer:
    while viewer.is_running():
        mujoco.mj_step(m, d)
