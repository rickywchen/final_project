"""
Grasp Execution Using Operational Space Control
Following supervisor's complete pipeline
"""

import numpy as np
import mujoco
import mujoco.viewer
from simple_osc import SimpleOSC, move_to_pose


def get_object_com(model, data, object_name="cube"):
    """
    Get object center of mass (supervisor's instruction!)
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        object_name: Name of object body
    
    Returns:
        (3,) center of mass position
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    com = data.subtree_com[body_id].copy()
    return com


def compute_top_down_grasp_pose(object_com, offset_z=0.10):
    """
    Compute top-down grasp pose
    
    Args:
        object_com: (3,) object center of mass
        offset_z: Height above object to position gripper
    
    Returns:
        grasp_pos, grasp_quat
    """
    # Position: above object
    grasp_pos = object_com.copy()
    grasp_pos[2] += offset_z
    
    # Orientation: gripper pointing down
    # This is the proven working orientation for Panda
    grasp_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180° around Y
    
    return grasp_pos, grasp_quat


def execute_top_down_grasp(model, data, viewer=None):
    """
    Complete top-down grasp execution
    
    Pipeline (supervisor's guidance):
    1. Get object CoM
    2. Compute grasp target
    3. Use OSC to move gripper
    4. Close gripper
    5. Lift
    
    Args:
        model: MuJoCo model
        data: MuJoCo data
        viewer: Optional viewer
    
    Returns:
        bool: Success
    """
    print("\n" + "="*70)
    print("EXECUTING TOP-DOWN GRASP WITH OSC")
    print("="*70)
    
    # Get object and gripper IDs
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    
    # Initialize OSC controller
    osc = SimpleOSC(model, data, end_effector_name="hand")
    
    # Reset to home
    home_q = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    data.qpos[:7] = home_q
    data.qpos[7:9] = 0.04  # Gripper open
    mujoco.mj_forward(model, data)
    
    # Settle
    print("\n[1/5] Settling...")
    for _ in range(500):
        data.ctrl[:7] = home_q
        data.ctrl[7] = 255.0  # Gripper open
        mujoco.mj_step(model, data)
    
    # Get object CoM (supervisor's method!)
    object_com = get_object_com(model, data, "cube")
    print(f"  Object CoM: [{object_com[0]:.3f}, {object_com[1]:.3f}, {object_com[2]:.3f}]")
    
    initial_z = data.xpos[cube_id][2]
    
    # Compute grasp pose
    grasp_pos, grasp_quat = compute_top_down_grasp_pose(object_com, offset_z=0.10)
    print(f"  Target grasp: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
    
    # Approach pose (15cm above grasp)
    approach_pos = grasp_pos.copy()
    approach_pos[2] += 0.15
    
    # Phase 1: Move to approach
    print("\n[2/5] Moving to approach position...")
    data.ctrl[7] = 255.0  # Keep gripper open
    for i in range(2500):
        torques = osc.compute_control(approach_pos, grasp_quat)
        data.ctrl[:7] = torques
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()
    
    current_pos, _ = osc.get_ee_pose()
    error = np.linalg.norm(current_pos - approach_pos) * 100
    print(f"  Position error: {error:.1f}mm")
    
    # Phase 2: Descend to grasp
    print("\n[3/5] Descending to grasp...")
    for i in range(2000):
        torques = osc.compute_control(grasp_pos, grasp_quat)
        data.ctrl[:7] = torques
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()
    
    current_pos, _ = osc.get_ee_pose()
    error = np.linalg.norm(current_pos - grasp_pos) * 100
    print(f"  Position error: {error:.1f}mm")
    
    # Phase 3: Close gripper
    print("\n[4/5] Closing gripper...")
    for i in range(2000):
        torques = osc.compute_control(grasp_pos, grasp_quat)
        data.ctrl[:7] = torques
        
        # Gradually close
        t = i / 1500
        data.ctrl[7] = (1 - t) * 255.0 + t * (-100.0)
        
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()
    
    # Hold closed
    for _ in range(300):
        torques = osc.compute_control(grasp_pos, grasp_quat)
        data.ctrl[:7] = torques
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
    
    # Phase 4: Lift
    print("\n[5/5] Lifting...")
    lift_pos = grasp_pos.copy()
    lift_pos[2] += 0.20  # Lift 20cm
    
    for i in range(3000):
        torques = osc.compute_control(lift_pos, grasp_quat)
        data.ctrl[:7] = torques
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()
    
    # Hold lifted
    for _ in range(500):
        torques = osc.compute_control(lift_pos, grasp_quat)
        data.ctrl[:7] = torques
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
    
    # Evaluate
    final_z = data.xpos[cube_id][2]
    lift_distance = final_z - initial_z
    success = lift_distance >= 0.08
    
    print("\n" + "="*70)
    print("RESULT:")
    print(f"  Initial Z: {initial_z:.4f}m")
    print(f"  Final Z:   {final_z:.4f}m")
    print(f"  Lift:      {lift_distance*100:.1f}cm")
    
    if success:
        print("  ✓✓✓ SUCCESS! ✓✓✓")
    else:
        print("  ✗ Failed to lift object")
    
    print("="*70)
    
    return success


if __name__ == "__main__":
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parent.parent
    XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"
    
    print("\n" + "="*70)
    print("FRESH START - OPERATIONAL SPACE CONTROL")
    print("Following supervisor's exact guidance:")
    print("  1. Get object CoM from data.subtree_com")
    print("  2. Compute grasp target")
    print("  3. Use OSC for Cartesian control")
    print("="*70)
    
    # Load scene
    print(f"\nLoading: {XML_PATH.name}")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    
    # Launch viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    
    # Execute grasp
    success = execute_top_down_grasp(model, data, viewer)
    
    if success:
        print("\n✓✓✓ Fresh start successful!")
        print("Next: Add antipodal and lateral strategies")
    else:
        print("\nNeed to tune OSC gains")
    
    # Hold viewer
    print("\nPress ESC to close viewer...")
    try:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
    except KeyboardInterrupt:
        pass
    
    viewer.close()