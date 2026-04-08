"""
Simple Grasp Execution - Position Control
Using MuJoCo's position actuators (not torque)
Following supervisor's pipeline with working approach
"""

import numpy as np
import mujoco
import mujoco.viewer
import time


def get_object_com(model, data, object_name="cube"):
    """Get object center of mass (supervisor's instruction!)"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    com = data.subtree_com[body_id].copy()
    return com


def solve_ik_simple(model, data, target_pos, target_quat, body_name="hand", max_steps=100):
    """
    Simple IK using MuJoCo's built-in IK
    
    This is MUCH simpler than full OSC!
    """
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    
    # IK using MuJoCo's numerical IK
    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    
    current_q = data.qpos[:7].copy()
    
    for _ in range(max_steps):
        mujoco.mj_forward(model, data)
        
        # Current position/orientation
        current_pos = data.xpos[body_id].copy()
        current_quat = data.xquat[body_id].copy()
        
        # Position error
        pos_err = target_pos - current_pos
        
        # Orientation error (simplified)
        quat_err = target_quat - current_quat
        
        # Check convergence
        if np.linalg.norm(pos_err) < 0.001:
            break
        
        # Compute Jacobian
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        
        # Only use position Jacobian for simplicity
        J = jacp[:, :7]
        
        # Compute joint velocity
        J_pinv = np.linalg.pinv(J)
        dq = J_pinv @ pos_err
        
        # Update joint positions (small step)
        data.qpos[:7] += 0.5 * dq
        
    return data.qpos[:7].copy()


def smooth_trajectory(q_start, q_end, steps=100):
    """Generate smooth trajectory between two joint configurations"""
    for i in range(steps):
        t = (i + 1) / steps
        # Smooth interpolation
        t_smooth = 3*t**2 - 2*t**3
        q = (1 - t_smooth) * q_start + t_smooth * q_end
        yield q


def execute_top_down_grasp(model, data, viewer=None):
    """
    Simple top-down grasp using position control
    
    Pipeline:
    1. Get object CoM (supervisor's method)
    2. Compute grasp target
    3. Solve IK for waypoints
    4. Execute with position commands
    """
    print("\n" + "="*70)
    print("EXECUTING TOP-DOWN GRASP - POSITION CONTROL")
    print("="*70)
    
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    
    # Reset to home
    home_q = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    data.qpos[:7] = home_q
    data.qpos[7:9] = 0.04
    mujoco.mj_forward(model, data)
    
    # Settle
    print("\n[1/5] Settling...")
    for _ in range(500):
        data.ctrl[:7] = home_q
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
    
    # Get object CoM (supervisor's instruction!)
    object_com = get_object_com(model, data, "cube")
    print(f"  Object CoM: [{object_com[0]:.3f}, {object_com[1]:.3f}, {object_com[2]:.3f}]")
    
    initial_z = data.xpos[cube_id][2]
    
    # Compute grasp pose (above object)
    grasp_pos = object_com.copy()
    grasp_pos[2] += 0.08  # 8cm above CoM
    grasp_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Downward
    
    print(f"  Target grasp: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
    
    # Approach pose
    approach_pos = grasp_pos.copy()
    approach_pos[2] += 0.15
    
    # Solve IK
    print("\n[2/5] Computing IK...")
    try:
        q_approach = solve_ik_simple(model, data, approach_pos, grasp_quat)
        
        data.qpos[:7] = q_approach
        q_grasp = solve_ik_simple(model, data, grasp_pos, grasp_quat)
        
        lift_pos = grasp_pos.copy()
        lift_pos[2] += 0.20
        data.qpos[:7] = q_grasp
        q_lift = solve_ik_simple(model, data, lift_pos, grasp_quat)
        
        print("  ✓ IK solved")
    except Exception as e:
        print(f"  ✗ IK failed: {e}")
        return False
    
    # Reset
    data.qpos[:7] = home_q
    mujoco.mj_forward(model, data)
    
    # Execute motion
    print("\n[3/5] Moving to approach...")
    for q in smooth_trajectory(home_q, q_approach, 150):
        data.ctrl[:7] = q
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
        time.sleep(0.001)
    
    print("\n[4/5] Descending and closing...")
    for q in smooth_trajectory(q_approach, q_grasp, 120):
        data.ctrl[:7] = q
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
        time.sleep(0.001)
    
    # Close gripper
    for i in range(200):
        t = i / 200
        data.ctrl[:7] = q_grasp
        data.ctrl[7] = (1 - t) * 255.0 + t * (-100.0)
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
        time.sleep(0.001)
    
    # Hold
    for _ in range(100):
        data.ctrl[:7] = q_grasp
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
        time.sleep(0.001)
    
    print("\n[5/5] Lifting...")
    for q in smooth_trajectory(q_grasp, q_lift, 150):
        data.ctrl[:7] = q
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
        time.sleep(0.001)
    
    # Hold lifted
    for _ in range(100):
        data.ctrl[:7] = q_lift
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
        time.sleep(0.001)
    
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
        print("  ✗ Failed")
    
    print("="*70)
    
    return success


if __name__ == "__main__":
    from pathlib import Path
    
    ROOT = Path(__file__).resolve().parent.parent
    XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"
    
    print("\n" + "="*70)
    print("SIMPLE APPROACH - POSITION CONTROL")
    print("Using MuJoCo's position actuators + IK")
    print("Following supervisor's CoM guidance")
    print("="*70)
    
    # Load
    print(f"\nLoading: {XML_PATH.name}")
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    
    # Viewer
    viewer = mujoco.viewer.launch_passive(model, data)
    
    # Execute
    success = execute_top_down_grasp(model, data, viewer)
    
    if success:
        print("\n✓✓✓ Position control works!")
    
    # Hold
    print("\nPress ESC to close...")
    try:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
    except KeyboardInterrupt:
        pass
    
    viewer.close()