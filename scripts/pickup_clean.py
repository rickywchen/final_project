"""
Working MuJoCo Panda pickup - RESEARCH VERSION
Hybrid IK: Position priority + soft orientation guidance
"""

from __future__ import annotations

from pathlib import Path
import time
import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parent.parent
XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"


def quaternion_multiply(q1, q2):
    """Multiply two quaternions (w, x, y, z)"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])


def quaternion_error(q_target, q_current):
    """Compute orientation error"""
    q_target = q_target / np.linalg.norm(q_target)
    q_current = q_current / np.linalg.norm(q_current)
    
    q_current_conj = np.array([q_current[0], -q_current[1], -q_current[2], -q_current[3]])
    q_error = quaternion_multiply(q_target, q_current_conj)
    
    return 2.0 * q_error[1:4]


def solve_ik_hybrid(model, data, body_id, target_pos, target_quat=None, initial_q=None, max_steps=2000):
    """
    Hybrid IK: Strong position tracking + weak orientation guidance
    This ensures we reach the target position while gently guiding orientation
    """
    qpos_save = data.qpos.copy()
    
    if initial_q is None:
        q = data.qpos[:7].copy()
    else:
        q = initial_q.copy()
    
    alpha = 0.5
    pos_tol = 5e-4
    ori_tol = 0.1  # Relaxed orientation tolerance
    
    best_q = q.copy()
    best_error = float('inf')
    
    # If no target orientation given, use current orientation
    use_orientation = target_quat is not None
    
    for iteration in range(max_steps):
        data.qpos[:7] = q
        mujoco.mj_kinematics(model, data)
        
        # Position error
        current_pos = data.xpos[body_id].copy()
        pos_error = target_pos - current_pos
        pos_error_norm = np.linalg.norm(pos_error)
        
        # Orientation error (optional)
        ori_error = np.zeros(3)
        ori_error_norm = 0.0
        if use_orientation:
            current_quat = data.xquat[body_id].copy()
            ori_error = quaternion_error(target_quat, current_quat)
            ori_error_norm = np.linalg.norm(ori_error)
        
        # Track best
        total_error = pos_error_norm + 0.05 * ori_error_norm
        if total_error < best_error:
            best_error = total_error
            best_q = q.copy()
        
        # Check convergence (position is critical, orientation is optional)
        if pos_error_norm < pos_tol and (not use_orientation or ori_error_norm < ori_tol):
            break
        
        # Jacobians
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
        
        J_pos = jacp[:, :7]
        J_ori = jacr[:, :7]
        
        if use_orientation:
            # Position gets 10x more weight than orientation
            J = np.vstack([10.0 * J_pos, J_ori])
            error = np.hstack([10.0 * pos_error, ori_error])
        else:
            J = J_pos
            error = pos_error
        
        # Damped least squares
        damping = 0.01
        JtJ = J.T @ J + damping * np.eye(7)
        dq = alpha * np.linalg.solve(JtJ, J.T @ error)
        
        # Update
        q += dq
        q = np.clip(q, model.jnt_range[:7, 0], model.jnt_range[:7, 1])
        
        # Adaptive step size
        if iteration > max_steps // 2:
            alpha = 0.3
    
    # Restore
    data.qpos[:] = qpos_save
    mujoco.mj_forward(model, data)
    
    return best_q


def compute_downward_orientation():
    """
    Quaternion for downward grasp
    Tries common Panda orientations
    """
    # 180° around X-axis (fingers pointing down)
    return np.array([0.0, 1.0, 0.0, 0.0])


def compute_grasp_waypoints(cube_pos, approach_height=0.15, grasp_height=0.10, lift_height=0.20):
    """
    Grasp waypoints with optimal height
    
    grasp_height tuning:
      - 0.11m: Fingers 102mm away (too far) ← YOUR CURRENT PROBLEM
      - 0.08m: Fingers ~80mm away (still far)
      - 0.075m: Fingers ~60mm away (good range)
      - 0.06m: Fingers ~40mm away (risky: might hit table)
    """
    approach_pos = cube_pos + np.array([0, 0, approach_height])
    grasp_pos = cube_pos + np.array([0, 0, grasp_height])
    lift_pos = grasp_pos + np.array([0, 0, lift_height])
    
    return approach_pos, grasp_pos, lift_pos


def smooth_trajectory(start_q, end_q, steps):
    """Smooth cubic interpolation"""
    t = np.linspace(0, 1, steps)
    t_smooth = t * t * (3 - 2 * t)
    
    return [(1 - ts) * start_q + ts * end_q for ts in t_smooth]


def check_contacts(model, data, cube_geom_id, left_finger_id, right_finger_id):
    """Check finger-cube contacts"""
    left_force = 0.0
    right_force = 0.0
    
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        
        if g1 != cube_geom_id and g2 != cube_geom_id:
            continue
        
        other_geom = g2 if g1 == cube_geom_id else g1
        other_body = int(model.geom_bodyid[other_geom])
        
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)
        force_mag = np.linalg.norm(force[:3])
        
        if other_body == left_finger_id:
            left_force = max(left_force, force_mag)
        elif other_body == right_finger_id:
            right_force = max(right_force, force_mag)
    
    threshold = 0.5
    return left_force > threshold, right_force > threshold, left_force, right_force


def execute_research_grasp(model, data, visualize=True):
    """
    Research-grade grasp with hybrid IK
    """
    
    # Get IDs
    hand_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cube_geom")
    left_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_finger")
    right_finger_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_finger")
    
    # Viewer
    viewer = None
    if visualize:
        try:
            viewer = mujoco.viewer.launch_passive(model, data)
            print("✓ Viewer launched")
        except Exception as e:
            print(f"⚠ Viewer error: {e}")
            viewer = None
    
    def update_viewer():
        if viewer:
            try:
                if viewer.is_running():
                    viewer.sync()
                    time.sleep(0.0001)
            except:
                pass
    
    # Reset
    mujoco.mj_resetData(model, data)
    
    # Home
    home_q = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    
    data.qpos[:7] = home_q
    data.qpos[7:9] = 0.04
    mujoco.mj_forward(model, data)
    
    data.ctrl[:7] = home_q
    data.ctrl[7] = 255.0
    
    print("\n" + "="*70)
    print("RESEARCH GRASP - HYBRID IK (Position Priority + Orientation Guidance)")
    print("="*70)
    
    # Settle
    print("\n[0/5] Settling...")
    for i in range(2000):
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            update_viewer()
    
    # Cube
    cube_pos = data.xpos[cube_id].copy()
    initial_cube_z = cube_pos[2]
    
    print(f"  Cube: [{cube_pos[0]:.3f}, {cube_pos[1]:.3f}, {cube_pos[2]:.3f}]")
    print(f"  Z: {initial_cube_z:.4f}m")
    
    if initial_cube_z < 0.42 or initial_cube_z > 0.47:
        print(f"  ✗ Cube not on table!")
        return False
    print(f"  ✓ Cube on table")
    
    # Waypoints
    approach_pos, grasp_pos, lift_pos = compute_grasp_waypoints(cube_pos)
    target_quat = compute_downward_orientation()
    
    print(f"\n  Waypoints:")
    print(f"    Approach: [{approach_pos[0]:.3f}, {approach_pos[1]:.3f}, {approach_pos[2]:.3f}]")
    print(f"    Grasp:    [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
    print(f"    Lift:     [{lift_pos[0]:.3f}, {lift_pos[1]:.3f}, {lift_pos[2]:.3f}]")
    print(f"  Target orientation: {target_quat}")
    
    # IK with orientation guidance
    print(f"\n  Solving IK (hybrid mode)...")
    q_approach = solve_ik_hybrid(model, data, hand_id, approach_pos, target_quat, home_q)
    q_grasp = solve_ik_hybrid(model, data, hand_id, grasp_pos, target_quat, q_approach)
    q_lift = solve_ik_hybrid(model, data, hand_id, lift_pos, target_quat, q_grasp)
    print(f"  ✓ Done")
    
    # Phase 1: Approach
    print(f"\n[1/5] Approach...")
    for i, q in enumerate(smooth_trajectory(home_q, q_approach, 2500)):
        data.ctrl[:7] = q
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            update_viewer()
    
    for i in range(500):
        data.ctrl[:7] = q_approach
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            update_viewer()
    
    reached = data.xpos[hand_id].copy()
    error = np.linalg.norm(reached - approach_pos)
    print(f"  Position error: {error*1000:.1f}mm")
    
    # Phase 2: Grasp
    print(f"\n[2/5] Descend to grasp...")
    for i, q in enumerate(smooth_trajectory(q_approach, q_grasp, 2000)):
        data.ctrl[:7] = q
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            update_viewer()
    
    for i in range(500):
        data.ctrl[:7] = q_grasp
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            update_viewer()
    
    reached = data.xpos[hand_id].copy()
    error = np.linalg.norm(reached - grasp_pos)
    dist_to_cube = np.linalg.norm(reached - cube_pos)
    reached_quat = data.xquat[hand_id].copy()
    
    print(f"  Position error: {error*1000:.1f}mm")
    print(f"  Cube distance: {dist_to_cube*1000:.1f}mm")
    print(f"  Hand orientation: [{reached_quat[0]:.2f}, {reached_quat[1]:.2f}, {reached_quat[2]:.2f}, {reached_quat[3]:.2f}]")
    
    # Phase 3: Close
    print(f"\n[3/5] Closing gripper...")
    for i in range(2000):
        t = (i + 1) / 2000
        data.ctrl[:7] = q_grasp
        data.ctrl[7] = (1 - t) * 255.0 + t * (-80.0)
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            update_viewer()
    
    for i in range(1000):
        data.ctrl[:7] = q_grasp
        data.ctrl[7] = -80.0
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            update_viewer()
    
    left_ok, right_ok, left_f, right_f = check_contacts(
        model, data, cube_geom_id, left_finger_id, right_finger_id
    )
    print(f"  Contacts:")
    print(f"    Left:  {'✓' if left_ok else '✗'} ({left_f:.2f}N)")
    print(f"    Right: {'✓' if right_ok else '✗'} ({right_f:.2f}N)")
    
    # Phase 4: Lift
    print(f"\n[4/5] Lifting...")
    for i, q in enumerate(smooth_trajectory(q_grasp, q_lift, 3000)):
        data.ctrl[:7] = q
        data.ctrl[7] = -80.0
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            update_viewer()
    
    for i in range(1500):
        data.ctrl[:7] = q_lift
        data.ctrl[7] = -80.0
        mujoco.mj_step(model, data)
        if i % 10 == 0:
            update_viewer()
    
    # Evaluate
    print(f"\n[5/5] Evaluating...")
    final_z = data.xpos[cube_id][2]
    lift = final_z - initial_cube_z
    
    print(f"\n" + "="*70)
    print(f"RESULTS")
    print(f"="*70)
    print(f"  Initial Z: {initial_cube_z:.4f}m")
    print(f"  Final Z:   {final_z:.4f}m")
    print(f"  Lift:      {lift:.4f}m ({lift*100:.1f}cm)")
    print(f"  Required:  0.0800m (8.0cm)")
    
    success = lift >= 0.08
    
    if success:
        print(f"\n  ✓✓✓ GRASP SUCCESSFUL ✓✓✓")
    else:
        print(f"\n  ✗ Failed")
        print(f"\n  Diagnostics:")
        print(f"    - Cube distance: {dist_to_cube*1000:.1f}mm (target: <55mm)")
        print(f"    - Contacts: L={left_f:.2f}N, R={right_f:.2f}N")
        if dist_to_cube > 0.06:
            print(f"    → Try: grasp_height=0.03 (currently 0.035)")
        if not (left_ok and right_ok):
            print(f"    → Fingers not contacting cube")
            print(f"    → Check gripper orientation in viewer")
    
    print(f"="*70 + "\n")
    
    # Hold
    if visualize and viewer:
        print("Holding for 10 seconds... (ESC to close)")
        try:
            for i in range(1000):
                if viewer and not viewer.is_running():
                    break
                data.ctrl[:7] = q_lift
                data.ctrl[7] = -200.0
                mujoco.mj_step(model, data)
                if i % 10 == 0:
                    update_viewer()
                time.sleep(0.01)
        except KeyboardInterrupt:
            pass
    
    if viewer:
        try:
            viewer.close()
        except:
            pass
    
    return success


if __name__ == "__main__":
    if not XML_PATH.exists():
        print(f"ERROR: {XML_PATH} not found")
        exit(1)
    
    print("\n" + "="*70)
    print("MUJOCO PANDA GRASPING - RESEARCH VERSION v2")
    print("Hybrid IK: Position-priority with orientation guidance")
    print("="*70)
    
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    
    success = execute_research_grasp(model, data, visualize=True)
    
    if success:
        print("\n✓✓✓ SUCCESS! Ready for affordance mapping!")
        print("\nNext: Loop this grasp over different approach angles")
    else:
        print("\n✗ Grasp failed")
        print("\nIf cube distance is still >60mm, the issue is likely:")
        print("  1. IK can't reach that low with current arm config")
        print("  2. Table is blocking the arm")
        print("\nTry moving cube or table in scene_grasp_clean.xml")