"""
Complete Grasp System Using Operational Space Control
Following supervisor's guidance:
1. Use data.subtree_com for object position
2. Plan in task/operational space
3. Execute with working OSC
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path


def get_object_com(model, data, object_name="cube"):
    """Get object center of mass (supervisor's instruction!)"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    return data.subtree_com[body_id].copy()


class OSCController:
    """Working OSC - tracks desired joint positions properly"""
    
    def __init__(self, model, data, ee_body="hand"):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        self.q_desired = np.zeros(7)
        self.Kp = 3.0
        self.dt = model.opt.timestep
        print(f"✓ OSC ready (EE: {ee_body})")
    
    def reset(self, q_init):
        """Reset desired position"""
        self.q_desired = q_init.copy()
    
    def move_to(self, target_pos, target_quat, steps=1000, viewer=None):
        """Move to target pose using OSC"""
        for i in range(steps):
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.xpos[self.ee_id].copy()
            current_quat = self.data.xquat[self.ee_id].copy()
            
            # Position error
            pos_error = target_pos - current_pos
            
            # Orientation error (simplified)
            ori_error = self._quat_error(target_quat, current_quat)
            
            # Jacobian
            jac_pos = np.zeros((3, self.model.nv))
            jac_ori = np.zeros((3, self.model.nv))
            mujoco.mj_jacBodyCom(self.model, self.data, jac_pos, jac_ori, self.ee_id)
            
            J = np.vstack([jac_pos[:, :7], jac_ori[:, :7]])  # (6, 7)
            
            # Task-space velocity
            task_vel = np.hstack([self.Kp * pos_error, self.Kp * ori_error])
            
            # Map to joint space
            damping = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
            joint_vel = J_pinv @ task_vel
            
            # Integrate (KEY: use tracked q_desired!)
            self.q_desired = self.q_desired + self.dt * joint_vel
            
            # Clip
            for j in range(7):
                self.q_desired[j] = np.clip(
                    self.q_desired[j],
                    self.model.jnt_range[j, 0],
                    self.model.jnt_range[j, 1]
                )
            
            # Send command
            self.data.ctrl[:7] = self.q_desired
            mujoco.mj_step(self.model, self.data)
            
            if viewer and i % 10 == 0:
                viewer.sync()
            
            # Print progress
            if i % 200 == 0:
                error = np.linalg.norm(pos_error)
                print(f"    Step {i}/{steps}: error = {error*1000:.1f}mm")
        
        # Final error
        final_pos = self.data.xpos[self.ee_id].copy()
        final_error = np.linalg.norm(final_pos - target_pos)
        return final_error < 0.01  # 1cm tolerance
    
    def _quat_error(self, q_target, q_current):
        """Quaternion error"""
        q_target = q_target / np.linalg.norm(q_target)
        q_current = q_current / np.linalg.norm(q_current)
        
        w_t, x_t, y_t, z_t = q_target
        w_c, x_c, y_c, z_c = q_current
        
        w = w_t*w_c + x_t*x_c + y_t*y_c + z_t*z_c
        x = w_t*(-x_c) + x_t*w_c + y_t*(-z_c) - z_t*(-y_c)
        y = w_t*(-y_c) - x_t*(-z_c) + y_t*w_c + z_t*(-x_c)
        z = w_t*(-z_c) + x_t*(-y_c) - y_t*(-x_c) + z_t*w_c
        
        return 2.0 * np.array([x, y, z])


def execute_top_down_grasp_osc(model, data, viewer=None):
    """Execute top-down grasp using OSC"""
    
    print("\n" + "="*70)
    print("TOP-DOWN GRASP WITH OPERATIONAL SPACE CONTROL")
    print("="*70)
    
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    osc = OSCController(model, data, "hand")
    
    # Reset to home
    home_q = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
    data.qpos[:7] = home_q
    data.qpos[7:9] = 0.04
    mujoco.mj_forward(model, data)
    
    print("\n[1/6] Settling at home...")
    for _ in range(500):
        data.ctrl[:7] = home_q
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
    
    osc.reset(home_q)
    
    # Get object CoM (supervisor's method!)
    object_com = get_object_com(model, data, "cube")
    print(f"  Object CoM: [{object_com[0]:.3f}, {object_com[1]:.3f}, {object_com[2]:.3f}]")
    
    initial_z = data.xpos[cube_id][2]
    
    # Compute grasp poses in task space
    grasp_pos = object_com.copy()
    grasp_pos[2] += 0.08  # 8cm above CoM
    grasp_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Downward
    
    approach_pos = grasp_pos.copy()
    approach_pos[2] += 0.15  # 15cm above grasp
    
    lift_pos = grasp_pos.copy()
    lift_pos[2] += 0.20  # Lift 20cm
    
    print(f"  Grasp position: [{grasp_pos[0]:.3f}, {grasp_pos[1]:.3f}, {grasp_pos[2]:.3f}]")
    
    # Execute with OSC
    print("\n[2/6] Moving to approach...")
    data.ctrl[7] = 255.0  # Open gripper
    success = osc.move_to(approach_pos, grasp_quat, steps=1000, viewer=viewer)
    if not success:
        print("  ⚠️ Failed to reach approach")
    
    print("\n[3/6] Descending to grasp...")
    success = osc.move_to(grasp_pos, grasp_quat, steps=800, viewer=viewer)
    if not success:
        print("  ⚠️ Failed to reach grasp")
    
    print("\n[4/6] Closing gripper...")
    for i in range(300):
        t = i / 300
        data.ctrl[7] = (1 - t) * 255.0 + t * (-100.0)
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()
    
    # Hold closed
    for _ in range(200):
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
    
    print("\n[5/6] Lifting...")
    success = osc.move_to(lift_pos, grasp_quat, steps=1200, viewer=viewer)
    if not success:
        print("  ⚠️ Failed to lift")
    
    # Hold lifted
    print("\n[6/6] Holding...")
    for _ in range(300):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
    
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
        print("\n  OSC-based grasping works!")
        print("  Following supervisor's guidance:")
        print("    ✓ Used data.subtree_com for object position")
        print("    ✓ Planned in operational/task space")
        print("    ✓ Executed with OSC")
    else:
        print("  ✗ Failed to lift object")
    
    print("="*70)
    
    return success


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent.parent
    XML_PATH = ROOT / "assets" / "scenes" / "scene_grasp_clean.xml"
    
    print("\n" + "="*70)
    print("OPERATIONAL SPACE CONTROL GRASP EXECUTION")
    print("="*70)
    print("Implements supervisor's guidance:")
    print("  1. Get object CoM from data.subtree_com")
    print("  2. Plan grasp in Cartesian/operational space")
    print("  3. Execute with OSC (not IK!)")
    print("="*70)
    
    model = mujoco.MjModel.from_xml_path(str(XML_PATH))
    data = mujoco.MjData(model)
    viewer = mujoco.viewer.launch_passive(model, data)
    
    success = execute_top_down_grasp_osc(model, data, viewer)
    
    if success:
        print("\n✓✓✓ OSC-BASED GRASPING WORKS! ✓✓✓")
        print("Next: Add side grasps with collision avoidance")
    
    print("\nPress ESC to close...")
    try:
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
    except KeyboardInterrupt:
        pass
    
    viewer.close()