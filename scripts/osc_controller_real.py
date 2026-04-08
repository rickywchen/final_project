"""
FIXED Operational Space Controller
Works properly with MuJoCo's position-controlled actuators

Key insight: Don't fight the built-in PD controller!
Instead, compute smooth position trajectories in task space.
"""

import numpy as np
import mujoco


class OSCFixed:
    """
    OSC that works with position-controlled actuators
    
    Strategy: Compute desired end-effector velocities in task space,
    map to joint velocities, integrate smoothly to positions.
    """
    
    def __init__(self, model, data, end_effector="hand"):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, end_effector)
        
        # Control gains (tune these!)
        self.Kp = 5.0   # Position gain (how fast to approach target)
        self.Kd = 1.0   # Damping (smoothness)
        
        # State
        self.q_target = np.zeros(7)
        
        print(f"✓ Fixed OSC initialized (EE body: {end_effector}, ID: {self.ee_id})")
    
    def compute_joint_positions(self, target_pos, target_quat):
        """
        Compute joint positions to reach target pose
        
        This uses OSC philosophy: control in task space, map to joint space
        
        Returns:
            (7,) joint positions
        """
        # Get current state
        current_pos = self.data.xpos[self.ee_id].copy()
        current_quat = self.data.xquat[self.ee_id].copy()
        
        # Task-space errors
        pos_error = target_pos - current_pos
        ori_error = self._quat_error(target_quat, current_quat)
        
        # Compute Jacobian
        jac_pos = np.zeros((3, self.model.nv))
        jac_ori = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jac_pos, jac_ori, self.ee_id)
        
        J_pos = jac_pos[:, :7]
        J_ori = jac_ori[:, :7]
        J = np.vstack([J_pos, J_ori])
        
        # Current joint velocity
        current_qvel = self.data.qvel[:7].copy()
        current_ee_vel = J @ current_qvel
        
        # Desired task-space velocity (PD control in task space)
        desired_ee_vel = np.hstack([
            self.Kp * pos_error - self.Kd * current_ee_vel[:3],
            self.Kp * ori_error - self.Kd * current_ee_vel[3:]
        ])
        
        # Map to joint space (damped pseudoinverse)
        damping = 0.05
        J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
        desired_qvel = J_pinv @ desired_ee_vel
        
        # Integrate to get target joint positions
        # Use smaller integration step for smoothness
        dt = self.model.opt.timestep
        self.q_target = self.data.qpos[:7] + dt * desired_qvel
        
        # Clip to limits
        for i in range(7):
            self.q_target[i] = np.clip(
                self.q_target[i],
                self.model.jnt_range[i, 0],
                self.model.jnt_range[i, 1]
            )
        
        return self.q_target
    
    def _quat_error(self, q_target, q_current):
        """Quaternion error as axis-angle"""
        q_target = q_target / np.linalg.norm(q_target)
        q_current = q_current / np.linalg.norm(q_current)
        
        # q_error = q_target * q_current^{-1}
        w_t, x_t, y_t, z_t = q_target
        w_c, x_c, y_c, z_c = q_current
        
        w = w_t * w_c + x_t * x_c + y_t * y_c + z_t * z_c
        x = w_t * (-x_c) + x_t * w_c + y_t * (-z_c) - z_t * (-y_c)
        y = w_t * (-y_c) - x_t * (-z_c) + y_t * w_c + z_t * (-x_c)
        z = w_t * (-z_c) + x_t * (-y_c) - y_t * (-x_c) + z_t * w_c
        
        return 2.0 * np.array([x, y, z])
    
    def get_ee_pose(self):
        """Get current end-effector pose"""
        pos = self.data.xpos[self.ee_id].copy()
        quat = self.data.xquat[self.ee_id].copy()
        return pos, quat


def move_to_pose_osc(osc, target_pos, target_quat, data, viewer, steps=2000):
    """
    Move to target pose using OSC
    
    Args:
        osc: OSC controller
        target_pos: Target position
        target_quat: Target orientation
        data: MuJoCo data
        viewer: Viewer
        steps: Number of control steps
    """
    for i in range(steps):
        # Compute joint positions with OSC
        q_cmd = osc.compute_joint_positions(target_pos, target_quat)
        
        # Send to robot (position control)
        data.ctrl[:7] = q_cmd
        
        # Step simulation
        mujoco.mj_step(osc.model, data)
        
        if viewer and i % 5 == 0:
            viewer.sync()
        
        # Print progress
        if i % 400 == 0:
            current_pos, _ = osc.get_ee_pose()
            error = np.linalg.norm(current_pos - target_pos)
            print(f"  Step {i}/{steps}: error = {error*1000:.1f}mm")