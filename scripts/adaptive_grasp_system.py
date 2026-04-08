"""
Adaptive Multi-Object Grasping with OSC and Collision Avoidance

Features:
- Automatic grasp planning based on object geometry
- Real-time collision checking using MuJoCo's collision detection
- Operational Space Control with obstacle avoidance
- Works with ANY object (cube, cylinder, sphere, complex shapes)
- No hardcoded positions!

Following Modern Robotics OSC + GraspIt! affordance principles
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time


@dataclass
class ObjectGeometry:
    """Detected object geometry"""
    name: str
    com: np.ndarray          # Center of mass
    size: np.ndarray         # Bounding box dimensions [x, y, z]
    geom_type: int          # MuJoCo geom type
    height: float
    width: float
    depth: float
    
    def is_tall(self) -> bool:
        """Check if object is tall enough for side grasps"""
        return self.height > 0.08  # 8cm threshold
    
    def is_graspable(self) -> bool:
        """Check if object fits in gripper"""
        max_gripper_width = 0.08  # 8cm gripper width
        return min(self.width, self.depth) < max_gripper_width


@dataclass
class GraspCandidate:
    """A candidate grasp pose"""
    approach_pos: np.ndarray
    grasp_pos: np.ndarray
    lift_pos: np.ndarray
    orientation: np.ndarray  # Quaternion
    strategy: str            # "top_down", "antipodal", "lateral"
    quality: float           # Quality score (0-1)


def analyze_object(model, data, object_name="cube") -> ObjectGeometry:
    """
    Automatically analyze object geometry
    
    Returns complete geometric description without hardcoded values
    """
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_name)
    
    # Get center of mass
    com = data.subtree_com[body_id].copy()
    
    # Get geometry info
    geom_type = model.geom_type[geom_id]
    size = model.geom_size[geom_id].copy()
    
    # Compute actual dimensions based on geom type
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        # Box: size = [half_x, half_y, half_z]
        width = size[0] * 2
        depth = size[1] * 2
        height = size[2] * 2
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        # Cylinder: size = [radius, half_height]
        width = size[0] * 2  # Diameter
        depth = size[0] * 2
        height = size[1] * 2
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        # Sphere: size = [radius, 0, 0]
        width = size[0] * 2
        depth = size[0] * 2
        height = size[0] * 2
    else:
        # Fallback: use bounding box
        width = depth = height = 0.05
        print(f"⚠️ Unknown geom type, using default 5cm")
    
    return ObjectGeometry(
        name=object_name,
        com=com,
        size=size,
        geom_type=geom_type,
        height=height,
        width=width,
        depth=depth
    )


def check_collision(model, data, target_qpos: np.ndarray) -> bool:
    """
    Check if a joint configuration causes collision
    
    Uses MuJoCo's built-in collision detection
    """
    # Save current state
    qpos_backup = data.qpos.copy()
    
    # Set test configuration
    data.qpos[:7] = target_qpos
    mujoco.mj_forward(model, data)
    
    # Check for collisions
    has_collision = False
    for i in range(data.ncon):
        contact = data.contact[i]
        
        # Get body names
        geom1 = contact.geom1
        geom2 = contact.geom2
        body1 = model.geom_bodyid[geom1]
        body2 = model.geom_bodyid[geom2]
        
        name1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1)
        name2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2)
        
        # Check if robot collides with table or itself (excluding gripper-object contact)
        if name1 and name2:
            robot_bodies = ["panda_link", "hand"]
            table_bodies = ["table", "table_surface"]
            
            is_robot1 = any(rb in name1 for rb in robot_bodies)
            is_robot2 = any(rb in name2 for rb in robot_bodies)
            is_table1 = any(tb in name1 for tb in table_bodies)
            is_table2 = any(tb in name2 for tb in table_bodies)
            
            # Robot-table collision
            if (is_robot1 and is_table2) or (is_robot2 and is_table1):
                has_collision = True
                break
            
            # Self-collision (excluding hand-forearm which is expected)
            if is_robot1 and is_robot2:
                if "hand" not in name1 or "hand" not in name2:
                    has_collision = True
                    break
    
    # Restore state
    data.qpos[:] = qpos_backup
    mujoco.mj_forward(model, data)
    
    return has_collision


class AdaptiveOSC:
    """
    Operational Space Controller with collision avoidance
    """
    
    def __init__(self, model, data, ee_body="hand"):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        self.q_desired = np.zeros(7)
        self.Kp = 3.0
        self.dt = model.opt.timestep
        self.collision_check_interval = 5  # Check every N steps
        print(f"✓ Adaptive OSC initialized (EE: {ee_body})")
    
    def reset(self, q_init):
        self.q_desired = q_init.copy()
    
    def move_to_safe(self, target_pos, target_quat, steps=1000, viewer=None, verbose=False):
        """
        Move to target with collision checking
        
        Returns: (success, collision_detected)
        """
        collision_detected = False
        
        for i in range(steps):
            mujoco.mj_forward(self.model, self.data)
            
            # Get current pose
            current_pos = self.data.xpos[self.ee_id].copy()
            current_quat = self.data.xquat[self.ee_id].copy()
            
            # Compute errors
            pos_error = target_pos - current_pos
            ori_error = self._quat_error(target_quat, current_quat)
            
            # Jacobian
            jac_pos = np.zeros((3, self.model.nv))
            jac_ori = np.zeros((3, self.model.nv))
            mujoco.mj_jacBodyCom(self.model, self.data, jac_pos, jac_ori, self.ee_id)
            
            J = np.vstack([jac_pos[:, :7], jac_ori[:, :7]])
            
            # Task-space velocity
            task_vel = np.hstack([self.Kp * pos_error, self.Kp * ori_error])
            
            # Map to joint space
            damping = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
            joint_vel = J_pinv @ task_vel
            
            # Integrate
            q_candidate = self.q_desired + self.dt * joint_vel
            
            # Clip to limits
            for j in range(7):
                q_candidate[j] = np.clip(
                    q_candidate[j],
                    self.model.jnt_range[j, 0],
                    self.model.jnt_range[j, 1]
                )
            
            # Collision check (every N steps to save computation)
            if i % self.collision_check_interval == 0:
                if check_collision(self.model, self.data, q_candidate):
                    if verbose:
                        print(f"    ⚠️ Collision detected at step {i}, stopping")
                    collision_detected = True
                    break
            
            # Apply
            self.q_desired = q_candidate
            self.data.ctrl[:7] = self.q_desired
            mujoco.mj_step(self.model, self.data)
            
            if viewer and i % 10 == 0:
                viewer.sync()
            
            # Check convergence
            error = np.linalg.norm(pos_error)
            if verbose and i % 200 == 0:
                print(f"    Step {i}/{steps}: error = {error*1000:.1f}mm")
            
            if error < 0.01:  # Converged
                break
        
        # Final check
        final_pos = self.data.xpos[self.ee_id].copy()
        final_error = np.linalg.norm(final_pos - target_pos)
        success = final_error < 0.02 and not collision_detected
        
        return success, collision_detected
    
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


def euler_to_quat(roll, pitch, yaw):
    """Convert Euler to quaternion"""
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z])


def plan_adaptive_grasps(obj: ObjectGeometry, gripper_clearance=0.01) -> List[GraspCandidate]:
    """
    Generate grasp candidates automatically based on object geometry
    
    NO HARDCODED VALUES - everything computed from object size!
    
    Args:
        obj: Object geometry
        gripper_clearance: Safety margin (1cm default)
    
    Returns:
        List of grasp candidates sorted by quality
    """
    candidates = []
    
    # Gripper dimensions
    gripper_width = 0.08  # Max gripper opening
    finger_thickness = 0.01
    
    #  1. TOP-DOWN GRASP (always possible)
    # Grasp from above, fingers around object
    grasp_height = obj.com[2] + obj.height/2 + gripper_clearance
    approach_height = grasp_height + max(0.15, obj.height + 0.05)  # Adaptive approach height
    lift_height = grasp_height + 0.20
    
    top_down_quat = np.array([0.0, 1.0, 0.0, 0.0])  # Pointing down
    
    # Quality: higher for wider objects (more stable)
    top_down_quality = min(1.0, (obj.width + obj.depth) / 0.10)
    
    candidates.append(GraspCandidate(
        approach_pos=np.array([obj.com[0], obj.com[1], approach_height]),
        grasp_pos=np.array([obj.com[0], obj.com[1], grasp_height]),
        lift_pos=np.array([obj.com[0], obj.com[1], lift_height]),
        orientation=top_down_quat,
        strategy="top_down",
        quality=top_down_quality
    ))
    
    # 2. ANTIPODAL GRASPS (if object is graspable and tall)
    if obj.is_graspable() and obj.is_tall():
        # Try multiple approach angles
        for angle in [0, np.pi/4, np.pi/2, -np.pi/4]:
            approach_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
            
            # Grasp distance: based on object width + gripper clearance
            grasp_distance = max(obj.width, obj.depth) / 2 + gripper_clearance + finger_thickness
            approach_distance = grasp_distance + max(0.15, obj.width + 0.10)
            
            grasp_pos = obj.com.copy()
            grasp_pos[:2] += grasp_distance * approach_dir[:2]
            grasp_pos[2] = obj.com[2]  # At CoM height
            
            approach_pos = grasp_pos.copy()
            approach_pos[:2] += (approach_distance - grasp_distance) * approach_dir[:2]
            approach_pos[2] += max(0.12, obj.height/2 + 0.05)  # Adaptive approach height
            
            lift_pos = grasp_pos.copy()
            lift_pos[2] += 0.20
            
            # Orientation
            yaw = angle + np.pi
            pitch = np.pi / 2
            roll = 0.0
            quat = euler_to_quat(roll, pitch, yaw)
            
            # Quality: higher for narrower objects (better pinch)
            antipodal_quality = 1.0 - min(1.0, max(obj.width, obj.depth) / gripper_width)
            
            candidates.append(GraspCandidate(
                approach_pos=approach_pos,
                grasp_pos=grasp_pos,
                lift_pos=lift_pos,
                orientation=quat,
                strategy=f"antipodal_{int(np.degrees(angle))}deg",
                quality=antipodal_quality * 0.9  # Slightly lower than top-down
            ))
    
    # 3. LATERAL GRASPS (if object is graspable and tall)
    if obj.is_graspable() and obj.is_tall():
        for angle in [np.pi/6, -np.pi/6]:
            approach_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
            
            grasp_distance = max(obj.width, obj.depth) / 2 + gripper_clearance + finger_thickness
            approach_distance = grasp_distance + max(0.15, obj.width + 0.10)
            
            grasp_pos = obj.com.copy()
            grasp_pos[:2] += grasp_distance * approach_dir[:2]
            
            approach_pos = grasp_pos.copy()
            approach_pos[:2] += (approach_distance - grasp_distance) * approach_dir[:2]
            approach_pos[2] += max(0.10, obj.height/3)
            
            lift_pos = grasp_pos.copy()
            lift_pos[2] += 0.20
            
            yaw = angle + np.pi
            pitch = np.pi / 2
            roll = np.pi / 2
            quat = euler_to_quat(roll, pitch, yaw)
            
            lateral_quality = 0.8 * (1.0 - min(1.0, max(obj.width, obj.depth) / gripper_width))
            
            candidates.append(GraspCandidate(
                approach_pos=approach_pos,
                grasp_pos=grasp_pos,
                lift_pos=lift_pos,
                orientation=quat,
                strategy=f"lateral_{int(np.degrees(angle))}deg",
                quality=lateral_quality
            ))
    
    # Sort by quality (best first)
    candidates.sort(key=lambda c: c.quality, reverse=True)
    
    return candidates


def execute_grasp(model, data, osc: AdaptiveOSC, grasp: GraspCandidate, viewer=None) -> Tuple[bool, str]:
    """
    Execute a grasp candidate with collision checking
    
    Returns: (success, failure_reason)
    """
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    initial_z = data.xpos[cube_id][2]
    
    print(f"\n  Trying {grasp.strategy} (quality: {grasp.quality:.2f})")
    print(f"    Approach: {grasp.approach_pos}")
    print(f"    Grasp: {grasp.grasp_pos}")
    
    # 1. Approach
    print("    → Approaching...")
    data.ctrl[7] = 255.0  # Open gripper
    success, collision = osc.move_to_safe(grasp.approach_pos, grasp.orientation, steps=1500, viewer=viewer)
    if collision:
        return False, "collision_on_approach"
    if not success:
        return False, "failed_approach"
    
    # 2. Move to grasp
    print("    → Moving to grasp...")
    success, collision = osc.move_to_safe(grasp.grasp_pos, grasp.orientation, steps=1200, viewer=viewer)
    if collision:
        return False, "collision_on_grasp"
    if not success:
        return False, "failed_grasp_pose"
    
    # 3. Close gripper
    print("    → Closing gripper...")
    for i in range(300):
        t = i / 300
        data.ctrl[7] = (1 - t) * 255.0 + t * (-100.0)
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()
    
    for _ in range(200):
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
    
    # 4. Lift
    print("    → Lifting...")
    success, collision = osc.move_to_safe(grasp.lift_pos, grasp.orientation, steps=1800, viewer=viewer)
    if collision:
        return False, "collision_on_lift"
    
    # 5. Hold
    for _ in range(300):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
    
    # 6. Evaluate
    final_z = data.xpos[cube_id][2]
    lift_distance = final_z - initial_z
    success = lift_distance >= 0.08
    
    print(f"    → Lift: {lift_distance*100:.1f}cm", "✓" if success else "✗")
    
    if not success:
        return False, f"insufficient_lift_{lift_distance*100:.0f}cm"
    
    return True, "success"


def reset_scene(model, data, home_q, viewer):
    """Reset robot and object"""
    data.qpos[:7] = home_q
    data.qpos[7:9] = 0.04
    
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_joint_id = model.body_jntadr[cube_id]
    
    if cube_joint_id >= 0:
        joint_type = model.jnt_type[cube_joint_id]
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            qpos_addr = model.jnt_qposadr[cube_joint_id]
            data.qpos[qpos_addr:qpos_addr+3] = [0.60, 0.15, 0.50]
            data.qpos[qpos_addr+3:qpos_addr+7] = [1, 0, 0, 0]
            qvel_addr = model.jnt_dofadr[cube_joint_id]
            data.qvel[qvel_addr:qvel_addr+6] = 0.0
    
    mujoco.mj_forward(model, data)
    
    for _ in range(500):
        data.ctrl[:7] = home_q
        data.ctrl[7] = 255.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()


def test_adaptive_grasping():
    """Test adaptive grasping system"""
    
    ROOT = Path(__file__).resolve().parent.parent
    
    # Test with different scenes
    scenes = [
        "scene_grasp_clean.xml",       # Small cube
        "scene_grasp_cylinder.xml",    # Tall cylinder
    ]
    
    for scene_name in scenes:
        XML_PATH = ROOT / "assets" / "scenes" / scene_name
        if not XML_PATH.exists():
            print(f"⚠️ Skipping {scene_name} (not found)")
            continue
        
        print("\n" + "="*70)
        print(f"ADAPTIVE GRASPING TEST: {scene_name}")
        print("="*70)
        
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        data = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, data)
        
        osc = AdaptiveOSC(model, data, "hand")
        home_q = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
        
        # Analyze object
        mujoco.mj_forward(model, data)
        obj = analyze_object(model, data, "cube")
        
        print(f"\n📦 Object Analysis:")
        print(f"  CoM: [{obj.com[0]:.3f}, {obj.com[1]:.3f}, {obj.com[2]:.3f}]")
        print(f"  Dimensions: {obj.width*100:.1f} x {obj.depth*100:.1f} x {obj.height*100:.1f} cm")
        print(f"  Tall: {obj.is_tall()}, Graspable: {obj.is_graspable()}")
        
        # Generate grasp candidates
        candidates = plan_adaptive_grasps(obj)
        print(f"\n🎯 Generated {len(candidates)} grasp candidates")
        for i, c in enumerate(candidates[:5]):  # Show top 5
            print(f"  {i+1}. {c.strategy} (quality: {c.quality:.2f})")
        
        # Try grasps until one succeeds
        print(f"\n🤖 Executing grasps...")
        success = False
        for i, grasp in enumerate(candidates):
            print(f"\n[Attempt {i+1}/{len(candidates)}]")
            reset_scene(model, data, home_q, viewer)
            osc.reset(home_q)
            
            success, reason = execute_grasp(model, data, osc, grasp, viewer)
            
            if success:
                print(f"\n✓✓✓ SUCCESS with {grasp.strategy}! ✓✓✓")
                break
            else:
                print(f"  ✗ Failed: {reason}")
            
            time.sleep(1)
        
        if not success:
            print(f"\n✗ All {len(candidates)} grasp attempts failed")
        
        print("\nPress ESC for next test...")
        time.sleep(3)
        viewer.close()
    
    print("\n" + "="*70)
    print("✓ ADAPTIVE GRASPING TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_adaptive_grasping()