"""
FINAL OSC-Based Adaptive Grasping System

Back to basics: Use the OSC that WORKED, make it truly adaptive
- Proper height calculations (relative to table, not CoM!)
- Adaptive distances based on object size
- No IK pre-filtering (just try and adapt)
- Works for cube, cylinder, any object
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import time


@dataclass
class ObjectGeometry:
    """Object geometry analysis"""
    name: str
    com: np.ndarray
    height: float
    width: float
    depth: float
    table_z: float  # Table surface height
    
    def is_tall(self) -> bool:
        return self.height > 0.08
    
    def is_graspable(self) -> bool:
        return min(self.width, self.depth) < 0.08


def analyze_object(model, data, object_name="cube") -> ObjectGeometry:
    """Analyze object geometry"""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_name)
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, object_name)
    
    com = data.subtree_com[body_id].copy()
    geom_type = model.geom_type[geom_id]
    size = model.geom_size[geom_id].copy()
    
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        width = size[0] * 2
        depth = size[1] * 2
        height = size[2] * 2
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        width = size[0] * 2
        depth = size[0] * 2
        height = size[1] * 2
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        width = size[0] * 2
        depth = size[0] * 2
        height = size[0] * 2
    else:
        width = depth = height = 0.05
    
    # Find table surface height (object bottom)
    table_z = com[2] - height / 2
    
    return ObjectGeometry(
        name=object_name,
        com=com,
        height=height,
        width=width,
        depth=depth,
        table_z=table_z
    )


@dataclass
class GraspPlan:
    """A grasp plan"""
    approach_pos: np.ndarray
    grasp_pos: np.ndarray
    lift_pos: np.ndarray
    orientation: np.ndarray
    strategy: str


def plan_grasp(obj: ObjectGeometry, strategy="top_down", angle=0.0) -> GraspPlan:
    """
    Plan a single grasp adaptively
    
    All heights computed from TABLE_Z (not CoM!)
    All distances computed from OBJECT SIZE
    """
    
    if strategy == "top_down":
        # Grasp from above
        # Target: just above object top
        grasp_z = obj.table_z + obj.height + 0.03  # 3cm above object top
        approach_z = grasp_z + 0.20  # 20cm above grasp
        lift_z = grasp_z + 0.25  # 25cm lift
        
        return GraspPlan(
            approach_pos=np.array([obj.com[0], obj.com[1], approach_z]),
            grasp_pos=np.array([obj.com[0], obj.com[1], grasp_z]),
            lift_pos=np.array([obj.com[0], obj.com[1], lift_z]),
            orientation=np.array([0.0, 1.0, 0.0, 0.0]),  # Down
            strategy="top_down"
        )
    
    elif strategy == "antipodal":
        # Side grasp
        approach_dir = np.array([np.cos(angle), np.sin(angle), 0.0])
        
        # Distance: object radius + gripper fingers + clearance
        grasp_dist = max(obj.width, obj.depth) / 2 + 0.02 + 0.01
        approach_dist = grasp_dist + 0.15
        
        grasp_xy = obj.com[:2] + grasp_dist * approach_dir[:2]
        approach_xy = obj.com[:2] + approach_dist * approach_dir[:2]
        
        # Height: at object center
        grasp_z = obj.com[2]
        approach_z = grasp_z + 0.15  # 15cm above
        lift_z = grasp_z + 0.25
        
        yaw = angle + np.pi
        pitch = np.pi / 2
        quat = euler_to_quat(0, pitch, yaw)
        
        return GraspPlan(
            approach_pos=np.array([approach_xy[0], approach_xy[1], approach_z]),
            grasp_pos=np.array([grasp_xy[0], grasp_xy[1], grasp_z]),
            lift_pos=np.array([grasp_xy[0], grasp_xy[1], lift_z]),
            orientation=quat,
            strategy=f"antipodal_{int(np.degrees(angle))}deg"
        )
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def euler_to_quat(roll, pitch, yaw):
    """Euler to quaternion"""
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


class OSCController:
    """Operational Space Controller (the one that WORKED!)"""
    
    def __init__(self, model, data, ee_body="hand"):
        self.model = model
        self.data = data
        self.ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        self.q_desired = np.zeros(7)
        self.Kp = 3.0
        self.dt = model.opt.timestep
        print(f"✓ OSC initialized")
    
    def reset(self, q_init):
        self.q_desired = q_init.copy()
    
    def move_to(self, target_pos, target_quat, steps=1000, viewer=None):
        """Move to target using OSC (the version that worked!)"""
        for i in range(steps):
            mujoco.mj_forward(self.model, self.data)
            
            current_pos = self.data.xpos[self.ee_id].copy()
            current_quat = self.data.xquat[self.ee_id].copy()
            
            pos_error = target_pos - current_pos
            ori_error = self._quat_error(target_quat, current_quat)
            
            jac_pos = np.zeros((3, self.model.nv))
            jac_ori = np.zeros((3, self.model.nv))
            mujoco.mj_jacBodyCom(self.model, self.data, jac_pos, jac_ori, self.ee_id)
            
            J = np.vstack([jac_pos[:, :7], jac_ori[:, :7]])
            task_vel = np.hstack([self.Kp * pos_error, self.Kp * ori_error])
            
            damping = 0.01
            J_pinv = J.T @ np.linalg.inv(J @ J.T + damping * np.eye(6))
            joint_vel = J_pinv @ task_vel
            
            # KEY: Integrate from tracked q_desired!
            self.q_desired = self.q_desired + self.dt * joint_vel
            
            for j in range(7):
                self.q_desired[j] = np.clip(
                    self.q_desired[j],
                    self.model.jnt_range[j, 0],
                    self.model.jnt_range[j, 1]
                )
            
            self.data.ctrl[:7] = self.q_desired
            mujoco.mj_step(self.model, self.data)
            
            if viewer and i % 10 == 0:
                viewer.sync()
            
            error = np.linalg.norm(pos_error)
            if error < 0.01:
                break
        
        final_pos = self.data.xpos[self.ee_id].copy()
        final_error = np.linalg.norm(final_pos - target_pos)
        return final_error < 0.02
    
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


def execute_grasp(model, data, osc, grasp: GraspPlan, viewer=None) -> Tuple[bool, float]:
    """Execute grasp with OSC"""
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    initial_z = data.xpos[cube_id][2]
    
    print(f"\n  [{grasp.strategy}]")
    print(f"    Approach: {grasp.approach_pos}")
    print(f"    Grasp: {grasp.grasp_pos}")
    
    # Approach
    print("    → Approaching...")
    data.ctrl[7] = 255.0
    success = osc.move_to(grasp.approach_pos, grasp.orientation, steps=1500, viewer=viewer)
    if not success:
        return False, 0.0
    
    # Grasp
    print("    → Moving to grasp...")
    success = osc.move_to(grasp.grasp_pos, grasp.orientation, steps=1200, viewer=viewer)
    if not success:
        return False, 0.0
    
    # Close gripper
    print("    → Closing...")
    for i in range(400):
        t = i / 400
        data.ctrl[7] = (1 - t) * 255.0 + t * (-100.0)
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0:
            viewer.sync()
    
    for _ in range(300):
        data.ctrl[7] = -100.0
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
    
    # Lift
    print("    → Lifting...")
    osc.move_to(grasp.lift_pos, grasp.orientation, steps=1800, viewer=viewer)
    
    for _ in range(400):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()
    
    final_z = data.xpos[cube_id][2]
    lift_distance = final_z - initial_z
    success = lift_distance >= 0.08
    
    print(f"    → Lift: {lift_distance*100:.1f}cm", "✓" if success else "✗")
    
    return success, lift_distance


def reset_scene(model, data, home_q, viewer):
    """Reset scene"""
    data.qpos[:7] = home_q
    data.qpos[7:9] = 0.04
    
    cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_joint_id = model.body_jntadr[cube_id]
    
    if cube_joint_id >= 0 and model.jnt_type[cube_joint_id] == mujoco.mjtJoint.mjJNT_FREE:
        qpos_addr = model.jnt_qposadr[cube_joint_id]
        data.qpos[qpos_addr:qpos_addr+3] = [0.55, 0.00, 0.50]
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


def test_final_system():
    """Test final adaptive OSC system"""
    ROOT = Path(__file__).resolve().parent.parent
    
    scenes = [
        "scene_grasp_clean.xml",
        "scene_grasp_cylinder.xml",
    ]
    
    for scene_name in scenes:
        XML_PATH = ROOT / "assets" / "scenes" / scene_name
        if not XML_PATH.exists():
            continue
        
        print("\n" + "="*70)
        print(f"ADAPTIVE OSC GRASPING: {scene_name}")
        print("="*70)
        
        model = mujoco.MjModel.from_xml_path(str(XML_PATH))
        data = mujoco.MjData(model)
        viewer = mujoco.viewer.launch_passive(model, data)
        
        osc = OSCController(model, data)
        home_q = np.array([0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0])
        
        reset_scene(model, data, home_q, viewer)
        obj = analyze_object(model, data, "cube")
        
        print(f"\n📦 Object Analysis:")
        print(f"   Size: {obj.width*100:.1f} x {obj.depth*100:.1f} x {obj.height*100:.1f} cm")
        print(f"   CoM: [{obj.com[0]:.3f}, {obj.com[1]:.3f}, {obj.com[2]:.3f}]")
        print(f"   Table Z: {obj.table_z:.3f}")
        print(f"   Tall: {obj.is_tall()}, Graspable: {obj.is_graspable()}")
        
        # Generate grasp strategies
        strategies = [("top_down", 0.0)]
        
        if obj.is_tall() and obj.is_graspable():
            strategies.extend([
                ("antipodal", np.pi),  # From behind
                ("antipodal", np.pi + np.pi/6),
                ("antipodal", np.pi - np.pi/6),
            ])
        
        print(f"\n🎯 Generated {len(strategies)} grasp strategies")
        
        # Try each strategy
        success = False
        for i, (strategy, angle) in enumerate(strategies):
            print(f"\n[Attempt {i+1}/{len(strategies)}]")
            reset_scene(model, data, home_q, viewer)
            osc.reset(home_q)
            
            grasp = plan_grasp(obj, strategy, angle)
            success, lift = execute_grasp(model, data, osc, grasp, viewer)
            
            if success:
                print(f"\n✓✓✓ SUCCESS! Lifted {lift*100:.1f}cm ✓✓✓")
                break
            
            time.sleep(1)
        
        if not success:
            print(f"\n✗ All {len(strategies)} strategies failed")
        
        time.sleep(3)
        viewer.close()
    
    print("\n" + "="*70)
    print("✓ ADAPTIVE OSC TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    test_final_system()