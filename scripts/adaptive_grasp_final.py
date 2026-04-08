"""
Adaptive Multi-Object Grasping System — v12
Franka Panda + MuJoCo 3.x

v12: Shifted CoM scenario — naive vs affordance-guided comparison.
     Naive planner grasps geometric centre (ignores CoM).
     Affordance planner grasps heatmap best contact (CoM-aware).
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional
import time
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))


# ─────────────────────────────────────────────────────────────────────────────
# PANDA GEOMETRY CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────

HAND_TO_FINGERTIP_Z = 0.103
MAX_HAND_Z          = 0.72


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ObjectGeometry:
    name: str
    com: np.ndarray
    geom_centre: np.ndarray
    geom_type: int
    size_raw: np.ndarray
    height: float
    width: float
    depth: float
    table_z: float
    top_z: float

    def is_tall(self) -> bool:
        return self.height > 0.06

    def is_graspable_laterally(self) -> bool:
        return min(self.width, self.depth) < 0.075

    def radius_xy(self) -> float:
        return min(self.width, self.depth) / 2.0

    def com_offset(self) -> float:
        return float(np.linalg.norm(self.com[:2] - self.geom_centre[:2]))

    def __str__(self) -> str:
        gtype = {2: "sphere", 5: "cylinder", 6: "box"}.get(
            self.geom_type, f"type{self.geom_type}")
        off = self.com_offset()
        return (f"ObjectGeometry({self.name}, {gtype}, "
                f"{self.width*100:.1f}×{self.depth*100:.1f}×{self.height*100:.1f} cm, "
                f"geom_centre_z={self.geom_centre[2]:.3f}, "
                f"com_z={self.com[2]:.3f}"
                + (f", CoM_offset={off*100:.1f}cm" if off > 0.005 else "") + ")")


@dataclass
class GraspPlan:
    strategy: str
    quality: float
    safe_home: np.ndarray
    approach_pos: np.ndarray
    grasp_pos: np.ndarray
    lift_pos: np.ndarray
    orientation: np.ndarray
    gripper_open_width: float = 255.0
    gripper_close_cmd: float = -100.0
    approach_steps: int = 1500
    grasp_steps: int = 1500
    final_steps: int = 1200
    final_tol: float = 0.025
    final_grasp_pos: np.ndarray = None


# ─────────────────────────────────────────────────────────────────────────────
# OBJECT ANALYSIS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_object(model: mujoco.MjModel,
                   data: mujoco.MjData,
                   object_body_name: str = "cube") -> ObjectGeometry:
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body_name)
    if body_id < 0:
        raise ValueError(f"Body '{object_body_name}' not found.")

    com = data.subtree_com[body_id].copy()

    geom_id = -1
    for cname in [f"{object_body_name}_geom", object_body_name]:
        gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, cname)
        if gid >= 0 and model.geom_bodyid[gid] == body_id:
            geom_id = gid
            break
    if geom_id < 0:
        for gid in range(model.ngeom):
            if model.geom_bodyid[gid] == body_id:
                geom_id = gid
                break
    if geom_id < 0:
        raise ValueError(f"No geom for '{object_body_name}'.")

    geom_centre = data.geom_xpos[geom_id].copy()
    geom_type   = model.geom_type[geom_id]
    size        = model.geom_size[geom_id].copy()

    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        width, depth, height = size[0]*2, size[1]*2, size[2]*2
    elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
        width = depth = size[0] * 2
        height        = size[1] * 2
    elif geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
        width = depth = height = size[0] * 2
    elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
        width = depth = size[0] * 2
        height        = size[1]*2 + size[0]*2
    else:
        print(f"⚠️  Unknown geom type {geom_type}, defaulting to 5cm")
        width = depth = height = 0.05

    return ObjectGeometry(
        name=object_body_name,
        com=com, geom_centre=geom_centre,
        geom_type=geom_type, size_raw=size,
        height=height, width=width, depth=depth,
        table_z=geom_centre[2] - height / 2.0,
        top_z  =geom_centre[2] + height / 2.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# QUATERNION UTILITIES
# ─────────────────────────────────────────────────────────────────────────────

def euler_to_quat(roll, pitch, yaw):
    cy, sy = np.cos(yaw*0.5),   np.sin(yaw*0.5)
    cp, sp = np.cos(pitch*0.5), np.sin(pitch*0.5)
    cr, sr = np.cos(roll*0.5),  np.sin(roll*0.5)
    return np.array([cr*cp*cy + sr*sp*sy,
                     sr*cp*cy - cr*sp*sy,
                     cr*sp*cy + sr*cp*sy,
                     cr*cp*sy - sr*sp*cy])

def gripper_down_quat():
    return euler_to_quat(np.pi, 0.0, 0.0)

def gripper_side_quat(approach_angle, roll=0.0):
    return euler_to_quat(np.pi + roll, -np.pi / 2.0, approach_angle + np.pi)


# ─────────────────────────────────────────────────────────────────────────────
# GRASP PLANNING
# ─────────────────────────────────────────────────────────────────────────────

def plan_top_down_grasp(obj: ObjectGeometry,
                        safe_home: np.ndarray,
                        use_com: bool = True) -> GraspPlan:
    centre       = obj.com if use_com else obj.geom_centre
    ideal_hand_z = centre[2] + HAND_TO_FINGERTIP_Z
    hand_z       = min(ideal_hand_z, MAX_HAND_Z)
    cx, cy       = centre[0], centre[1]
    label        = "top_down_map" if use_com else "top_down_naive"
    return GraspPlan(
        strategy=label, quality=0.85,
        safe_home      =safe_home.copy(),
        approach_pos   =np.array([cx, cy, hand_z + 0.15]),
        grasp_pos      =np.array([cx, cy, hand_z]),
        lift_pos       =np.array([cx, cy, hand_z + 0.20]),
        orientation    =gripper_down_quat(),
        approach_steps =1500, grasp_steps=1200,
        final_steps=800, final_tol=0.025, final_grasp_pos=None,
    )


def plan_side_grasp(obj: ObjectGeometry,
                    safe_home: np.ndarray,
                    approach_angle: float,
                    label: str = "side",
                    quality_override: float = None,
                    roll: float = 0.0,
                    use_com: bool = True) -> Optional[GraspPlan]:
    if not obj.is_graspable_laterally() or not obj.is_tall():
        return None

    CLEARANCE = 0.005
    STANDOFF  = 0.18
    MIN_Z     = 0.20

    centre  = obj.com if use_com else obj.geom_centre
    grasp_z = centre[2]
    if grasp_z < MIN_Z:
        return None

    dx = np.cos(approach_angle)
    dy = np.sin(approach_angle)

    obj_radius    = obj.radius_xy()
    hand_standoff = obj_radius + CLEARANCE

    final_grasp_pos = np.array([
        centre[0] + dx * hand_standoff,
        centre[1] + dy * hand_standoff,
        grasp_z,
    ])
    approach_pos = np.array([
        centre[0] + dx * (hand_standoff + STANDOFF),
        centre[1] + dy * (hand_standoff + STANDOFF),
        grasp_z + 0.25,
    ])
    mid_pos  = np.array([approach_pos[0], approach_pos[1], grasp_z + 0.04])
    lift_pos = np.array([final_grasp_pos[0], final_grasp_pos[1], grasp_z + 0.22])

    final_xy    = np.linalg.norm(final_grasp_pos[:2])
    approach_xy = np.linalg.norm(approach_pos[:2])
    if final_xy < 0.28 or final_xy > 0.75:   return None
    if approach_xy > 0.68:                     return None
    if final_grasp_pos[0] < 0.25:             return None

    quality = quality_override if quality_override is not None else 0.70
    return GraspPlan(
        strategy=label, quality=quality,
        safe_home=safe_home.copy(),
        approach_pos=approach_pos, grasp_pos=mid_pos,
        lift_pos=lift_pos, final_grasp_pos=final_grasp_pos,
        orientation=gripper_side_quat(approach_angle, roll=roll),
        approach_steps=1500, grasp_steps=1500,
        final_steps=1200, final_tol=0.025,
    )


def plan_side_grasp_mirrored(obj, safe_home, source_plan, label="side_right"):
    if source_plan is None:
        return None
    def mp(p): m = p.copy(); m[1] = -m[1]; return m
    def mq(q): w,x,y,z = q; return np.array([w,-x,y,-z])

    fm = mp(source_plan.final_grasp_pos)
    am = mp(source_plan.approach_pos)
    mm = mp(source_plan.grasp_pos)
    lm = mp(source_plan.lift_pos)

    if np.linalg.norm(fm[:2]) < 0.28 or np.linalg.norm(fm[:2]) > 0.75: return None
    if np.linalg.norm(am[:2]) > 0.68:  return None
    if fm[0] < 0.25:                    return None

    return GraspPlan(
        strategy=label, quality=source_plan.quality - 0.12,
        safe_home=mp(source_plan.safe_home),
        approach_pos=am, grasp_pos=mm, lift_pos=lm, final_grasp_pos=fm,
        orientation=mq(source_plan.orientation),
        approach_steps=source_plan.approach_steps,
        grasp_steps=source_plan.grasp_steps,
        final_steps=source_plan.final_steps,
        final_tol=source_plan.final_tol,
    )


def generate_grasp_candidates(obj: ObjectGeometry,
                               safe_home: np.ndarray,
                               heatmap=None,
                               use_map: bool = True) -> List[GraspPlan]:
    candidates: List[GraspPlan] = []

    if not use_map:
        # ── NAIVE: grasp geometric centre, side grasps first ─────────────────
        print("  [NAIVE] Grasping geometric centre — ignoring CoM")

        if obj.is_tall() and obj.is_graspable_laterally():
            # Side grasps ranked FIRST — with shifted CoM the object will tip
            left = plan_side_grasp(obj, safe_home,
                                   approach_angle=-np.pi/2,
                                   label="side_naive_left",
                                   quality_override=0.60,
                                   use_com=False)
            if left is not None:
                candidates.append(left)
            right = plan_side_grasp_mirrored(obj, safe_home, left,
                                              label="side_naive_right")
            if right is not None:
                candidates.append(right)

        # Top-down naive as lower-ranked fallback
        top_plan = plan_top_down_grasp(obj, safe_home, use_com=False)
        top_plan.quality = 0.40
        candidates.append(top_plan)

    else:
        # ── AFFORDANCE-GUIDED: use heatmap ────────────────────────────────────
        top_plan = plan_top_down_grasp(obj, safe_home, use_com=True)
        if heatmap is not None:
            top_plan.quality = heatmap.top_down_quality()
        candidates.append(top_plan)

        if obj.is_tall() and obj.is_graspable_laterally():
            if heatmap is not None:
                ranked = heatmap.best_side_approach(n_best=3)
                if ranked:
                    print(f"\n  Heatmap top-{len(ranked)} side angles:")
                    for ang_rad, qual in ranked:
                        print(f"    {np.degrees(ang_rad):6.1f}°  q={qual:.3f}")

                added_left = added_right = False
                for approach_angle, quality in ranked:
                    deg = np.degrees(approach_angle) % 360
                    if 70 <= deg <= 110 and not added_left:
                        p = plan_side_grasp(obj, safe_home,
                                            approach_angle=np.pi/2,
                                            label="side_left",
                                            quality_override=quality,
                                            use_com=True)
                        if p:
                            candidates.append(p); added_left = True
                    elif 250 <= deg <= 290 and not added_right:
                        tmp = plan_side_grasp(obj, safe_home,
                                              approach_angle=np.pi/2,
                                              label="side_left_tmp",
                                              quality_override=quality,
                                              use_com=True)
                        if tmp:
                            rp = plan_side_grasp_mirrored(obj, safe_home, tmp,
                                                           label="side_right")
                            if rp:
                                rp.quality = quality
                                candidates.append(rp); added_right = True
            else:
                left = plan_side_grasp(obj, safe_home,
                                       approach_angle=np.pi/2,
                                       label="side_left",
                                       quality_override=0.82)
                if left:
                    candidates.append(left)
                    right = plan_side_grasp_mirrored(obj, safe_home, left)
                    if right:
                        candidates.append(right)

    candidates.sort(key=lambda c: c.quality, reverse=True)
    mode = "MAP" if use_map else "NAIVE"
    print(f"\n  [{mode}] Generated {len(candidates)} candidates:")
    for i, c in enumerate(candidates):
        fd = (np.linalg.norm(c.final_grasp_pos[:2])
              if c.final_grasp_pos is not None
              else np.linalg.norm(c.grasp_pos[:2]))
        print(f"    {i+1}. {c.strategy:26s}  q={c.quality:.3f}  final_xy={fd:.3f}m")
    return candidates


# ─────────────────────────────────────────────────────────────────────────────
# OSC CONTROLLER
# ─────────────────────────────────────────────────────────────────────────────

class OSController:
    def __init__(self, model, data, ee_body="hand",
                 kp_pos=2.8, kp_ori=1.5, damping=0.01, null_kp=0.3):
        self.model     = model
        self.data      = data
        self.ee_id     = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if self.ee_id < 0:
            raise ValueError(f"Body '{ee_body}' not found.")
        self.kp_pos    = kp_pos
        self.kp_ori    = kp_ori
        self.damping   = damping
        self.null_kp   = null_kp
        self.dt        = model.opt.timestep
        self.nj        = 7
        self.q_desired = np.zeros(self.nj)
        self.q_min     = model.jnt_range[:self.nj, 0].copy()
        self.q_max     = model.jnt_range[:self.nj, 1].copy()
        self.q_home    = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785])
        print(f"  OSController ready (ee={ee_body})")

    def reset(self, q):
        self.q_desired = q[:self.nj].copy()

    def _quat_error(self, q_tgt, q_cur):
        qt = q_tgt / np.linalg.norm(q_tgt)
        qc = q_cur / np.linalg.norm(q_cur)
        if np.dot(qt, qc) < 0: qc = -qc
        wt,xt,yt,zt = qt; wc,xc,yc,zc = qc
        return 2.0 * np.array([
            -wt*xc+xt*wc-yt*zc+zt*yc,
            -wt*yc+xt*zc+yt*wc-zt*xc,
            -wt*zc-xt*yc+yt*xc+zt*wc])

    def move_to(self, target_pos, target_quat, steps=1000,
                viewer=None, pos_tol=0.012) -> bool:
        target_quat = target_quat / np.linalg.norm(target_quat)
        for i in range(steps):
            mujoco.mj_forward(self.model, self.data)
            pos_err = target_pos - self.data.xpos[self.ee_id]
            ori_err = self._quat_error(target_quat, self.data.xquat[self.ee_id])
            if np.linalg.norm(pos_err) < pos_tol: break
            Jp = np.zeros((3, self.model.nv))
            Jr = np.zeros((3, self.model.nv))
            mujoco.mj_jacBodyCom(self.model, self.data, Jp, Jr, self.ee_id)
            J      = np.vstack([Jp[:,:self.nj], Jr[:,:self.nj]])
            v_task = np.hstack([self.kp_pos*pos_err, self.kp_ori*ori_err])
            Jinv   = J.T @ np.linalg.inv(J@J.T + self.damping*np.eye(6))
            dq     = Jinv @ v_task
            dq    += (np.eye(self.nj) - Jinv@J) @ (self.null_kp * (
                      self.q_home - self.q_desired))
            self.q_desired = np.clip(
                self.q_desired + self.dt*dq, self.q_min, self.q_max)
            self.data.ctrl[:self.nj] = self.q_desired
            mujoco.mj_step(self.model, self.data)
            if viewer and i % 10 == 0: viewer.sync()
        return np.linalg.norm(self.data.xpos[self.ee_id]-target_pos) < pos_tol*2.5


# ─────────────────────────────────────────────────────────────────────────────
# GRIPPER HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def set_gripper(data, model, cmd, steps=300, viewer=None):
    for i in range(steps):
        data.ctrl[7] = cmd
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0: viewer.sync()

def ramp_gripper(data, model, from_cmd, to_cmd, steps=400, viewer=None):
    for i in range(steps):
        t = i / steps
        data.ctrl[7] = (1-t)*from_cmd + t*to_cmd
        mujoco.mj_step(model, data)
        if viewer and i % 10 == 0: viewer.sync()


# ─────────────────────────────────────────────────────────────────────────────
# GRASP EXECUTION
# ─────────────────────────────────────────────────────────────────────────────

def execute_grasp(model, data, osc, grasp, viewer=None,
                  verbose=True) -> Tuple[bool, float, str]:
    cube_id   = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    initial_z = data.xpos[cube_id][2]

    if verbose:
        print(f"\n  ─── {grasp.strategy}  q={grasp.quality:.3f} ───")
        print(f"      approach    = {np.round(grasp.approach_pos,3)}")
        if grasp.final_grasp_pos is not None:
            print(f"      final_grasp = {np.round(grasp.final_grasp_pos,3)}")

    data.ctrl[7] = grasp.gripper_open_width
    osc.move_to(grasp.safe_home, gripper_down_quat(), steps=800, viewer=viewer)

    if verbose: print("      → Stage 1: approach...")
    data.ctrl[7] = grasp.gripper_open_width
    if not osc.move_to(grasp.approach_pos, grasp.orientation,
                       steps=grasp.approach_steps, viewer=viewer):
        return False, 0.0, "failed_approach"

    if verbose: print("      → Stage 2: descend...")
    if not osc.move_to(grasp.grasp_pos, grasp.orientation,
                       steps=grasp.grasp_steps, viewer=viewer):
        return False, 0.0, "failed_descent"

    if grasp.final_grasp_pos is not None:
        if verbose: print("      → Stage 3: inward...")
        if not osc.move_to(grasp.final_grasp_pos, grasp.orientation,
                           steps=grasp.final_steps,
                           pos_tol=grasp.final_tol, viewer=viewer):
            return False, 0.0, "failed_final_pose"

    if verbose: print("      → Closing gripper...")
    ramp_gripper(data, model, grasp.gripper_open_width,
                 grasp.gripper_close_cmd, steps=400, viewer=viewer)
    set_gripper(data, model, grasp.gripper_close_cmd, steps=300, viewer=viewer)

    if verbose: print("      → Lifting...")
    lift_target = (np.array([grasp.final_grasp_pos[0],
                              grasp.final_grasp_pos[1],
                              grasp.lift_pos[2]])
                   if grasp.final_grasp_pos is not None else grasp.lift_pos)
    osc.move_to(lift_target, grasp.orientation, steps=1800, viewer=viewer)
    set_gripper(data, model, grasp.gripper_close_cmd, steps=400, viewer=viewer)

    final_z   = data.xpos[cube_id][2]
    lift_dist = final_z - initial_z
    success   = lift_dist >= 0.07
    if verbose:
        print(f"      → Lift: {lift_dist*100:.1f} cm  {'✓' if success else '✗'}")
    return success, lift_dist, "success" if success else f"lift_{lift_dist*100:.0f}cm"


# ─────────────────────────────────────────────────────────────────────────────
# SCENE RESET
# ─────────────────────────────────────────────────────────────────────────────

HOME_Q      = np.array([0.0, -0.3, 0.0, -2.0, 0.0, 1.8, 0.785])
DEFAULT_OBJ = np.array([0.55, 0.00, 0.50])
SAFE_HOME   = np.array([0.45, 0.00, 0.70])


def reset_scene(model, data, home_q, obj_pos=DEFAULT_OBJ, viewer=None):
    data.qpos[:7] = home_q
    data.qvel[:7] = 0.0
    if model.nq > 7:
        data.qpos[7:9] = 0.04
        data.qvel[7:9] = 0.0
    cube_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    cube_jnt = model.body_jntadr[cube_id]
    if cube_jnt >= 0 and model.jnt_type[cube_jnt] == mujoco.mjtJoint.mjJNT_FREE:
        qa = model.jnt_qposadr[cube_jnt]
        data.qpos[qa:qa+3]   = obj_pos
        data.qpos[qa+3:qa+7] = [1,0,0,0]
        va = model.jnt_dofadr[cube_jnt]
        data.qvel[va:va+6]   = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(600):
        data.ctrl[:7] = home_q
        data.ctrl[7]  = 255.0
        mujoco.mj_step(model, data)
        if viewer: viewer.sync()


# ─────────────────────────────────────────────────────────────────────────────
# RUN STANDARD SCENE
# ─────────────────────────────────────────────────────────────────────────────

def run_scene(xml_path: str, obj_name="cube",
              max_attempts=6, test_all=False) -> bool:
    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.lookat   = [0.5, 0.0, 0.4]
        osc = OSController(model, data)

        print(f"\n{'='*60}")
        print(f"  Scene: {Path(xml_path).name}")
        print(f"{'='*60}")

        reset_scene(model, data, HOME_Q, viewer=viewer)
        osc.reset(HOME_Q)
        obj = analyze_object(model, data, obj_name)
        print(f"\n  {obj}")

        try:
            from grasp_affordance_heatmap import AffordanceHeatmap
            hm = AffordanceHeatmap(model, data, obj_name)
            hm.compute(n_top_radial=8, n_top_angular=12,
                       n_side_height=8, n_side_angular=24)
            candidates = generate_grasp_candidates(obj, SAFE_HOME,
                                                   heatmap=hm, use_map=True)
        except ImportError:
            candidates = generate_grasp_candidates(obj, SAFE_HOME,
                                                   heatmap=None, use_map=True)

        results_table = []
        any_success   = False

        for i, grasp in enumerate(candidates[:max_attempts]):
            print(f"\n  ── Attempt {i+1}/{min(len(candidates),max_attempts)} ──")
            reset_scene(model, data, HOME_Q, viewer=viewer)
            osc.reset(HOME_Q)
            ok, lift, reason = execute_grasp(model, data, osc, grasp, viewer=viewer)
            results_table.append((grasp.strategy, grasp.quality, ok, lift, reason))
            if ok:
                any_success = True
                print(f"\n  ✓✓✓  SUCCESS: {grasp.strategy} ({lift*100:.1f} cm)")
                if not test_all:
                    time.sleep(2.0); break
                time.sleep(1.0)
            else:
                print(f"  ✗  {reason}")
            time.sleep(0.4)

        print(f"\n  {'─'*62}")
        print(f"  {'#':<3} {'Strategy':<26} {'Q':>6}  {'R':<6}  {'Lift':>6}  Reason")
        print(f"  {'─'*62}")
        for idx,(strat,qual,ok,lift,reason) in enumerate(results_table):
            icon = "✓" if ok else "✗"
            print(f"  {idx+1:<3} {strat:<26} {qual:>6.3f}  {icon:<6}  "
                  f"{lift*100:>5.1f}cm  {reason}")
        successes = sum(1 for _,_,ok,_,_ in results_table if ok)
        print(f"  {'─'*62}")
        print(f"\n  Summary: {successes}/{len(results_table)} succeeded")
        print("\n  Closing in 3s...")
        time.sleep(3.0)
    return any_success


# ─────────────────────────────────────────────────────────────────────────────
# RUN SHIFTED CoM COMPARISON SCENE
# ─────────────────────────────────────────────────────────────────────────────

def run_shifted_com_scene(xml_path: str,
                          obj_name: str = "cube",
                          output_dir: str = "outputs/heatmaps") -> dict:
    from grasp_affordance_heatmap import AffordanceHeatmap

    model = mujoco.MjModel.from_xml_path(xml_path)
    data  = mujoco.MjData(model)

    print(f"\n{'#'*60}")
    print(f"  SHIFTED CoM COMPARISON: {Path(xml_path).name}")
    print(f"  Scenario 1: NAIVE  (geometric centre, no map)")
    print(f"  Scenario 2: MAP    (heatmap CoM-aware contact)")
    print(f"{'#'*60}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.8
        viewer.cam.lookat   = [0.5, 0.0, 0.4]
        osc = OSController(model, data)

        reset_scene(model, data, HOME_Q, viewer=viewer)
        osc.reset(HOME_Q)

        obj = analyze_object(model, data, obj_name)
        print(f"\n  {obj}")
        print(f"  CoM offset: {obj.com_offset()*100:.2f} cm")

        hm = AffordanceHeatmap(model, data, obj_name)
        hm.compute(n_top_radial=8, n_top_angular=12,
                   n_side_height=8, n_side_angular=24)

        naive_contact = hm.naive_contact()
        best_contact  = hm.best_contact()

        print(f"\n  Naive contact:  pos={np.round(naive_contact.position,3)}"
              f"  q={naive_contact.quality:.3f}"
              f"  angle={np.degrees(naive_contact.approach_angle):.0f}°")
        print(f"  Best contact:   pos={np.round(best_contact.position,3)}"
              f"  q={best_contact.quality:.3f}"
              f"  angle={np.degrees(best_contact.approach_angle):.0f}°")
        print(f"  Quality gap:    {(best_contact.quality-naive_contact.quality)*100:.1f}%")

        results = {}

        # ── Scenario 1: NAIVE ─────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  SCENARIO 1: NAIVE — grasping geometric centre")
        print(f"{'─'*50}")

        reset_scene(model, data, HOME_Q, viewer=viewer)
        osc.reset(HOME_Q)

        naive_candidates = generate_grasp_candidates(
            obj, SAFE_HOME, heatmap=None, use_map=False)

        naive_ok, naive_lift, naive_reason = False, 0.0, "no_candidate"
        if naive_candidates:
            naive_ok, naive_lift, naive_reason = execute_grasp(
                model, data, osc, naive_candidates[0], viewer=viewer)
            # Tipping object may lift briefly then fall — needs 10cm to count
            naive_ok = naive_lift >= 0.10

        results["naive"] = {
            "contact": naive_contact,
            "lift_cm": naive_lift * 100,
            "success": naive_ok,
            "label":   "Scenario 1: NAIVE\n(geometric centre, no map)",
        }
        print(f"\n  NAIVE result: {'✓' if naive_ok else '✗'}  "
              f"lift={naive_lift*100:.1f}cm  ({naive_reason})")
        time.sleep(1.5)

        # ── Scenario 2: MAP ───────────────────────────────────────────────────
        print(f"\n{'─'*50}")
        print(f"  SCENARIO 2: MAP — grasping heatmap best contact")
        print(f"{'─'*50}")

        reset_scene(model, data, HOME_Q, viewer=viewer)
        osc.reset(HOME_Q)

        map_candidates = generate_grasp_candidates(
            obj, SAFE_HOME, heatmap=hm, use_map=True)

        map_ok, map_lift, map_reason = False, 0.0, "no_candidate"
        if map_candidates:
            map_ok, map_lift, map_reason = execute_grasp(
                model, data, osc, map_candidates[0], viewer=viewer)

        results["map"] = {
            "contact": best_contact,
            "lift_cm": map_lift * 100,
            "success": map_ok,
            "label":   "Scenario 2: AFFORDANCE MAP\n(CoM-aware contact)",
        }
        print(f"\n  MAP result:   {'✓' if map_ok else '✗'}  "
              f"lift={map_lift*100:.1f}cm  ({map_reason})")
        time.sleep(1.5)

        # ── Summary ───────────────────────────────────────────────────────────
        print(f"\n  {'═'*50}")
        print(f"  COMPARISON SUMMARY")
        print(f"  {'═'*50}")
        print(f"  Naive  → lift={naive_lift*100:.1f}cm  {'✓' if naive_ok else '✗ FAIL'}")
        print(f"  Map    → lift={map_lift*100:.1f}cm  {'✓' if map_ok else '✗ FAIL'}")
        improvement = map_lift*100 - naive_lift*100
        print(f"  Improvement: {improvement:+.1f} cm  "
              f"({'map wins' if improvement > 0 else 'naive similar or better'})")
        print(f"  {'═'*50}")

        # ── Save all plots ────────────────────────────────────────────────────
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        hm.save_all(str(out), tag="shifted_com")
        hm.plot_comparison(
            naive_result=results["naive"],
            map_result  =results["map"],
            save_path   =str(out / "comparison_shifted_com.png"),
        )

        print("\n  Closing in 3s...")
        time.sleep(3.0)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ROOT       = Path(__file__).resolve().parent.parent
    SCENES_DIR = ROOT / "assets" / "scenes"
    OUTPUT_DIR = ROOT / "outputs" / "heatmaps"

    # ── Standard scenes ───────────────────────────────────────────────────────
    standard_scenes = [
        ("scene_grasp_clean.xml",    "cube"),
        ("scene_grasp_cylinder.xml", "cube"),
        ("scene_grasp_tallbox.xml",  "cube"),
    ]

    TEST_ALL = True
    results  = {}

    for scene_file, obj_name in standard_scenes:
        xml = SCENES_DIR / scene_file
        if not xml.exists():
            print(f"⚠️  Not found: {xml}"); continue
        print(f"\n\n{'#'*60}\n  TESTING: {scene_file}\n{'#'*60}")
        results[scene_file] = (
            "✓ SUCCESS"
            if run_scene(str(xml), obj_name, test_all=TEST_ALL)
            else "✗ FAILED"
        )

    # ── Shifted CoM comparison ────────────────────────────────────────────────
    shifted_xml = SCENES_DIR / "scene_shifted_com.xml"
    if shifted_xml.exists():
        shifted_results = run_shifted_com_scene(
            str(shifted_xml), "cube", str(OUTPUT_DIR))
        naive_ok = shifted_results["naive"]["success"]
        map_ok   = shifted_results["map"]["success"]
        results["scene_shifted_com.xml"] = (
            f"NAIVE={'✓' if naive_ok else '✗'}  "
            f"MAP={'✓' if map_ok else '✗'}")
    else:
        print(f"⚠️  Not found: {shifted_xml}")

    # ── Overall results ───────────────────────────────────────────────────────
    print(f"\n\n{'='*60}\n  OVERALL RESULTS\n{'='*60}")
    for s, r in results.items():
        print(f"  {r}  {s}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()