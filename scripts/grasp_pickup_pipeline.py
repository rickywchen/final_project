from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dataclasses import dataclass
from pathlib import Path
import math
import numpy as np
import mujoco
import mujoco.viewer

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets" / "scenes" / "panda_table_mocap.xml"


@dataclass(frozen=True)
class PickupConfig:
    """Configuration for single pickup attempt - research grade"""
    
    # Timing (all in simulation steps)
    steps_settle: int = 200          # Let physics stabilize
    steps_per_waypoint: int = 300    # Slow, smooth motion
    steps_grasp_close: int = 150     # Slow finger closure
    steps_lift_hold: int = 200       # Hold at lift height
    steps_before_lift: int = 100     # Dwell after grasp closes
    
    # CRITICAL: Panda hand → fingertip offset
    # From diagnosis: finger_body is 0.0584m below hand, fingertips extend 0.05m more
    hand_to_fingertip_offset: float = 0.11  # meters in Z axis
    
    # Geometry (meters) - ALL RELATIVE TO CUBE CENTER
    pregrasp_height_above_cube: float = 0.25   # High clearance for approach
    pregrasp_radius_from_cube: float = 0.15    # Distance from cube in XY plane
    
    # For grasp: we want FINGERTIPS at cube center height
    # So hand needs to be ABOVE by hand_to_fingertip_offset
    grasp_fingertip_height_above_cube: float = 0.00  # Fingertips at cube center
    grasp_radius_from_cube: float = 0.00             # Directly above cube
    
    lift_height: float = 0.15                   # Lift distance to verify grasp
    
    # Contact verification
    min_contact_force: float = 0.5             # Newton
    min_contacts_for_success: int = 2          # Both fingers touching
    
    # Control
    gripper_open: float = 0.0                  # Actuator8 ctrlrange
    gripper_closed: float = 255.0
    
    # Names
    cube_body: str = "cube"
    cube_geom: str = "cube_geom"
    mocap_body: str = "hand_mocap"
    hand_body: str = "hand"
    left_finger_body: str = "left_finger"
    right_finger_body: str = "right_finger"
    
    arm_actuators: tuple[str, ...] = (
        "actuator1", "actuator2", "actuator3", "actuator4",
        "actuator5", "actuator6", "actuator7"
    )
    gripper_actuator: str = "actuator8"


def get_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    """Get MuJoCo ID with error handling"""
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise RuntimeError(f"Could not find {objtype} named '{name}'")
    return idx


def quat_from_euler_zyx(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    Convert Euler angles (Z-Y-X intrinsic) to quaternion (w, x, y, z).
    For top-down grasp: roll=0, pitch=pi, yaw=approach_angle
    """
    cr, cp, cy = math.cos(roll/2), math.cos(pitch/2), math.cos(yaw/2)
    sr, sp, sy = math.sin(roll/2), math.sin(pitch/2), math.sin(yaw/2)
    
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy
    
    return np.array([w, x, y, z], dtype=np.float64)


def set_mocap_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    mocap_body_name: str,
    pos: np.ndarray,
    quat: np.ndarray
) -> None:
    """Set mocap body pose"""
    bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
    madr = model.body_mocapid[bid]
    if madr < 0:
        raise RuntimeError(f"Body '{mocap_body_name}' is not mocap-enabled")
    data.mocap_pos[madr] = pos
    data.mocap_quat[madr] = quat


def set_arm_compliant(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm_actuator_ids: list[int]
) -> None:
    """
    Make arm actuators track current joint positions.
    Critical for mocap control - prevents fighting the weld constraint.
    """
    q_arm = data.qpos[:7]
    for i, aid in enumerate(arm_actuator_ids):
        data.ctrl[aid] = float(q_arm[i])


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Linear interpolation"""
    return (1.0 - t) * a + t * b


def slerp(q0: np.ndarray, q1: np.ndarray, t: float) -> np.ndarray:
    """
    Spherical linear interpolation for quaternions.
    More correct than linear interpolation for rotations.
    """
    dot = np.dot(q0, q1)
    
    # Take shorter path
    if dot < 0.0:
        q1 = -q1
        dot = -dot
    
    # Clamp for numerical stability
    dot = np.clip(dot, -1.0, 1.0)
    
    # Close to parallel - use linear interpolation
    if dot > 0.9995:
        result = q0 + t * (q1 - q0)
        return result / np.linalg.norm(result)
    
    # Spherical interpolation
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    w0 = math.sin((1.0 - t) * theta) / sin_theta
    w1 = math.sin(t * theta) / sin_theta
    
    return w0 * q0 + w1 * q1


def move_mocap_smooth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: PickupConfig,
    arm_actuator_ids: list[int],
    pos_start: np.ndarray,
    quat_start: np.ndarray,
    pos_end: np.ndarray,
    quat_end: np.ndarray,
    steps: int,
    viewer=None,
    label: str = ""
) -> None:
    """Smooth interpolated motion between two poses"""
    print(f"  Moving: {label}")
    print(f"    From: {pos_start}")
    print(f"    To:   {pos_end}")
    
    for i in range(steps):
        t = (i + 1) / steps
        pos = lerp(pos_start, pos_end, t)
        quat = slerp(quat_start, quat_end, t)
        
        set_mocap_pose(model, data, cfg.mocap_body, pos, quat)
        set_arm_compliant(model, data, arm_actuator_ids)
        
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()


def close_gripper_smooth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: PickupConfig,
    arm_actuator_ids: list[int],
    gripper_actuator_id: int,
    steps: int,
    viewer=None
) -> None:
    """Smoothly close gripper"""
    print(f"  Closing gripper over {steps} steps...")
    
    for i in range(steps):
        t = (i + 1) / steps
        grip_val = (1.0 - t) * cfg.gripper_open + t * cfg.gripper_closed
        
        set_arm_compliant(model, data, arm_actuator_ids)
        data.ctrl[gripper_actuator_id] = float(grip_val)
        
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()


def dwell(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: PickupConfig,
    arm_actuator_ids: list[int],
    steps: int,
    viewer=None,
    label: str = ""
) -> None:
    """Hold position while physics settles"""
    if label:
        print(f"  Dwelling: {label}")
    
    for _ in range(steps):
        set_arm_compliant(model, data, arm_actuator_ids)
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()


def check_finger_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cube_geom_id: int,
    left_finger_body_id: int,
    right_finger_body_id: int,
    min_force: float
) -> tuple[bool, bool, float, float]:
    """
    Check if left and right fingers are in contact with cube.
    Returns: (left_contact, right_contact, left_force, right_force)
    """
    left_force = 0.0
    right_force = 0.0
    
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        
        # Must involve cube
        if g1 != cube_geom_id and g2 != cube_geom_id:
            continue
        
        other_gid = g2 if g1 == cube_geom_id else g1
        other_bid = int(model.geom_bodyid[other_gid])
        
        # Compute force magnitude
        force_arr = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, i, force_arr)
        force_mag = float(np.linalg.norm(force_arr[:3]))
        
        if other_bid == left_finger_body_id:
            left_force = max(left_force, force_mag)
        elif other_bid == right_finger_body_id:
            right_force = max(right_force, force_mag)
    
    left_contact = left_force >= min_force
    right_contact = right_force >= min_force
    
    return left_contact, right_contact, left_force, right_force


def get_fingertip_positions(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_finger_body_id: int,
    right_finger_body_id: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Estimate fingertip positions in world frame.
    Fingertips are ~0.05m in +Z from finger body (in finger local frame).
    """
    left_finger_pos = data.xpos[left_finger_body_id].copy()
    right_finger_pos = data.xpos[right_finger_body_id].copy()
    
    left_finger_quat = data.xquat[left_finger_body_id].copy()
    right_finger_quat = data.xquat[right_finger_body_id].copy()
    
    # Fingertip offset in local frame: [0, 0, 0.05] (from diagnostic)
    local_offset = np.array([0.0, 0.0, 0.05], dtype=np.float64)
    
    # Rotate to world frame (simplified - assumes fingers point down)
    # For more accuracy, use proper quaternion rotation
    left_tip = left_finger_pos + local_offset
    right_tip = right_finger_pos + local_offset
    
    return left_tip, right_tip


def verify_grasp_success(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: PickupConfig,
    cube_body_id: int,
    initial_cube_z: float
) -> bool:
    """
    Check if cube was successfully lifted.
    Success = cube height increased significantly.
    """
    current_cube_z = data.xpos[cube_body_id][2]
    lift_achieved = current_cube_z - initial_cube_z
    
    success = lift_achieved > (cfg.lift_height * 0.7)  # 70% of target lift
    
    print(f"\n{'='*60}")
    print(f"GRASP VERIFICATION")
    print(f"{'='*60}")
    print(f"  Initial cube Z: {initial_cube_z:.4f} m")
    print(f"  Current cube Z: {current_cube_z:.4f} m")
    print(f"  Lift achieved:  {lift_achieved:.4f} m")
    print(f"  Target lift:    {cfg.lift_height:.4f} m")
    print(f"  Result:         {'✓ SUCCESS' if success else '✗ FAILED'}")
    print(f"{'='*60}\n")
    
    return success


def execute_pickup(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cfg: PickupConfig,
    approach_angle: float = 0.0,
    visualize: bool = True
) -> bool:
    """
    Execute a single pickup attempt.
    
    Returns: True if grasp succeeded (cube lifted), False otherwise
    """
    # Get IDs
    cube_bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.cube_body)
    cube_gid = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.cube_geom)
    left_finger_bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.left_finger_body)
    right_finger_bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.right_finger_body)
    hand_bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.hand_body)
    
    arm_actuator_ids = [get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in cfg.arm_actuators]
    gripper_actuator_id = get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, cfg.gripper_actuator)
    
    # Reset simulation
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    
    # Open gripper
    data.ctrl[gripper_actuator_id] = cfg.gripper_open
    
    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(model, data)
    
    try:
        print(f"\n{'='*60}")
        print(f"PICKUP ATTEMPT")
        print(f"  Approach angle: {math.degrees(approach_angle):.1f}°")
        print(f"  Hand→fingertip offset: {cfg.hand_to_fingertip_offset:.4f} m")
        print(f"{'='*60}\n")
        
        # Settle initial state
        print("[1/7] Initial settling...")
        dwell(model, data, cfg, arm_actuator_ids, cfg.steps_settle, viewer, "initial")
        
        # Get cube position
        cube_pos = data.xpos[cube_bid].copy()
        initial_cube_z = cube_pos[2]
        print(f"  Cube at: {cube_pos}")
        
        # Get current hand pose
        mocap_bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.mocap_body)
        mocap_adr = model.body_mocapid[mocap_bid]
        current_pos = data.mocap_pos[mocap_adr].copy()
        current_quat = data.mocap_quat[mocap_adr].copy()
        
        # Top-down grasp orientation (fingers pointing down, approach from angle)
        # Pitch = π (flip upside down), Yaw = approach direction
        grasp_quat = quat_from_euler_zyx(roll=0.0, pitch=math.pi, yaw=approach_angle + math.pi)
        
        # Waypoint 1: Pre-grasp (high, away from cube)
        # Position is for HAND body, will be higher than fingertips
        pregrasp_pos = np.array([
            cube_pos[0] + cfg.pregrasp_radius_from_cube * math.cos(approach_angle),
            cube_pos[1] + cfg.pregrasp_radius_from_cube * math.sin(approach_angle),
            cube_pos[2] + cfg.pregrasp_height_above_cube + cfg.hand_to_fingertip_offset
        ], dtype=np.float64)
        
        print("[2/7] Moving to pre-grasp...")
        move_mocap_smooth(
            model, data, cfg, arm_actuator_ids,
            current_pos, current_quat,
            pregrasp_pos, grasp_quat,
            cfg.steps_per_waypoint,
            viewer,
            label="pre-grasp"
        )
        
        dwell(model, data, cfg, arm_actuator_ids, cfg.steps_settle, viewer, "at pre-grasp")
        
        # Waypoint 2: Grasp pose
        # We want FINGERTIPS at cube center height
        # So HAND needs to be at: cube_z + fingertip_offset + grasp_fingertip_height
        grasp_hand_height = cube_pos[2] + cfg.hand_to_fingertip_offset + cfg.grasp_fingertip_height_above_cube
        
        grasp_pos = np.array([
            cube_pos[0] + cfg.grasp_radius_from_cube * math.cos(approach_angle),
            cube_pos[1] + cfg.grasp_radius_from_cube * math.sin(approach_angle),
            grasp_hand_height
        ], dtype=np.float64)
        
        print("[3/7] Moving to grasp pose...")
        print(f"  Target hand height: {grasp_hand_height:.4f} m")
        print(f"  Expected fingertip height: {grasp_hand_height - cfg.hand_to_fingertip_offset:.4f} m")
        
        move_mocap_smooth(
            model, data, cfg, arm_actuator_ids,
            pregrasp_pos, grasp_quat,
            grasp_pos, grasp_quat,
            cfg.steps_per_waypoint,
            viewer,
            label="grasp approach"
        )
        
        dwell(model, data, cfg, arm_actuator_ids, cfg.steps_settle, viewer, "at grasp pose")
        
        # Check positioning before closing
        hand_pos = data.xpos[hand_bid].copy()
        left_tip, right_tip = get_fingertip_positions(model, data, left_finger_bid, right_finger_bid)
        
        dist_hand_cube = np.linalg.norm(hand_pos - cube_pos)
        dist_left_tip_cube = np.linalg.norm(left_tip - cube_pos)
        dist_right_tip_cube = np.linalg.norm(right_tip - cube_pos)
        
        print(f"  Hand position: {hand_pos}")
        print(f"  Left fingertip estimate: {left_tip}")
        print(f"  Right fingertip estimate: {right_tip}")
        print(f"  Hand-cube distance: {dist_hand_cube:.4f} m")
        print(f"  Left tip-cube distance: {dist_left_tip_cube:.4f} m")
        print(f"  Right tip-cube distance: {dist_right_tip_cube:.4f} m")
        
        # Close gripper
        print("[4/7] Closing gripper...")
        close_gripper_smooth(
            model, data, cfg, arm_actuator_ids,
            gripper_actuator_id,
            cfg.steps_grasp_close,
            viewer
        )
        
        # Dwell to let contacts stabilize
        print("[5/7] Stabilizing grasp...")
        dwell(model, data, cfg, arm_actuator_ids, cfg.steps_before_lift, viewer, "grasp settling")
        
        # Check contacts
        left_contact, right_contact, left_force, right_force = check_finger_contacts(
            model, data, cube_gid, left_finger_bid, right_finger_bid, cfg.min_contact_force
        )
        
        print(f"\n  Contact check:")
        print(f"    Left finger:  {'✓' if left_contact else '✗'}  (force: {left_force:.2f} N)")
        print(f"    Right finger: {'✓' if right_contact else '✗'}  (force: {right_force:.2f} N)")
        
        if not (left_contact and right_contact):
            print(f"  ⚠ Warning: Insufficient finger contact before lift")
        
        # Lift
        print("[6/7] Attempting lift...")
        lift_pos = grasp_pos.copy()
        lift_pos[2] += cfg.lift_height
        
        move_mocap_smooth(
            model, data, cfg, arm_actuator_ids,
            grasp_pos, grasp_quat,
            lift_pos, grasp_quat,
            cfg.steps_per_waypoint,
            viewer,
            label="lift"
        )
        
        # Hold at lift height
        print("[7/7] Holding at lift height...")
        dwell(model, data, cfg, arm_actuator_ids, cfg.steps_lift_hold, viewer, "lift hold")
        
        # Verify success
        success = verify_grasp_success(model, data, cfg, cube_bid, initial_cube_z)
        
        return success
        
    finally:
        if viewer:
            print("\nClose viewer window to continue...")
            while viewer.is_running():
                mujoco.mj_step(model, data)
                viewer.sync()
            viewer.close()


def main():
    """Run single pickup test"""
    if not XML.exists():
        raise FileNotFoundError(f"Scene XML not found: {XML}")
    
    cfg = PickupConfig()
    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)
    
    # Test single grasp from front (0°)
    approach_angle = 0.0  # radians (0 = +X direction)
    
    success = execute_pickup(
        model, data, cfg,
        approach_angle=approach_angle,
        visualize=True
    )
    
    if success:
        print("\n✓ Pickup successful!")
    else:
        print("\n✗ Pickup failed - cube not lifted")


if __name__ == "__main__":
    main()