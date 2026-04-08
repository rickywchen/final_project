from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from dataclasses import dataclass
from pathlib import Path
import math
import time
import numpy as np
import mujoco
import mujoco.viewer

from contact_logging import (
    ContactSample,
    quat_from_yaw,
    get_contacts_for_object_filtered_by_other_bodies,
)

ROOT = Path(__file__).resolve().parents[1]
XML = ROOT / "assets" / "scenes" / "panda_table_mocap.xml"

OUT_DIR = ROOT / "assets" / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class Config:
    n_trials: int = 30
    seed: int = 0

    # motion/stability
    steps_settle: int = 150
    steps_move: int = 200
    steps_close_total: int = 250
    steps_record: int = 250
    close_ramp_steps: int = 120

    # sampling around cube
    radius_pre: float = 0.18
    radius_app: float = 0.11
    z_offset_pre: float = 0.12
    z_offset_app: float = 0.08

    # contact filtering
    min_force_n: float = 2.0
    max_contacts_per_step: int = 8

    # names in your XML/model
    cube_body_name: str = "cube"
    cube_geom_name: str = "cube_geom"
    mocap_body_name: str = "hand_mocap"

    # finger bodies (these DO exist in your panda.xml)
    left_finger_body: str = "left_finger"
    right_finger_body: str = "right_finger"

    # actuator names (these exist in your panda.xml)
    arm_actuator_names: tuple[str, ...] = (
        "actuator1", "actuator2", "actuator3", "actuator4", "actuator5", "actuator6", "actuator7"
    )
    gripper_actuator_name: str = "actuator8"

    # gripper ctrl values for panda.xml ctrlrange 0..255
    gripper_open: float = 0.0
    gripper_closed: float = 255.0


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def get_id(model: mujoco.MjModel, objtype: mujoco.mjtObj, name: str) -> int:
    idx = mujoco.mj_name2id(model, objtype, name)
    if idx < 0:
        raise RuntimeError(f"Could not find {objtype} named '{name}'.")
    return idx


def set_mocap_pose(model: mujoco.MjModel, data: mujoco.MjData, mocap_body_name: str, pos: np.ndarray, quat: np.ndarray) -> None:
    bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, mocap_body_name)
    madr = model.body_mocapid[bid]
    if madr < 0:
        raise RuntimeError(f"Body '{mocap_body_name}' is not mocap-enabled (mocap='true').")
    data.mocap_pos[madr] = pos
    data.mocap_quat[madr] = quat


def step_n(model: mujoco.MjModel, data: mujoco.MjData, n: int, viewer=None) -> None:
    for _ in range(n):
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()


def set_arm_compliant(model: mujoco.MjModel, data: mujoco.MjData, arm_aids: list[int]) -> None:
    """
    Key stability trick:
    Make the arm position servos track their current qpos each step.
    This stops them from fighting the mocap weld.
    """
    q_arm = data.qpos[:7]
    for i, aid in enumerate(arm_aids):
        data.ctrl[aid] = float(q_arm[i])


def move_mocap_smooth(model: mujoco.MjModel, data: mujoco.MjData, cfg: Config, pos_target: np.ndarray, quat_target: np.ndarray, viewer=None) -> None:
    bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.mocap_body_name)
    madr = model.body_mocapid[bid]
    pos0 = data.mocap_pos[madr].copy()
    quat0 = data.mocap_quat[madr].copy()

    for i in range(cfg.steps_move):
        t = (i + 1) / cfg.steps_move
        pos = lerp(pos0, pos_target, t)
        quat = lerp(quat0, quat_target, t)
        quat = quat / (np.linalg.norm(quat) + 1e-12)
        set_mocap_pose(model, data, cfg.mocap_body_name, pos, quat)
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()

    step_n(model, data, cfg.steps_settle, viewer=viewer)


def ramp_gripper_close(model: mujoco.MjModel, data: mujoco.MjData, cfg: Config, grip_aid: int, arm_aids: list[int], viewer=None) -> None:
    for i in range(cfg.close_ramp_steps):
        t = (i + 1) / cfg.close_ramp_steps
        v = (1.0 - t) * cfg.gripper_open + t * cfg.gripper_closed
        set_arm_compliant(model, data, arm_aids)
        data.ctrl[grip_aid] = float(v)
        mujoco.mj_step(model, data)
        if viewer:
            viewer.sync()

    remaining = max(0, cfg.steps_close_total - cfg.close_ramp_steps)
    step_n(model, data, remaining, viewer=viewer)


def save_contacts_npz(path: Path, samples: list[ContactSample]) -> None:
    if len(samples) == 0:
        np.savez_compressed(
            path,
            pos_world=np.zeros((0, 3), dtype=np.float64),
            normal_world=np.zeros((0, 3), dtype=np.float64),
            force_n=np.zeros((0,), dtype=np.float64),
            other_geom_id=np.zeros((0,), dtype=np.int32),
            other_body_id=np.zeros((0,), dtype=np.int32),
            cube_pos=np.zeros((0, 3), dtype=np.float64),
            cube_quat=np.zeros((0, 4), dtype=np.float64),
            trial_index=np.zeros((0,), dtype=np.int32),
            timestamp=np.zeros((0,), dtype=np.float64),
        )
        return

    pos_world = np.stack([s.pos_world for s in samples], axis=0)
    normal_world = np.stack([s.normal_world for s in samples], axis=0)
    force_n = np.array([s.force_n for s in samples], dtype=np.float64)
    other_geom_id = np.array([s.other_geom_id for s in samples], dtype=np.int32)
    other_body_id = np.array([s.other_body_id for s in samples], dtype=np.int32)
    cube_pos = np.stack([s.cube_pos_world for s in samples], axis=0)
    cube_quat = np.stack([s.cube_quat_world for s in samples], axis=0)
    trial_index = np.array([s.trial_index for s in samples], dtype=np.int32)
    timestamp = np.array([s.timestamp for s in samples], dtype=np.float64)

    np.savez_compressed(
        path,
        pos_world=pos_world,
        normal_world=normal_world,
        force_n=force_n,
        other_geom_id=other_geom_id,
        other_body_id=other_body_id,
        cube_pos=cube_pos,
        cube_quat=cube_quat,
        trial_index=trial_index,
        timestamp=timestamp,
    )


def run(cfg: Config, visualize: bool = True) -> None:
    if not XML.exists():
        raise FileNotFoundError(f"Scene XML not found: {XML}")

    np.random.seed(cfg.seed)

    model = mujoco.MjModel.from_xml_path(str(XML))
    data = mujoco.MjData(model)

    cube_gid = get_id(model, mujoco.mjtObj.mjOBJ_GEOM, cfg.cube_geom_name)
    cube_bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.cube_body_name)

    left_finger_bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.left_finger_body)
    right_finger_bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.right_finger_body)
    allowed_bodies = {left_finger_bid, right_finger_bid}

    arm_aids = [get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in cfg.arm_actuator_names]
    grip_aid = get_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, cfg.gripper_actuator_name)

    all_samples: list[ContactSample] = []

    viewer = None
    if visualize:
        viewer = mujoco.viewer.launch_passive(model, data)

    try:
        for k in range(cfg.n_trials):
            mujoco.mj_resetData(model, data)
            mujoco.mj_forward(model, data)

            # open gripper
            set_arm_compliant(model, data, arm_aids)
            data.ctrl[grip_aid] = float(cfg.gripper_open)
            step_n(model, data, cfg.steps_settle, viewer=viewer)

            # cube position
            cube_pos = data.xpos[cube_bid].copy()
            # ---- SANITY CHECK: are hand and mocap near the cube? ----
            hand_bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, "hand")
            mocap_bid = get_id(model, mujoco.mjtObj.mjOBJ_BODY, cfg.mocap_body_name)
            madr = model.body_mocapid[mocap_bid]

            hand_pos = data.xpos[hand_bid].copy()
            mocap_pos = data.mocap_pos[madr].copy()

            print(f"[trial {k}] cube_pos   = {cube_pos}")
            print(f"[trial {k}] hand_pos   = {hand_pos}   dist(hand,cube)={np.linalg.norm(hand_pos-cube_pos):.3f}")
            print(f"[trial {k}] mocap_pos  = {mocap_pos}  dist(mocap,cube)={np.linalg.norm(mocap_pos-cube_pos):.3f}")
            print(f"[trial {k}] dist(hand,mocap)={np.linalg.norm(hand_pos-mocap_pos):.3f}")


            # sample around cube
            yaw = (2.0 * math.pi) * (k / cfg.n_trials)
            q = quat_from_yaw(yaw + math.pi)  # face inward

            pregrasp = np.array([
                cube_pos[0] + cfg.radius_pre * math.cos(yaw),
                cube_pos[1] + cfg.radius_pre * math.sin(yaw),
                cube_pos[2] + cfg.z_offset_pre
            ], dtype=np.float64)

            approach = np.array([
                cube_pos[0] + cfg.radius_app * math.cos(yaw),
                cube_pos[1] + cfg.radius_app * math.sin(yaw),
                cube_pos[2] + cfg.z_offset_app
            ], dtype=np.float64)

            # smooth motions
            move_mocap_smooth(model, data, cfg, pregrasp, q, viewer=viewer)
            move_mocap_smooth(model, data, cfg, approach, q, viewer=viewer)

            # close gripper smoothly
            ramp_gripper_close(model, data, cfg, grip_aid, arm_aids, viewer=viewer)

            # record window
            for _ in range(cfg.steps_record):
                set_arm_compliant(model, data, arm_aids)
                mujoco.mj_step(model, data)
                if viewer:
                    viewer.sync()

                samples = get_contacts_for_object_filtered_by_other_bodies(
                    model=model,
                    data=data,
                    object_geom_id=cube_gid,
                    allowed_other_body_ids=allowed_bodies,
                    min_force_n=cfg.min_force_n,
                    max_per_step=cfg.max_contacts_per_step,
                    object_body_id=cube_bid,
                    trial_index=k,
                    debug_nonfinger=True,
                    debug_limit=10,
                )
                all_samples.extend(samples)

            print(f"trial {k+1}/{cfg.n_trials} | total saved contacts = {len(all_samples)}")

    finally:
        if viewer:
            viewer.close()

    # Save research-friendly dataset
    out_npz = OUT_DIR / "cube_contacts_structured.npz"
    save_contacts_npz(out_npz, all_samples)
    print("Saved structured contacts:", out_npz)

    # Also save raw positions for quick plotting/backward compatibility
    raw_pts = np.stack([s.pos_world for s in all_samples], axis=0) if all_samples else np.zeros((0, 3), dtype=np.float64)
    out_npy = OUT_DIR / "cube_contact_points.npy"
    np.save(out_npy, raw_pts)
    print("Saved raw contact points:", out_npy)


if __name__ == "__main__":
    cfg = Config(
        n_trials=3,
        seed=0,
        min_force_n=0.01,
    )
    run(cfg, visualize=True)
