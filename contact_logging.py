from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
import mujoco


@dataclass(frozen=True)
class ContactSample:
    pos_world: np.ndarray        # (3,)
    normal_world: np.ndarray     # (3,)
    force_n: float               # scalar (approx magnitude of normal+tangent)
    other_geom_id: int           # the non-object geom in the contact pair
    other_body_id: int           # the body that geom belongs to
    cube_pos_world: np.ndarray   # (3,)
    cube_quat_world: np.ndarray  # (4,) (w,x,y,z)
    trial_index: int
    timestamp: float


def quat_from_yaw(yaw_rad: float) -> np.ndarray:
    # MuJoCo quats are (w, x, y, z). Yaw about z-axis.
    half = yaw_rad * 0.5
    return np.array([math.cos(half), 0.0, 0.0, math.sin(half)], dtype=np.float64)


def get_body_pose_world(model: mujoco.MjModel, data: mujoco.MjData, body_id: int) -> tuple[np.ndarray, np.ndarray]:
    pos = data.xpos[body_id].copy()
    quat = data.xquat[body_id].copy()
    return pos, quat


def contact_force_mag(model: mujoco.MjModel, data: mujoco.MjData, contact_index: int) -> float:
    """
    Returns magnitude of the first 3 components (normal + tangents) in the contact frame.
    Used to filter out tiny jitter contacts.
    """
    arr = np.zeros(6, dtype=np.float64)
    mujoco.mj_contactForce(model, data, contact_index, arr)
    return float(np.linalg.norm(arr[:3]))


def contact_normal_world(contact: mujoco.MjContact) -> np.ndarray:
    """
    MuJoCo stores a 3x3 contact frame in contact.frame (flattened).
    The first column is the contact normal direction in world coordinates.
    """
    frame = np.array(contact.frame, dtype=np.float64).reshape(3, 3)
    n = frame[:, 0]
    n_norm = np.linalg.norm(n) + 1e-12
    return n / n_norm


def get_contacts_for_object_filtered_by_other_bodies(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_geom_id: int,
    allowed_other_body_ids: set[int],
    min_force_n: float,
    max_per_step: int,
    object_body_id: int,
    trial_index: int,
    debug_nonfinger: bool = False,
    debug_limit: int = 10,
) -> list[ContactSample]:
    """
    Extract contacts involving object_geom_id (cube geom) and keep only contacts where the
    OTHER geom belongs to one of allowed_other_body_ids (left_finger/right_finger).

    Debug:
      - If debug_nonfinger=True, it will print up to debug_limit examples of
        non-finger bodies that are touching the object, to diagnose why no contacts are saved.
    """
    out: list[ContactSample] = []
    obj_pos, obj_quat = get_body_pose_world(model, data, object_body_id)

    debug_left = int(debug_limit)

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)

        # Contact must involve the object geom
        if g1 != object_geom_id and g2 != object_geom_id:
            continue

        other_gid = g2 if g1 == object_geom_id else g1
        other_bid = int(model.geom_bodyid[other_gid])

        # Keep only allowed bodies (fingers)
        if other_bid not in allowed_other_body_ids:
            if debug_nonfinger and debug_left > 0:
                bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, other_bid)
                gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, other_gid)
                print("Cube touched by NON-finger:", bname, "geom:", gname)
                debug_left -= 1
            continue

        # Filter tiny jitter contacts
        f = contact_force_mag(model, data, i)
        if f < min_force_n:
            continue

        p = np.array(c.pos, dtype=np.float64)
        n = contact_normal_world(c)

        out.append(
            ContactSample(
                pos_world=p,
                normal_world=n,
                force_n=f,
                other_geom_id=other_gid,
                other_body_id=other_bid,
                cube_pos_world=obj_pos,
                cube_quat_world=obj_quat,
                trial_index=int(trial_index),
                timestamp=float(data.time),
            )
        )

        if len(out) >= max_per_step:
            break

    return out
