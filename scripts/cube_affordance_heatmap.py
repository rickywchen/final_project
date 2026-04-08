from __future__ import annotations

from pathlib import Path
import numpy as np


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    # q = (w,x,y,z)
    w, x, y, z = q
    # Standard quaternion to rotation matrix
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def world_to_cube_local(p_world: np.ndarray, cube_pos: np.ndarray, cube_quat: np.ndarray) -> np.ndarray:
    R = quat_to_rotmat(cube_quat)
    return R.T @ (p_world - cube_pos)


def cube_face_and_uv(p_local: np.ndarray, half_extent: float) -> tuple[int, float, float]:
    """
    Returns (face_id, u, v) where face_id in [0..5]
    Faces:
      0: +X, 1: -X, 2: +Y, 3: -Y, 4: +Z, 5: -Z

    u,v are in [-1, +1] on that face.
    """
    x, y, z = p_local
    ax, ay, az = abs(x), abs(y), abs(z)

    # Determine which face the point is closest to
    if ax >= ay and ax >= az:
        if x >= 0:
            face = 0  # +X
            u, v = y / half_extent, z / half_extent
        else:
            face = 1  # -X
            u, v = -y / half_extent, z / half_extent
    elif ay >= ax and ay >= az:
        if y >= 0:
            face = 2  # +Y
            u, v = -x / half_extent, z / half_extent
        else:
            face = 3  # -Y
            u, v = x / half_extent, z / half_extent
    else:
        if z >= 0:
            face = 4  # +Z
            u, v = x / half_extent, y / half_extent
        else:
            face = 5  # -Z
            u, v = x / half_extent, -y / half_extent

    # Clamp to be safe
    u = float(np.clip(u, -1.0, 1.0))
    v = float(np.clip(v, -1.0, 1.0))
    return face, u, v


def bin_heatmaps(
    pos_world: np.ndarray,
    cube_pos: np.ndarray,
    cube_quat: np.ndarray,
    force_n: np.ndarray,
    half_extent: float,
    H: int = 64,
    W: int = 64,
    weight_mode: str = "force",  # "count" or "force"
) -> np.ndarray:
    heat = np.zeros((6, H, W), dtype=np.float64)

    for i in range(pos_world.shape[0]):
        pL = world_to_cube_local(pos_world[i], cube_pos[i], cube_quat[i])
        face, u, v = cube_face_and_uv(pL, half_extent)

        # map u,v in [-1,1] -> pixel
        x = int((u * 0.5 + 0.5) * (W - 1))
        y = int((v * 0.5 + 0.5) * (H - 1))

        w = 1.0 if weight_mode == "count" else float(force_n[i])
        heat[face, y, x] += w

    return heat


def gaussian_blur_2d(img: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    # Simple separable gaussian blur without extra deps.
    if sigma <= 0:
        return img

    radius = int(3 * sigma)
    xs = np.arange(-radius, radius + 1)
    kernel = np.exp(-(xs**2) / (2 * sigma**2))
    kernel = kernel / (kernel.sum() + 1e-12)

    # Convolve rows then cols
    tmp = np.zeros_like(img, dtype=np.float64)
    out = np.zeros_like(img, dtype=np.float64)

    # rows
    for r in range(img.shape[0]):
        tmp[r] = np.convolve(img[r], kernel, mode="same")
    # cols
    for c in range(img.shape[1]):
        out[:, c] = np.convolve(tmp[:, c], kernel, mode="same")
    return out


def smooth_heatmaps(heat: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    sm = np.zeros_like(heat)
    for f in range(6):
        sm[f] = gaussian_blur_2d(heat[f], sigma=sigma)
    return sm


def main():
    root = Path(__file__).resolve().parents[1]
    inp = root / "assets" / "cache" / "cube_contacts_structured.npz"
    out = root / "assets" / "cache" / "cube_heatmaps.npz"

    data = np.load(inp)
    pos_world = data["pos_world"]
    cube_pos = data["cube_pos"]
    cube_quat = data["cube_quat"]
    force_n = data["force_n"]

    # Your cube size in XML: size="0.03 0.03 0.03"
    # That means half-extent = 0.03
    half_extent = 0.03

    heat = bin_heatmaps(
        pos_world=pos_world,
        cube_pos=cube_pos,
        cube_quat=cube_quat,
        force_n=force_n,
        half_extent=half_extent,
        H=64,
        W=64,
        weight_mode="force",
    )

    heat_smooth = smooth_heatmaps(heat, sigma=1.0)

    # Normalize per-face for visualization convenience
    heat_norm = heat_smooth.copy()
    for f in range(6):
        m = heat_norm[f].max()
        if m > 0:
            heat_norm[f] /= m

    np.savez_compressed(out, heat_raw=heat, heat_smooth=heat_smooth, heat_norm=heat_norm)
    print("Saved:", out)


if __name__ == "__main__":
    main()
