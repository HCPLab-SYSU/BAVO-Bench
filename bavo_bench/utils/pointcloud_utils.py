import math
from dataclasses import dataclass
from typing import Optional

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - optional dependency for FPS
    torch = None

try:
    import pytorch3d.ops as torch3d_ops
except ImportError:  # pragma: no cover - optional dependency for FPS
    torch3d_ops = None


@dataclass
class PointCloudObservation:
    point_cloud: np.ndarray
    depth: np.ndarray
    rgb: Optional[np.ndarray]
    camera_pose_world: np.ndarray
    reference_pose_world: np.ndarray


def _reshape_rotation(matrix_like) -> np.ndarray:
    return np.asarray(matrix_like, dtype=np.float64).reshape(3, 3)


def pose_matrix(position, rotation) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _reshape_rotation(rotation)
    transform[:3, 3] = np.asarray(position, dtype=np.float64)
    return transform


def get_body_pose(physics, body_name: str) -> np.ndarray:
    return pose_matrix(
        physics.named.data.xpos[body_name],
        physics.named.data.xmat[body_name],
    )


def get_site_pose(physics, site_name: str) -> np.ndarray:
    return pose_matrix(
        physics.named.data.site_xpos[site_name],
        physics.named.data.site_xmat[site_name],
    )


def get_camera_intrinsics(
    height: int,
    width: int,
    fovy_degrees: float,
) -> np.ndarray:
    fovy = math.radians(fovy_degrees)
    focal = height / (2.0 * math.tan(fovy / 2.0))
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    intrinsics = np.array(
        [
            [focal, 0.0, cx],
            [0.0, focal, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return intrinsics


def depth_to_camera_points(
    depth: np.ndarray,
    intrinsics: np.ndarray,
) -> np.ndarray:
    height, width = depth.shape
    fx = intrinsics[0, 0]
    fy = intrinsics[1, 1]
    cx = intrinsics[0, 2]
    cy = intrinsics[1, 2]

    v_coords, u_coords = np.indices((height, width), dtype=np.float64)
    z = depth.astype(np.float64)
    valid = np.isfinite(z) & (z > 0)
    if not np.any(valid):
        return np.zeros((0, 3), dtype=np.float32)

    x = (u_coords - cx) * z / fx
    y = (v_coords - cy) * z / fy
    points = np.stack([x[valid], y[valid], z[valid]], axis=-1)
    return points.astype(np.float32)


def transform_points(points: np.ndarray, transform: np.ndarray) -> np.ndarray:
    if len(points) == 0:
        return points.astype(np.float32)
    homogeneous = np.concatenate(
        [points.astype(np.float64), np.ones((len(points), 1), dtype=np.float64)],
        axis=1,
    )
    transformed = (transform @ homogeneous.T).T[:, :3]
    return transformed.astype(np.float32)


def crop_point_cloud(
    point_cloud: np.ndarray,
    crop_min: Optional[np.ndarray] = None,
    crop_max: Optional[np.ndarray] = None,
) -> np.ndarray:
    if len(point_cloud) == 0:
        return point_cloud.astype(np.float32)

    xyz = point_cloud[:, :3]
    mask = np.ones(len(point_cloud), dtype=bool)
    if crop_min is not None:
        mask &= np.all(xyz >= np.asarray(crop_min, dtype=np.float32), axis=1)
    if crop_max is not None:
        mask &= np.all(xyz <= np.asarray(crop_max, dtype=np.float32), axis=1)
    return point_cloud[mask].astype(np.float32)


def sample_point_cloud(
    point_cloud: np.ndarray,
    num_points: int,
    method: str = "fps",
) -> np.ndarray:
    if point_cloud.shape[0] == 0:
        return np.zeros((num_points, point_cloud.shape[1]), dtype=np.float32)

    if point_cloud.shape[0] <= num_points:
        pad = np.zeros(
            (num_points - point_cloud.shape[0], point_cloud.shape[1]),
            dtype=np.float32,
        )
        return np.concatenate([point_cloud.astype(np.float32), pad], axis=0)

    if method == "uniform":
        indices = np.random.choice(point_cloud.shape[0], num_points, replace=False)
        return point_cloud[indices].astype(np.float32)

    if method == "fps" and (torch3d_ops is None or torch is None):
        raise ImportError(
            "sampling_method='fps' requires pytorch3d and torch to be installed "
            "in the current environment. Install pytorch3d or use "
            "sampling_method='uniform' instead."
        )

    if method != "fps":
        raise NotImplementedError(f"Unsupported point cloud sampling method: {method}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    points = torch.from_numpy(point_cloud.astype(np.float32)).unsqueeze(0).to(device)
    _, indices = torch3d_ops.sample_farthest_points(
        points=points[..., :3],
        K=num_points,
    )
    sampled_points = points[0, indices[0]]
    return sampled_points.cpu().numpy().astype(np.float32)


def sample_point_cloud_indices(
    point_cloud: np.ndarray,
    num_points: int,
    method: str = "fps",
) -> np.ndarray:
    """Return ordered source-row indices selected by the sampling method."""
    if num_points <= 0:
        raise ValueError("num_points must be positive")
    point_cloud = np.asarray(point_cloud)
    point_count = int(point_cloud.shape[0])
    if point_count <= num_points:
        return np.arange(point_count, dtype=np.int64)

    if method == "uniform":
        return np.random.choice(
            point_count, num_points, replace=False
        ).astype(np.int64)

    if method != "fps":
        raise NotImplementedError(
            f"Unsupported point cloud sampling method: {method}"
        )
    if torch3d_ops is None or torch is None:
        raise ImportError(
            "sampling_method='fps' requires pytorch3d and torch to be installed "
            "in the current environment. Install pytorch3d or use "
            "sampling_method='uniform' instead."
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    points = torch.from_numpy(
        np.asarray(point_cloud[:, :3], dtype=np.float32)
    ).unsqueeze(0).to(device)
    _, indices = torch3d_ops.sample_farthest_points(
        points=points,
        K=num_points,
    )
    return indices[0].cpu().numpy().astype(np.int64)


def select_and_sample_point_cloud(
    point_cloud: np.ndarray,
    selection_points: np.ndarray,
    crop_min: Optional[np.ndarray],
    crop_max: Optional[np.ndarray],
    num_points: int,
    method: str,
) -> np.ndarray:
    """Crop/sample in one frame while returning rows expressed in another."""
    point_cloud = np.asarray(point_cloud, dtype=np.float32)
    selection_points = np.asarray(selection_points, dtype=np.float32)
    if point_cloud.ndim != 2 or selection_points.ndim != 2:
        raise ValueError("point_cloud and selection_points must both be rank 2")
    if len(point_cloud) != len(selection_points):
        raise ValueError("point_cloud and selection_points must have equal rows")
    if selection_points.shape[1] < 3:
        raise ValueError("selection_points must contain XYZ")

    xyz = selection_points[:, :3]
    mask = np.all(np.isfinite(xyz), axis=1)
    if crop_min is not None:
        mask &= np.all(
            xyz >= np.asarray(crop_min, dtype=np.float32), axis=1
        )
    if crop_max is not None:
        mask &= np.all(
            xyz <= np.asarray(crop_max, dtype=np.float32), axis=1
        )
    candidates = point_cloud[mask]
    selection_candidates = selection_points[mask]
    indices = sample_point_cloud_indices(
        selection_candidates,
        num_points=num_points,
        method=method,
    )
    sampled = candidates[indices]
    if len(sampled) < num_points:
        padding = np.zeros(
            (num_points - len(sampled), point_cloud.shape[1]),
            dtype=np.float32,
        )
        sampled = np.concatenate([sampled, padding], axis=0)
    return sampled.astype(np.float32)


def render_point_cloud_observation(
    physics,
    camera_name: str,
    camera_site_name: str,
    reference_body_name: str,
    height: int,
    width: int,
    pointcloud_frame: str = "reference",
    point_selection_frame: Optional[str] = None,
    crop_min: Optional[np.ndarray] = None,
    crop_max: Optional[np.ndarray] = None,
    num_points: int = 1024,
    sampling_method: str = "fps",
    include_rgb: bool = False,
) -> PointCloudObservation:
    depth_render = physics.render(
        height=height,
        width=width,
        camera_id=camera_name,
        depth=True,
    )
    rgb = None
    if isinstance(depth_render, tuple):
        rgb, depth = depth_render
    else:
        depth = depth_render
    if include_rgb and rgb is None:
        rgb = physics.render(
            height=height,
            width=width,
            camera_id=camera_name,
        )
    intrinsics = get_camera_intrinsics(
        height=height,
        width=width,
        fovy_degrees=float(physics.named.model.cam_fovy[camera_name]),
    )
    camera_pose_world = get_site_pose(physics, camera_site_name)
    reference_pose_world = get_body_pose(physics, reference_body_name)

    valid_depth = np.isfinite(depth) & (depth > 0)
    points_camera = depth_to_camera_points(depth, intrinsics)
    if pointcloud_frame == "camera":
        points_target = points_camera
    elif pointcloud_frame == "reference":
        points_world = transform_points(points_camera, camera_pose_world)
        points_target = transform_points(points_world, np.linalg.inv(reference_pose_world))
    else:
        raise ValueError(
            f"Unsupported pointcloud_frame: {pointcloud_frame}. "
            "Expected one of ['camera', 'reference']."
        )

    if include_rgb:
        colors = np.asarray(rgb, dtype=np.float32).reshape(-1, 3)[valid_depth.reshape(-1)]
        if colors.size and colors.max() > 1.0:
            colors /= 255.0
        points_target = np.concatenate([points_target, colors], axis=-1)

    if point_selection_frame is None or point_selection_frame == pointcloud_frame:
        points_target = crop_point_cloud(points_target, crop_min=crop_min, crop_max=crop_max)
        points_target = sample_point_cloud(
            points_target,
            num_points=num_points,
            method=sampling_method,
        )
    else:
        # Crop and FPS may be owned by a different frame than the one the XYZ is
        # stored in: the selection frame decides which rows survive and in what
        # order, while the returned coordinates stay in ``pointcloud_frame``.
        # Needed to reproduce camera-frame episodes whose crop and FPS were
        # defined on reference-frame coordinates.
        if point_selection_frame == "camera":
            points_selection = points_camera
        elif point_selection_frame == "reference":
            points_selection = transform_points(
                transform_points(points_camera, camera_pose_world),
                np.linalg.inv(reference_pose_world),
            )
        else:
            raise ValueError(
                f"Unsupported point_selection_frame: {point_selection_frame}. "
                "Expected one of ['camera', 'reference']."
            )
        points_target = select_and_sample_point_cloud(
            point_cloud=points_target,
            selection_points=points_selection,
            crop_min=crop_min,
            crop_max=crop_max,
            num_points=num_points,
            method=sampling_method,
        )

    return PointCloudObservation(
        point_cloud=points_target,
        depth=depth.astype(np.float32),
        rgb=rgb if include_rgb else None,
        camera_pose_world=camera_pose_world.astype(np.float32),
        reference_pose_world=reference_pose_world.astype(np.float32),
    )
