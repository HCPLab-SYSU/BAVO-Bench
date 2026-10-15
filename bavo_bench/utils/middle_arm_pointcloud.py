from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np


@dataclass(frozen=True)
class SurfacePointSpec:
    body_name: str
    mesh_name: str
    count: int


MIDDLE_ARM_SURFACE_SPECS: Tuple[SurfacePointSpec, ...] = (
    SurfacePointSpec("middle_shoulder_link", "vx300s_2_shoulder", 3),
    SurfacePointSpec("middle_upper_arm_link", "vx300s_3_upper_arm", 5),
    SurfacePointSpec("middle_upper_forearm_link", "vx300s_4_upper_forearm", 5),
    SurfacePointSpec("middle_lower_forearm_link", "vx300s_5_lower_forearm", 4),
    SurfacePointSpec("middle_wrist_1_link", "vx300s_7dof_wrist_1", 3),
    SurfacePointSpec("middle_wrist_2_link", "vx300s_7dof_wrist_2", 3),
    SurfacePointSpec("vx300s_7dof_wrist_3_link", "vx300s_7dof_wrist_3", 4),
    SurfacePointSpec("vx300s_7dof_wrist_3_link", "zedm", 5),
)

DEFAULT_REFERENCE_BODY = "middle_base_link"
DEFAULT_SURFACE_POINTS_PATH = (
    Path(__file__).resolve().parents[1]
    / "assets"
    / "robot_pointcloud"
    / "middle_arm_surface_points_32.npz"
)
VISUAL_GEOM_GROUP = 2
MUJOCO_MESH_GEOM_TYPE = 7


def _as_scalar_string(value: np.ndarray) -> str:
    return str(np.asarray(value).item())


class MiddleArmPointCloud:
    """Transforms fixed, compiled mesh-local surface points with MuJoCo FK."""

    def __init__(
        self,
        physics,
        asset_path: Optional[Path] = None,
        reference_body_name: Optional[str] = None,
    ) -> None:
        self.physics = physics
        self.asset_path = Path(asset_path or DEFAULT_SURFACE_POINTS_PATH).resolve()
        if not self.asset_path.is_file():
            raise FileNotFoundError(f"Missing robot surface-point asset: {self.asset_path}")

        with np.load(str(self.asset_path), allow_pickle=False) as asset:
            self.points_local = np.asarray(asset["points_local"], dtype=np.float64)
            self.colors = np.asarray(asset["colors"], dtype=np.float32)
            self.body_names = np.asarray(asset["body_names"]).astype(str)
            self.mesh_names = np.asarray(asset["mesh_names"]).astype(str)
            asset_reference_body = _as_scalar_string(asset["reference_body_name"])
            self.schema_version = _as_scalar_string(asset["schema_version"])

        if self.points_local.ndim != 2 or self.points_local.shape[1] != 3:
            raise ValueError(
                f"Expected points_local with shape (N, 3), got {self.points_local.shape}"
            )
        expected_shape = (len(self.points_local),)
        if self.body_names.shape != expected_shape or self.mesh_names.shape != expected_shape:
            raise ValueError("Point, body-name, and mesh-name counts differ in robot asset")
        if self.colors.shape != (len(self.points_local), 3):
            raise ValueError(f"Expected colors with shape (N, 3), got {self.colors.shape}")
        if not np.isfinite(self.points_local).all() or not np.isfinite(self.colors).all():
            raise ValueError("Robot surface-point asset contains non-finite values")

        self.reference_body_name = reference_body_name or asset_reference_body
        self.reference_body_id = self._name2id(self.reference_body_name, "body")
        self._segments = self._resolve_segments()

    @property
    def num_points(self) -> int:
        return int(len(self.points_local))

    def _name2id(self, name: str, object_type: str) -> int:
        try:
            object_id = int(self.physics.model.name2id(name, object_type))
        except Exception as exc:
            raise KeyError(f"MuJoCo model has no {object_type} named {name!r}") from exc
        if object_id < 0:
            raise KeyError(f"MuJoCo model has no {object_type} named {name!r}")
        return object_id

    def _resolve_visual_mesh_geom(self, body_name: str, mesh_name: str) -> int:
        model = self.physics.model
        body_id = self._name2id(body_name, "body")
        mesh_id = self._name2id(mesh_name, "mesh")
        matches = []
        for geom_id in range(int(model.ngeom)):
            if int(model.geom_bodyid[geom_id]) != body_id:
                continue
            if int(model.geom_type[geom_id]) != MUJOCO_MESH_GEOM_TYPE:
                continue
            if int(model.geom_dataid[geom_id]) != mesh_id:
                continue
            if int(model.geom_group[geom_id]) != VISUAL_GEOM_GROUP:
                continue
            matches.append(geom_id)
        if len(matches) != 1:
            raise RuntimeError(
                "Expected exactly one visual mesh geom for "
                f"body={body_name!r}, mesh={mesh_name!r}; found {matches}"
            )
        return matches[0]

    def _resolve_segments(self):
        segments = []
        start = 0
        while start < self.num_points:
            body_name = self.body_names[start]
            mesh_name = self.mesh_names[start]
            end = start + 1
            while (
                end < self.num_points
                and self.body_names[end] == body_name
                and self.mesh_names[end] == mesh_name
            ):
                end += 1
            geom_id = self._resolve_visual_mesh_geom(body_name, mesh_name)
            segments.append((start, end, geom_id))
            start = end
        return tuple(segments)

    def points_in_reference_frame(self) -> np.ndarray:
        data = self.physics.data
        reference_position = np.asarray(data.xpos[self.reference_body_id], dtype=np.float64)
        reference_rotation = np.asarray(
            data.xmat[self.reference_body_id], dtype=np.float64
        ).reshape(3, 3)

        points_reference = np.empty_like(self.points_local)
        for start, end, geom_id in self._segments:
            geom_position = np.asarray(data.geom_xpos[geom_id], dtype=np.float64)
            geom_rotation = np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
            points_world = self.points_local[start:end] @ geom_rotation.T + geom_position
            points_reference[start:end] = (
                points_world - reference_position
            ) @ reference_rotation
        return points_reference.astype(np.float32)

    def point_cloud_in_reference_frame(self) -> np.ndarray:
        return np.concatenate(
            [self.points_in_reference_frame(), self.colors], axis=-1
        ).astype(np.float32)

