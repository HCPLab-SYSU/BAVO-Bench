import mujoco
import numpy as np
from gymnasium import spaces

from bavo_bench.env.evaluation import OcclusionEvaluationMixin
from bavo_bench.env.tasks.block_stack.base import BlockStackEnv


class BlockStackOcclusionEnv(OcclusionEvaluationMixin, BlockStackEnv):
    """Shared target-aware occluder placement for block-stack tasks."""

    ENV_STATE_DIM = BlockStackEnv.ENV_STATE_DIM + 11

    OCCLUSION_CAMERA = "zed_cam_right"
    CVG_STAGE_COMPLETION_MILESTONES = (2, 4, 5)
    OCCLUSION_RENDER_HEIGHT = 240
    OCCLUSION_RENDER_WIDTH = 304
    OCCLUDER_CLEARANCE = 0.015
    OCCLUDER_CLEARANCE_QUERY_DISTANCE = 0.25
    MAX_VISIBLE_TARGET_PIXELS = 2
    OCCLUDER_CAMERA_DISTANCE_RANGE = (0.08, 0.12)
    OCCLUDER_X_OFFSET_RANGE = (-0.05, 0.05)
    OCCLUDER_Z_OFFSET_RANGE = (-0.05, 0.05)
    OCCLUDER_RANDOM_CANDIDATES = 30
    OCCLUDER_CENTER_DISTANCES = (0.08, 0.10, 0.12)
    MAX_RESET_ATTEMPTS = 10

    def __init__(
        self,
        occlusion_camera=OCCLUSION_CAMERA,
        enable_cvg=False,
        cvg_horizon=60,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.physics.model.vis.quality.offsamples = 0
        self.occlusion_camera = occlusion_camera
        self.occluder_body = self.mjcf_root.find("body", "occluder")
        self.occluder_geom = self.mjcf_root.find("geom", "occluder")

        self._occlusion_stage = 0
        self._last_occlusion_ratio = 0.0
        self._last_visible_target_pixels = 0
        self._occluder_placement_failed = False
        self._last_placement_attempts = 0
        self._last_placement_failure_reason = ""
        self._last_closest_protected_geom = "none"
        self._occluder_geom_id = self.physics.model.name2id("occluder", "geom")
        self._arm_geom_ids = self._find_robot_collision_geoms(("left_", "right_"))
        self._arm_geom_ids.append(
            self.physics.model.name2id("middle_zed_collision_box", "geom")
        )

        self.observation_space_dict["environment_state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.ENV_STATE_DIM,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict(self.observation_space_dict)
        self._initialize_occlusion_evaluation(
            enable_cvg=enable_cvg,
            cvg_horizon=cvg_horizon,
        )

    def _body_belongs_to_arm(self, body_id, prefixes):
        while body_id > 0:
            body_name = self.physics.model.id2name(body_id, "body") or ""
            if body_name.startswith(prefixes):
                return True
            body_id = int(self.physics.model.body_parentid[body_id])
        return False

    def _find_robot_collision_geoms(self, prefixes):
        geom_ids = []
        model = self.physics.model
        for geom_id in range(model.ngeom):
            if geom_id == self._occluder_geom_id:
                continue
            if model.geom_contype[geom_id] == 0 and model.geom_conaffinity[geom_id] == 0:
                continue
            if self._body_belongs_to_arm(int(model.geom_bodyid[geom_id]), prefixes):
                geom_ids.append(geom_id)
        return geom_ids

    def get_environment_state(self):
        return np.concatenate([
            super().get_environment_state(),
            self.physics.bind(self.occluder_body).pos.copy(),
            self.physics.bind(self.occluder_body).quat.copy(),
            self.physics.bind(self.occluder_geom).size.copy(),
            np.array([self._occlusion_stage], dtype=np.float64),
        ])

    def set_state(self, state, environment_state):
        if len(environment_state) != self.ENV_STATE_DIM:
            raise ValueError(
                f"Expected environment_state with {self.ENV_STATE_DIM} values, "
                f"got {len(environment_state)}"
            )
        super().set_state(state, environment_state[:BlockStackEnv.ENV_STATE_DIM])
        self.physics.bind(self.occluder_body).pos = environment_state[16:19]
        self.physics.bind(self.occluder_body).quat = environment_state[19:23]
        self.physics.bind(self.occluder_geom).size = environment_state[23:26]
        self._occlusion_stage = int(round(environment_state[26]))
        self.physics.forward()

    def _target_name_for_stage(self, stage):
        if stage == 0:
            return self.BASE_BLOCK_NAME
        if stage == 1:
            return self.TOP_BLOCK_NAME
        return self.BASE_BLOCK_NAME

    def _target_geom_for_stage(self, stage):
        if stage == 0:
            return self.block_right_geom
        if stage == 1:
            return self.block_left_geom
        return self.block_right_geom

    def _camera_position(self):
        camera_id = self.physics.model.name2id(self.occlusion_camera, "camera")
        return self.physics.data.cam_xpos[camera_id].copy()

    @staticmethod
    def _billboard_quaternion(camera_pos, target_pos):
        local_y = target_pos - camera_pos
        norm = np.linalg.norm(local_y)
        if norm < 1e-8:
            raise RuntimeError("Camera and occlusion target have the same position")
        local_y /= norm

        world_up = np.array([0.0, 0.0, 1.0])
        local_x = np.cross(local_y, world_up)
        if np.linalg.norm(local_x) < 1e-6:
            local_x = np.array([1.0, 0.0, 0.0])
        else:
            local_x /= np.linalg.norm(local_x)
        local_z = np.cross(local_x, local_y)
        local_z /= np.linalg.norm(local_z)

        rotation = np.column_stack([local_x, local_y, local_z])
        quat = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, np.ascontiguousarray(rotation.ravel()))
        return quat, local_x, local_z

    def _target_visible_pixels(self, target_geom_id):
        segmentation = self.physics.render(
            height=self.OCCLUSION_RENDER_HEIGHT,
            width=self.OCCLUSION_RENDER_WIDTH,
            camera_id=self.occlusion_camera,
            segmentation=True,
        )
        target_mask = (
            (segmentation[..., 0] == target_geom_id)
            & (segmentation[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM))
        )
        return int(np.count_nonzero(target_mask))

    def _minimum_arm_clearance(self):
        from_to = np.empty(6, dtype=np.float64)
        min_distance = np.inf
        closest_geom_id = None
        for geom_id in self._arm_geom_ids:
            distance = mujoco.mj_geomDistance(
                self.physics.model.ptr,
                self.physics.data.ptr,
                self._occluder_geom_id,
                geom_id,
                self.OCCLUDER_CLEARANCE_QUERY_DISTANCE,
                from_to,
            )
            if distance < min_distance:
                min_distance = distance
                closest_geom_id = geom_id
        if closest_geom_id is not None:
            geom_name = self.physics.model.id2name(closest_geom_id, "geom")
            if geom_name is None:
                body_id = int(self.physics.model.geom_bodyid[closest_geom_id])
                body_name = self.physics.model.id2name(body_id, "body")
                geom_name = body_name or f"geom_{closest_geom_id}"
            self._last_closest_protected_geom = geom_name
        return min_distance

    def _set_occluder_pose(self, position, quaternion):
        self.physics.bind(self.occluder_body).pos = position
        self.physics.bind(self.occluder_body).quat = quaternion
        self.physics.forward()

    def _sample_occluder_candidates(self):
        random_candidate_count = (
            self.OCCLUDER_RANDOM_CANDIDATES
            - len(self.OCCLUDER_CENTER_DISTANCES)
        )
        random_candidates = np.column_stack([
            self.np_random.uniform(
                *self.OCCLUDER_CAMERA_DISTANCE_RANGE,
                size=random_candidate_count,
            ),
            self.np_random.uniform(
                *self.OCCLUDER_X_OFFSET_RANGE,
                size=random_candidate_count,
            ),
            self.np_random.uniform(
                *self.OCCLUDER_Z_OFFSET_RANGE,
                size=random_candidate_count,
            ),
        ])
        center_candidates = np.array([
            (camera_distance, 0.0, 0.0)
            for camera_distance in self.OCCLUDER_CENTER_DISTANCES
        ], dtype=np.float64)
        candidates = np.concatenate([random_candidates, center_candidates], axis=0)
        self.np_random.shuffle(candidates)
        return candidates

    def _place_occluder(self, target_geom, placement_label, candidates=None):
        self._last_placement_failure_reason = ""
        target_geom_id = self.physics.bind(target_geom).element_id

        original_pos = self.physics.bind(self.occluder_body).pos.copy()
        original_quat = self.physics.bind(self.occluder_body).quat.copy()
        self._set_occluder_pose(np.array([0.0, 0.0, -2.0]), original_quat)
        target_pixels = self._target_visible_pixels(target_geom_id)

        camera_pos = self._camera_position()
        target_pos = self.physics.bind(target_geom).xpos.copy()
        ray = target_pos - camera_pos
        target_distance = np.linalg.norm(ray)
        if target_distance < 1e-8:
            self._set_occluder_pose(original_pos, original_quat)
            self._last_placement_attempts = 0
            self._last_placement_failure_reason = (
                f"{placement_label}: camera and target positions coincide"
            )
            return False
        ray_direction = ray / target_distance
        quat, lateral_axis, vertical_axis = self._billboard_quaternion(
            camera_pos, target_pos
        )

        if candidates is None:
            candidates = self._sample_occluder_candidates()
        best_visible_pixels = None
        best_candidate = None
        best_clearance = -np.inf
        collision_rejections = 0
        visibility_rejections = 0
        behind_target_rejections = 0
        collision_geoms = {}
        center_candidate_results = []
        for attempt, (camera_distance, lateral, vertical) in enumerate(
            candidates,
            start=1,
        ):
            self._last_placement_attempts = attempt
            if camera_distance >= target_distance:
                behind_target_rejections += 1
                continue
            offset = lateral * lateral_axis + vertical * vertical_axis
            position = camera_pos + camera_distance * ray_direction + offset
            self._set_occluder_pose(position, quat)

            clearance = self._minimum_arm_clearance()
            best_clearance = max(best_clearance, clearance)
            is_center_candidate = (
                abs(lateral) < 1e-12
                and abs(vertical) < 1e-12
                and any(
                    abs(camera_distance - center_distance) < 1e-12
                    for center_distance in self.OCCLUDER_CENTER_DISTANCES
                )
            )
            if clearance < self.OCCLUDER_CLEARANCE:
                collision_rejections += 1
                geom_name = self._last_closest_protected_geom
                collision_geoms[geom_name] = collision_geoms.get(geom_name, 0) + 1
                if is_center_candidate:
                    center_candidate_results.append(
                        f"{camera_distance:.2f}m:collision("
                        f"{geom_name},{clearance:.4f}m)"
                    )
                continue

            visible_pixels = self._target_visible_pixels(target_geom_id)
            if best_visible_pixels is None or visible_pixels < best_visible_pixels:
                best_visible_pixels = visible_pixels
                best_candidate = (
                    camera_distance,
                    lateral,
                    vertical,
                    clearance,
                )
            if is_center_candidate:
                center_candidate_results.append(
                    f"{camera_distance:.2f}m:visible={visible_pixels},"
                    f"clearance={clearance:.4f}m"
                )
            if visible_pixels <= self.MAX_VISIBLE_TARGET_PIXELS:
                self._last_visible_target_pixels = visible_pixels
                self._last_occlusion_ratio = (
                    1.0
                    if target_pixels == 0
                    else 1.0 - visible_pixels / target_pixels
                )
                return True
            visibility_rejections += 1

        self._set_occluder_pose(original_pos, original_quat)
        self._last_visible_target_pixels = (
            -1 if best_visible_pixels is None else best_visible_pixels
        )
        self._last_occlusion_ratio = (
            0.0
            if best_visible_pixels is None or target_pixels == 0
            else 1.0 - best_visible_pixels / target_pixels
        )
        collision_summary = ",".join(
            f"{name}:{count}"
            for name, count in sorted(collision_geoms.items())
        ) or "none"
        best_clearance_text = (
            "none" if not np.isfinite(best_clearance) else f"{best_clearance:.4f}m"
        )
        best_candidate_text = (
            "none"
            if best_candidate is None
            else (
                f"d={best_candidate[0]:.4f}m,"
                f"x={best_candidate[1]:.4f}m,"
                f"z={best_candidate[2]:.4f}m,"
                f"clearance={best_candidate[3]:.4f}m"
            )
        )
        center_results_text = ";".join(center_candidate_results) or "none"
        self._last_placement_failure_reason = (
            f"{placement_label}, target_distance={target_distance:.4f}m, "
            f"target_pixels={target_pixels}, "
            f"attempts={self._last_placement_attempts}, "
            f"collision_rejections={collision_rejections}, "
            f"visibility_rejections={visibility_rejections}, "
            f"behind_target_rejections={behind_target_rejections}, "
            f"collision_geoms={collision_summary}, "
            f"best_clearance={best_clearance_text}, "
            f"required_clearance={self.OCCLUDER_CLEARANCE:.4f}m, "
            f"best_visible_pixels={self._last_visible_target_pixels}, "
            f"best_occlusion_ratio={self._last_occlusion_ratio:.4f}, "
            f"allowed_visible_pixels={self.MAX_VISIBLE_TARGET_PIXELS}, "
            f"best_candidate=({best_candidate_text}), "
            f"center_candidates=[{center_results_text}]"
        )
        return False

    def _place_occluder_for_stage(self, stage):
        return self._place_occluder(
            target_geom=self._target_geom_for_stage(stage),
            placement_label=(
                f"stage={stage}, target={self._target_name_for_stage(stage)}"
            ),
        )

    def _occlusion_info(self):
        return self._augment_occlusion_info({
            "occlusion_stage": self._occlusion_stage,
            "occlusion_target": self._target_name_for_stage(self._occlusion_stage),
            "occlusion_ratio": self._last_occlusion_ratio,
            "visible_target_pixels": self._last_visible_target_pixels,
            "placement_attempts": self._last_placement_attempts,
            "occluder_placement_failed": self._occluder_placement_failed,
            "occluder_placement_failure_reason": self._last_placement_failure_reason,
        })
