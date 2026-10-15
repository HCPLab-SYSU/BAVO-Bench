from __future__ import annotations

from copy import deepcopy
from typing import Any

import mujoco
import numpy as np


class CounterfactualVisibilityTracker:
    """Evaluator-only tracker for camera-attributable visibility gain."""

    def __init__(
        self,
        env,
        *,
        enabled: bool = False,
        horizon: int = 60,
        render_height: int = 240,
        render_width: int = 304,
    ):
        if horizon <= 0:
            raise ValueError(f"CVG horizon must be positive, got {horizon}")
        self.env = env
        self.enabled = bool(enabled)
        self.horizon = int(horizon)
        self.render_height = int(render_height)
        self.render_width = int(render_width)
        self.reset()

    def reset(self) -> None:
        self._events: list[dict[str, Any]] = []
        self._active_event: dict[str, Any] | None = None

    def start_event(
        self,
        *,
        event_index: int,
        tau: int,
        target_geom,
        target_name: str | None,
        occlusion_stage: int,
    ) -> None:
        if not self.enabled:
            return
        if self._active_event is not None:
            self._invalidate_active_event("superseded_by_event")

        physics = self.env.physics
        target_geom_id = int(physics.bind(target_geom).element_id)
        if target_name is None:
            target_name = physics.model.id2name(target_geom_id, "geom")
        camera_position, camera_rotation = self._camera_world_pose()
        event = {
            "event_index": int(event_index),
            "tau": int(tau),
            "occlusion_stage": int(occlusion_stage),
            "target": str(target_name),
            "target_geom_id": target_geom_id,
            "reference_camera_position": camera_position.tolist(),
            "reference_camera_rotation": camera_rotation.tolist(),
            "complete": False,
            "invalid_reason": None,
            "active_visibility": [],
            "frozen_visibility": [],
            "gain": [],
            "active_occ_pixels": [],
            "active_no_occ_pixels": [],
            "frozen_occ_pixels": [],
            "frozen_no_occ_pixels": [],
            "denominator_zero_frames": 0,
            "cvg": None,
            "stage_history": [],
            "milestone_history": [],
        }
        self._events.append(event)
        self._active_event = event
        self.sample()

    def sample(self) -> None:
        if not self.enabled or self._active_event is None:
            return
        event = self._active_event
        event["stage_history"].append(
            int(getattr(self.env, "_occlusion_stage", event["occlusion_stage"]))
        )
        event["milestone_history"].append(
            int(getattr(self.env, "_current_milestone", 0))
        )
        active, frozen = self._paired_visibility(
            target_geom_id=event["target_geom_id"],
            frozen_camera_position=np.asarray(
                event["reference_camera_position"], dtype=np.float64
            ),
            frozen_camera_rotation=np.asarray(
                event["reference_camera_rotation"], dtype=np.float64
            ),
        )
        active_visibility, active_occ, active_no_occ = active
        frozen_visibility, frozen_occ, frozen_no_occ = frozen
        gain = active_visibility - frozen_visibility

        event["active_visibility"].append(float(active_visibility))
        event["frozen_visibility"].append(float(frozen_visibility))
        event["gain"].append(float(gain))
        event["active_occ_pixels"].append(int(active_occ))
        event["active_no_occ_pixels"].append(int(active_no_occ))
        event["frozen_occ_pixels"].append(int(frozen_occ))
        event["frozen_no_occ_pixels"].append(int(frozen_no_occ))
        event["denominator_zero_frames"] += int(
            active_no_occ == 0 or frozen_no_occ == 0
        )

        if len(event["gain"]) == self.horizon:
            event["complete"] = True
            event["cvg"] = float(np.mean(event["gain"]))
            self._active_event = None

    def metrics(self) -> dict[str, Any]:
        events = deepcopy(self._events)
        for event in events:
            if not event["complete"] and event["invalid_reason"] is None:
                event["invalid_reason"] = "insufficient_horizon"
        valid_events = [event for event in events if event["complete"]]
        cvg = (
            float(np.mean([event["cvg"] for event in valid_events]))
            if valid_events
            else None
        )
        completion_milestones = tuple(
            int(value)
            for value in getattr(
                self.env,
                "CVG_STAGE_COMPLETION_MILESTONES",
                (1, 2, int(getattr(self.env, "max_reward", 1))),
            )
        )
        deepest_milestone = int(getattr(self.env, "_deepest_milestone", 0))
        strict_valid_events = []
        strict_invalid_reasons = {
            "incomplete": 0,
            "stage_changed_during_event": 0,
            "stage_not_completed": 0,
        }
        for event in events:
            event["m2_eligible"] = False
            if not event["complete"]:
                strict_invalid_reasons["incomplete"] += 1
                continue
            stage = int(event["occlusion_stage"])
            stage_history = set(int(value) for value in event["stage_history"])
            if stage_history != {stage}:
                strict_invalid_reasons["stage_changed_during_event"] += 1
                continue
            required = completion_milestones[min(stage, len(completion_milestones) - 1)]
            if deepest_milestone < required:
                strict_invalid_reasons["stage_not_completed"] += 1
                continue
            event["m2_eligible"] = True
            strict_valid_events.append(event)
        cvg_m2 = (
            float(np.mean([event["cvg"] for event in strict_valid_events]))
            if strict_valid_events
            else None
        )
        eligible_by_stage = {
            str(stage): sum(
                int(event["m2_eligible"] and int(event["occlusion_stage"]) == stage)
                for event in events
            )
            for stage in range(len(completion_milestones))
        }
        stage_completion = {
            str(stage): bool(
                deepest_milestone >= required
            )
            for stage, required in enumerate(completion_milestones)
        }
        for event in events:
            if event["complete"] and not event["m2_eligible"]:
                if set(int(value) for value in event["stage_history"]) != {
                    int(event["occlusion_stage"])
                }:
                    event["m2_invalid_reason"] = "stage_changed_during_event"
                elif deepest_milestone < completion_milestones[
                    min(int(event["occlusion_stage"]), len(completion_milestones) - 1)
                ]:
                    event["m2_invalid_reason"] = "stage_not_completed"
                else:
                    event["m2_invalid_reason"] = "not_eligible"
            elif event["m2_eligible"]:
                event["m2_invalid_reason"] = None
        return {
            "cvg_enabled": self.enabled,
            "cvg_horizon": self.horizon,
            "cvg": cvg,
            "cvg_m2": cvg_m2,
            "valid_cvg_events": len(valid_events),
            "valid_cvg_events_m2": len(strict_valid_events),
            "encountered_cvg_events": len(events),
            "cvg_m2_stage_completion_milestones": list(completion_milestones),
            "cvg_m2_stage_completion": stage_completion,
            "cvg_m2_eligible_events_by_stage": eligible_by_stage,
            "cvg_m2_invalid_event_counts": strict_invalid_reasons,
            "cvg_events": events,
        }

    def info(self) -> dict[str, Any]:
        valid_events = sum(event["complete"] for event in self._events)
        return {
            "cvg_enabled": self.enabled,
            "cvg_horizon": self.horizon,
            "valid_cvg_events": int(valid_events),
            "encountered_cvg_events": len(self._events),
        }

    def _invalidate_active_event(self, reason: str) -> None:
        if self._active_event is None:
            return
        self._active_event["invalid_reason"] = str(reason)
        self._active_event = None

    def _camera_world_pose(self) -> tuple[np.ndarray, np.ndarray]:
        model = self.env.physics.model
        data = self.env.physics.data
        camera_id = model.name2id(self.env.occlusion_camera, "camera")
        return (
            data.cam_xpos[camera_id].copy(),
            data.cam_xmat[camera_id].reshape(3, 3).copy(),
        )

    def _set_camera_world_pose(
        self,
        world_position: np.ndarray,
        world_rotation: np.ndarray,
    ) -> None:
        physics = self.env.physics
        model = physics.model
        data = physics.data
        camera_id = model.name2id(self.env.occlusion_camera, "camera")
        body_id = int(model.cam_bodyid[camera_id])
        body_position = data.xpos[body_id]
        body_rotation = data.xmat[body_id].reshape(3, 3)
        model.cam_pos[camera_id] = body_rotation.T @ (
            world_position - body_position
        )
        local_rotation = body_rotation.T @ world_rotation
        local_quaternion = np.empty(4, dtype=np.float64)
        mujoco.mju_mat2Quat(
            local_quaternion,
            np.ascontiguousarray(local_rotation.ravel()),
        )
        model.cam_quat[camera_id] = local_quaternion
        physics.forward()

    def _visible_pixels(self, target_geom_id: int) -> int:
        segmentation = self.env.physics.render(
            height=self.render_height,
            width=self.render_width,
            camera_id=self.env.occlusion_camera,
            segmentation=True,
        )
        target_mask = (
            (segmentation[..., 0] == target_geom_id)
            & (
                segmentation[..., 1]
                == int(mujoco.mjtObj.mjOBJ_GEOM)
            )
        )
        return int(np.count_nonzero(target_mask))

    def _visibility(self, target_geom_id: int) -> tuple[float, int, int]:
        physics = self.env.physics
        occluded_pixels = self._visible_pixels(target_geom_id)
        occluder_position = physics.bind(self.env.occluder_body).pos.copy()
        try:
            hidden_position = np.array([0.0, 0.0, -2.0], dtype=np.float64)
            physics.bind(self.env.occluder_body).pos = hidden_position
            physics.forward()
            no_occluder_pixels = self._visible_pixels(target_geom_id)
        finally:
            physics.bind(self.env.occluder_body).pos = occluder_position
            physics.forward()

        if no_occluder_pixels == 0:
            visibility = 0.0
        else:
            visibility = float(
                np.clip(occluded_pixels / no_occluder_pixels, 0.0, 1.0)
            )
        return visibility, occluded_pixels, no_occluder_pixels

    def _paired_visibility(
        self,
        *,
        target_geom_id: int,
        frozen_camera_position: np.ndarray,
        frozen_camera_rotation: np.ndarray,
    ) -> tuple[tuple[float, int, int], tuple[float, int, int]]:
        physics = self.env.physics
        model = physics.model
        camera_id = model.name2id(self.env.occlusion_camera, "camera")
        active_local_position = model.cam_pos[camera_id].copy()
        active_local_quaternion = model.cam_quat[camera_id].copy()
        try:
            active = self._visibility(target_geom_id)
            self._set_camera_world_pose(
                frozen_camera_position,
                frozen_camera_rotation,
            )
            frozen = self._visibility(target_geom_id)
        finally:
            model.cam_pos[camera_id] = active_local_position
            model.cam_quat[camera_id] = active_local_quaternion
            physics.forward()
        return active, frozen


class OcclusionEvaluationMixin:
    """Small integration surface shared by all occlusion task variants."""

    def _initialize_occlusion_evaluation(
        self,
        *,
        enable_cvg: bool,
        cvg_horizon: int,
    ) -> None:
        self._cvg_tracker = CounterfactualVisibilityTracker(
            self,
            enabled=enable_cvg,
            horizon=cvg_horizon,
            render_height=self.OCCLUSION_RENDER_HEIGHT,
            render_width=self.OCCLUSION_RENDER_WIDTH,
        )
        self._occlusion_event_index = 0
        self._occlusion_target_name: str | None = None

    def _reset_occlusion_evaluation(self) -> None:
        self._cvg_tracker.reset()
        self._occlusion_event_index = 0
        self._occlusion_target_name = None

    def _start_cvg_event(
        self,
        *,
        event_index: int,
        target_geom,
        target_name: str | None = None,
    ) -> None:
        self._occlusion_event_index = int(event_index)
        if target_name is None:
            target_geom_id = int(self.physics.bind(target_geom).element_id)
            target_name = self.physics.model.id2name(target_geom_id, "geom")
        self._occlusion_target_name = str(target_name)
        self._cvg_tracker.start_event(
            event_index=event_index,
            tau=self.step_count,
            target_geom=target_geom,
            target_name=target_name,
            occlusion_stage=self._occlusion_stage,
        )

    def _sample_cvg_event(self) -> None:
        self._cvg_tracker.sample()

    def _augment_occlusion_info(self, info: dict[str, Any]) -> dict[str, Any]:
        info.update(
            {
                "occlusion_event_index": self._occlusion_event_index,
                "occlusion_target": self._occlusion_target_name,
                **self._cvg_tracker.info(),
            }
        )
        return info

    def get_episode_metrics(self) -> dict[str, Any]:
        metrics = super().get_episode_metrics()
        metrics.update(self._cvg_tracker.metrics())
        return metrics
