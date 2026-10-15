import os

import numpy as np

from bavo_bench.env.tasks.cube_transfer_plate.base import TASK_XML_DIR
from bavo_bench.env.tasks.cube_transfer_plate.occlusion import (
    CubeTransferPlateOcclusionEnv,
)


class CubeTransferPlateRandomOcclusionEnv(CubeTransferPlateOcclusionEnv):
    """Re-occlude the current task target at seeded, time-based events."""

    XML = os.path.join(TASK_XML_DIR, "random_occlusion.xml")
    OCCLUSION_SWITCH_WINDOWS = (
        (60, 130),
        (130, 200),
        (200, 270),
        (270, 340),
        (340, 410),
        (410, 480),
    )
    MIN_OCCLUSION_SWITCH_GAP = 60

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._occlusion_switch_steps = np.zeros(
            len(self.OCCLUSION_SWITCH_WINDOWS),
            dtype=np.int64,
        )
        self._occlusion_event_index = 0
        event_count = len(self.OCCLUSION_SWITCH_WINDOWS) + 1
        self._occlusion_target_names = [None] * event_count
        self._occlusion_candidate_sets = []

    def _sample_switch_steps(self):
        switch_steps = []
        previous_step = None
        for low, high in self.OCCLUSION_SWITCH_WINDOWS:
            minimum_step = low
            if previous_step is not None:
                minimum_step = max(
                    minimum_step,
                    previous_step + self.MIN_OCCLUSION_SWITCH_GAP + 1,
                )
            if minimum_step > high:
                raise ValueError(
                    "Occlusion switch windows cannot satisfy the minimum gap: "
                    f"previous_step={previous_step}, window=({low}, {high}), "
                    f"minimum_gap={self.MIN_OCCLUSION_SWITCH_GAP}"
                )
            step = int(self.np_random.integers(minimum_step, high + 1))
            switch_steps.append(step)
            previous_step = step
        return np.asarray(switch_steps, dtype=np.int64)

    def _sample_episode_occlusion_spec(self):
        self._occlusion_switch_steps = self._sample_switch_steps()
        event_count = len(self._occlusion_switch_steps) + 1
        self._occlusion_target_names = [None] * event_count
        self._occlusion_candidate_sets = [
            self._sample_occluder_candidates()
            for _ in range(event_count)
        ]

    def _advance_task_stage(self, reward):
        previous_stage = self._occlusion_stage
        if self._occlusion_stage == 0 and reward >= 2:
            self._occlusion_stage = 1
        elif self._occlusion_stage == 1 and reward >= 4:
            self._occlusion_stage = 2
        return self._occlusion_stage != previous_stage

    def _target_name_for_current_stage(self):
        return "cube" if self._occlusion_stage < 2 else "plate"

    def _place_occluder_for_event(self, event_index):
        target_name = self._target_name_for_current_stage()
        placed = self._place_occluder(
            target_geom=self._target_geom_for_stage(self._occlusion_stage),
            placement_label=(
                f"event={event_index}, task_stage={self._occlusion_stage}, "
                f"target={target_name}"
            ),
            candidates=self._occlusion_candidate_sets[event_index],
        )
        if placed:
            self._occlusion_target_names[event_index] = target_name
        return placed

    def _random_occlusion_info(self):
        info = self._occlusion_info()
        info.update({
            "occlusion_event_index": self._occlusion_event_index,
            "occlusion_target": self._occlusion_target_names[
                self._occlusion_event_index
            ],
            "occlusion_switch_steps": self._occlusion_switch_steps.copy(),
        })
        return info

    def reset(self, seed=None, options=None):
        for reset_attempt in range(self.MAX_RESET_ATTEMPTS):
            current_seed = seed if reset_attempt == 0 else None
            _, info = super().reset(seed=current_seed, options=options)
            self._reset_occlusion_evaluation()
            self._occlusion_stage = 0
            self._occlusion_event_index = 0
            self._occluder_placement_failed = False
            self._sample_episode_occlusion_spec()
            if self._place_occluder_for_event(self._occlusion_event_index):
                self._start_cvg_event(
                    event_index=0,
                    target_geom=self._target_geom_for_stage(self._occlusion_stage),
                    target_name=self._occlusion_target_names[0],
                )
                info.update(self._random_occlusion_info())
                return self.get_obs(), info
            print(
                f"Initial random occluder placement failed "
                f"(reset attempt {reset_attempt + 1}/{self.MAX_RESET_ATTEMPTS}): "
                f"{self._last_placement_failure_reason}",
                flush=True,
            )
        raise RuntimeError(
            f"Failed to place the initial random occluder after "
            f"{self.MAX_RESET_ATTEMPTS} environment resets"
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self._occluder_placement_failed = False
        if self._advance_task_stage(reward):
            observation = self.get_obs()

        next_event = self._occlusion_event_index + 1
        event_started = False
        if (
            next_event < len(self._occlusion_candidate_sets)
            and self.step_count >= self._occlusion_switch_steps[next_event - 1]
        ):
            if self._place_occluder_for_event(next_event):
                self._occlusion_event_index = next_event
                self._start_cvg_event(
                    event_index=next_event,
                    target_geom=self._target_geom_for_stage(self._occlusion_stage),
                    target_name=self._occlusion_target_names[next_event],
                )
                event_started = True
                observation = self.get_obs()
            else:
                self._occluder_placement_failed = True

        if not event_started:
            self._sample_cvg_event()

        info.update(self._random_occlusion_info())
        if self._occluder_placement_failed:
            truncated = True
            print(
                f"Random occluder placement failed: "
                f"{self._last_placement_failure_reason}",
                flush=True,
            )
        return observation, reward, terminated, truncated, info
