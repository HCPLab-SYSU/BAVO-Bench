import os

from bavo_bench.env.tasks.slide_pickup.base import TASK_XML_DIR
from bavo_bench.env.tasks.slide_pickup.occlusion import (
    SlidePickupOcclusionEnv,
)


class SlidePickupStageOcclusionEnv(SlidePickupOcclusionEnv):
    """Reposition the occluder when the slide-pickup stage advances."""

    XML = os.path.join(TASK_XML_DIR, "stage_occlusion.xml")

    def _advance_occlusion_stage(self, reward):
        next_stage = self._occlusion_stage
        if self._occlusion_stage == 0 and reward >= 1:
            next_stage = 1
        elif self._occlusion_stage == 1 and reward >= 2:
            next_stage = 2

        if next_stage != self._occlusion_stage:
            if self._place_occluder_for_stage(next_stage):
                self._occlusion_stage = next_stage
                self._start_cvg_event(
                    event_index=self._occlusion_event_index + 1,
                    target_geom=self._target_geom_for_stage(next_stage),
                )
                return True
            self._occluder_placement_failed = True
        return False

    def reset(self, seed=None, options=None):
        for reset_attempt in range(self.MAX_RESET_ATTEMPTS):
            current_seed = seed if reset_attempt == 0 else None
            _, info = super().reset(seed=current_seed, options=options)
            self._reset_occlusion_evaluation()
            self._occlusion_stage = 0
            self._occluder_placement_failed = False
            if self._place_occluder_for_stage(self._occlusion_stage):
                self._start_cvg_event(
                    event_index=0,
                    target_geom=self._target_geom_for_stage(self._occlusion_stage),
                )
                info.update(self._occlusion_info())
                return self.get_obs(), info
            print(
                f"Initial slide-pickup occluder placement failed "
                f"(reset attempt {reset_attempt + 1}/{self.MAX_RESET_ATTEMPTS}): "
                f"{self._last_placement_failure_reason}",
                flush=True,
            )
        raise RuntimeError(
            f"Failed to place the initial occluder after "
            f"{self.MAX_RESET_ATTEMPTS} environment resets"
        )

    def step(self, action):
        observation, reward, terminated, truncated, info = super().step(action)
        self._occluder_placement_failed = False
        event_started = self._advance_occlusion_stage(reward)
        if event_started:
            observation = self.get_obs()
        else:
            self._sample_cvg_event()
        info.update(self._occlusion_info())
        if self._occluder_placement_failed:
            truncated = True
            print(
                f"Slide-pickup occluder placement failed: "
                f"{self._last_placement_failure_reason}",
                flush=True,
            )
        return observation, reward, terminated, truncated, info
