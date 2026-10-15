import os

import numpy as np
from gymnasium import spaces

from bavo_bench.env.sim_config import XML_DIR
from bavo_bench.env.sim_env import AVAlohaEnv

TASK_XML_DIR = os.path.join(XML_DIR, "tasks", "block_stack")


class BlockStackEnv(AVAlohaEnv):
    """Move the right block to the center, then stack the left block on it."""

    XML = os.path.join(TASK_XML_DIR, "task.xml")
    ENV_STATE_DIM = 16
    PROMPTS = ["stack yellow block on red block"]

    TOP_BLOCK_NAME = "block_left"
    BASE_BLOCK_NAME = "block_right"
    TABLE_GEOM_NAME = "table"

    BLOCK_HALF_SIZE = np.array([0.015, 0.015, 0.015], dtype=np.float64)
    BLOCK_START_Z = 0.02
    STACK_TARGET_X_RANGE = np.array([-0.08, 0.08], dtype=np.float64)
    STACK_TARGET_Y_RANGE = np.array([-0.15, 0.15], dtype=np.float64)
    LEFT_BLOCK_X_RANGE = np.array([-0.20, -0.10], dtype=np.float64)
    RIGHT_BLOCK_X_RANGE = np.array([0.10, 0.20], dtype=np.float64)
    BLOCK_Y_RANGE = np.array([0.05, 0.15], dtype=np.float64)

    PRESTACK_XY_TOLERANCE = 0.10
    PRESTACK_MIN_HEIGHT_DELTA = 0.025
    PRESTACK_MAX_HEIGHT_DELTA = 0.10
    STACK_SUCCESS_XY_TOLERANCE = 0.018
    STACK_SUCCESS_HEIGHT_DELTA = 0.024
    STACK_MAX_SPEED = 0.06
    STACK_STABLE_SUCCESS_STEPS = 2

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_reward = 5
        self.block_left_joint = self.mjcf_root.find("joint", "block_left_joint")
        self.block_right_joint = self.mjcf_root.find("joint", "block_right_joint")
        self.block_left_geom = self.mjcf_root.find("geom", self.TOP_BLOCK_NAME)
        self.block_right_geom = self.mjcf_root.find("geom", self.BASE_BLOCK_NAME)

        self._base_placed = False
        self._stable_stack_steps = 0

        self.observation_space_dict["environment_state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.ENV_STATE_DIM,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict(self.observation_space_dict)

    def get_environment_state(self):
        return np.concatenate([
            self.physics.bind(self.block_left_joint).qpos.copy(),
            self.physics.bind(self.block_right_joint).qpos.copy(),
            np.array(
                [float(self._base_placed), float(self._stable_stack_steps)],
                dtype=np.float64,
            ),
        ])

    def get_obs(self):
        obs = super().get_obs()
        obs["environment_state"] = self.get_environment_state()
        return obs

    def set_state(self, state, environment_state):
        if len(environment_state) != BlockStackEnv.ENV_STATE_DIM:
            raise ValueError(
                f"Expected environment_state with {BlockStackEnv.ENV_STATE_DIM} values, "
                f"got {len(environment_state)}"
            )
        AVAlohaEnv.set_state(self, state, environment_state)
        self.physics.bind(self.block_left_joint).qpos = environment_state[:7]
        self.physics.bind(self.block_right_joint).qpos = environment_state[7:14]
        self._base_placed = bool(round(environment_state[14]))
        self._stable_stack_steps = int(round(environment_state[15]))
        self.physics.forward()

    def reset(self, seed=None, options=None):
        _, info = AVAlohaEnv.reset(self, seed=seed, options=options)

        self.physics.bind(self.block_left_joint).qpos = self._sample_block_qpos(
            self.LEFT_BLOCK_X_RANGE
        )
        self.physics.bind(self.block_right_joint).qpos = self._sample_block_qpos(
            self.RIGHT_BLOCK_X_RANGE
        )

        self._base_placed = False
        self._stable_stack_steps = 0
        self.physics.forward()
        return self.get_obs(), info

    def _sample_block_qpos(self, x_range):
        for _ in range(50):
            position = np.array(
                [
                    self.np_random.uniform(*x_range),
                    self.np_random.uniform(*self.BLOCK_Y_RANGE),
                    self.BLOCK_START_Z,
                ],
                dtype=np.float64,
            )
            if not self._in_stack_target(position, margin=self.BLOCK_HALF_SIZE[0]):
                break
        else:
            raise RuntimeError("Could not sample a block position outside target area")

        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        return np.concatenate([position, quat])

    def _in_stack_target(self, position, margin=0.0):
        return (
            self.STACK_TARGET_X_RANGE[0] - margin
            <= position[0]
            <= self.STACK_TARGET_X_RANGE[1] + margin
            and self.STACK_TARGET_Y_RANGE[0] - margin
            <= position[1]
            <= self.STACK_TARGET_Y_RANGE[1] + margin
        )

    def _contact_pairs(self):
        contact_pairs = set()
        model = self.physics.model
        for contact_index in range(self.physics.data.ncon):
            geom_id_1 = self.physics.data.contact[contact_index].geom1
            geom_id_2 = self.physics.data.contact[contact_index].geom2
            geom_1 = model.id2name(geom_id_1, "geom")
            geom_2 = model.id2name(geom_id_2, "geom")
            if geom_1 is None or geom_2 is None:
                continue
            contact_pairs.add((geom_1, geom_2))
            contact_pairs.add((geom_2, geom_1))
        return contact_pairs

    def _block_touches_table(self, block_name, contact_pairs):
        return (block_name, self.TABLE_GEOM_NAME) in contact_pairs

    def _block_touches_gripper(self, block_name, gripper_prefix, contact_pairs):
        return any(
            geom_1 == block_name and geom_2.startswith(gripper_prefix)
            for geom_1, geom_2 in contact_pairs
        )

    def _blocks_touch(self, contact_pairs):
        return (self.TOP_BLOCK_NAME, self.BASE_BLOCK_NAME) in contact_pairs

    def _block_position(self, geom):
        return self.physics.bind(geom).xpos.copy()

    def _block_speed(self, joint):
        qvel = np.asarray(self.physics.bind(joint).qvel, dtype=np.float64)
        return float(np.linalg.norm(qvel))

    def _base_in_stack_target(self):
        return self._in_stack_target(self._block_position(self.block_right_geom))

    def _top_stack_alignment(self, xy_tolerance, height_delta):
        top_pos = self._block_position(self.block_left_geom)
        base_pos = self._block_position(self.block_right_geom)
        xy_distance = np.linalg.norm(top_pos[:2] - base_pos[:2])
        return (
            xy_distance <= xy_tolerance
            and top_pos[2] - base_pos[2] >= height_delta
        )

    def _top_in_prestack_region(self):
        top_pos = self._block_position(self.block_left_geom)
        base_pos = self._block_position(self.block_right_geom)
        xy_offset = np.abs(top_pos[:2] - base_pos[:2])
        height_delta = top_pos[2] - base_pos[2]
        return (
            np.all(xy_offset <= self.PRESTACK_XY_TOLERANCE)
            and self.PRESTACK_MIN_HEIGHT_DELTA
            <= height_delta
            <= self.PRESTACK_MAX_HEIGHT_DELTA
        )

    def get_reward(self):
        contact_pairs = self._contact_pairs()

        base_touch_right = self._block_touches_gripper(
            self.BASE_BLOCK_NAME,
            "right_",
            contact_pairs,
        )
        base_touch_left = self._block_touches_gripper(
            self.BASE_BLOCK_NAME,
            "left_",
            contact_pairs,
        )
        top_touch_left = self._block_touches_gripper(
            self.TOP_BLOCK_NAME,
            "left_",
            contact_pairs,
        )
        top_touch_right = self._block_touches_gripper(
            self.TOP_BLOCK_NAME,
            "right_",
            contact_pairs,
        )
        base_touch_table = self._block_touches_table(
            self.BASE_BLOCK_NAME,
            contact_pairs,
        )
        top_touch_table = self._block_touches_table(
            self.TOP_BLOCK_NAME,
            contact_pairs,
        )
        blocks_touch = self._blocks_touch(contact_pairs)
        base_ready = self._base_in_stack_target() and base_touch_table
        base_placed_now = base_ready and not base_touch_right
        if base_placed_now:
            self._base_placed = True

        top_in_prestack_region = (
            base_ready
            and self._base_placed
            and not top_touch_table
            and self._top_in_prestack_region()
        )
        stack_pose = (
            base_ready
            and blocks_touch
            and not top_touch_table
            and self._top_stack_alignment(
                self.STACK_SUCCESS_XY_TOLERANCE,
                self.STACK_SUCCESS_HEIGHT_DELTA,
            )
        )
        hands_released = not (
            base_touch_left
            or base_touch_right
            or top_touch_left
            or top_touch_right
        )
        blocks_slow = (
            self._block_speed(self.block_left_joint) <= self.STACK_MAX_SPEED
            and self._block_speed(self.block_right_joint) <= self.STACK_MAX_SPEED
        )
        if stack_pose and hands_released and blocks_slow:
            self._stable_stack_steps += 1
        else:
            self._stable_stack_steps = 0

        reward = 0
        if base_touch_right:
            reward = 1
        if base_placed_now:
            reward = 2
        if self._base_placed and top_touch_left:
            reward = 3
        if top_in_prestack_region:
            reward = 4
        if self._stable_stack_steps >= self.STACK_STABLE_SUCCESS_STEPS:
            reward = 5
        return reward
