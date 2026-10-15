import os

import numpy as np
from gymnasium import spaces

from bavo_bench.env.sim_config import XML_DIR
from bavo_bench.env.sim_env import AVAlohaEnv

TASK_XML_DIR = os.path.join(XML_DIR, "tasks", "dustpan_sweep")


class DustpanSweepEnv(AVAlohaEnv):
    """Hold a dustpan and sweep a small block through its opening."""

    XML = os.path.join(TASK_XML_DIR, "task.xml")
    ENV_STATE_DIM = 22
    PROMPTS = ["sweep the block into the dustpan"]

    DUSTPAN_JOINT_NAME = "dustpan_joint"
    STICK_JOINT_NAME = "sweeping_stick_joint"
    BLOCK_JOINT_NAME = "sweep_block_joint"
    DUSTPAN_TRAY_NAME = "dustpan_tray"
    DUSTPAN_HANDLE_NAME = "dustpan_handle"
    ENTRY_DETECTOR_NAME = "dustpan_entry_detector"
    STICK_NAME = "sweeping_stick"
    BLOCK_NAME = "sweep_block"
    TABLE_NAME = "table"

    DUSTPAN_X_RANGE = np.array([-0.13, -0.07], dtype=np.float64)
    DUSTPAN_Y_RANGE = np.array([0.04, 0.16], dtype=np.float64)
    DUSTPAN_Z = 0.0
    STICK_X_RANGE = np.array([0.19, 0.24], dtype=np.float64)
    STICK_Y_RANGE = np.array([0.07, 0.13], dtype=np.float64)
    STICK_Z = 0.021
    BLOCK_X_RANGE = np.array([0.03, 0.07], dtype=np.float64)
    BLOCK_Y_RANGE = np.array([0.08, 0.12], dtype=np.float64)
    BLOCK_Z = 0.016
    OPENING_RAY_HALF_WIDTH = 0.045
    INITIAL_LATERAL_OFFSET_RANGE = np.array([0.05, 0.12], dtype=np.float64)
    MAX_RESET_SAMPLE_ATTEMPTS = 100
    IDENTITY_QUATERNION = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_reward = 5
        self.dustpan_joint = self.mjcf_root.find(
            "joint", self.DUSTPAN_JOINT_NAME
        )
        self.sweeping_stick_joint = self.mjcf_root.find(
            "joint", self.STICK_JOINT_NAME
        )
        self.sweep_block_joint = self.mjcf_root.find(
            "joint", self.BLOCK_JOINT_NAME
        )
        self.dustpan_tray_geom = self.mjcf_root.find(
            "geom", self.DUSTPAN_TRAY_NAME
        )
        self.dustpan_handle_geom = self.mjcf_root.find(
            "geom", self.DUSTPAN_HANDLE_NAME
        )
        self.entry_detector_geom = self.mjcf_root.find(
            "geom", self.ENTRY_DETECTOR_NAME
        )
        self.sweeping_stick_geom = self.mjcf_root.find(
            "geom", self.STICK_NAME
        )
        self.sweep_block_geom = self.mjcf_root.find(
            "geom", self.BLOCK_NAME
        )
        self._task_stage = 0

        self.observation_space_dict["environment_state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.ENV_STATE_DIM,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict(self.observation_space_dict)

    def get_environment_state(self):
        return np.concatenate([
            self.physics.bind(self.dustpan_joint).qpos.copy(),
            self.physics.bind(self.sweeping_stick_joint).qpos.copy(),
            self.physics.bind(self.sweep_block_joint).qpos.copy(),
            np.array([self._task_stage], dtype=np.float64),
        ])

    def get_obs(self):
        obs = super().get_obs()
        obs["environment_state"] = self.get_environment_state()
        return obs

    def set_state(self, state, environment_state):
        if len(environment_state) != DustpanSweepEnv.ENV_STATE_DIM:
            raise ValueError(
                f"Expected environment_state with "
                f"{DustpanSweepEnv.ENV_STATE_DIM} values, "
                f"got {len(environment_state)}"
            )
        AVAlohaEnv.set_state(self, state, environment_state)
        self.physics.bind(self.dustpan_joint).qpos = environment_state[:7]
        self.physics.bind(self.sweeping_stick_joint).qpos = environment_state[7:14]
        self.physics.bind(self.sweep_block_joint).qpos = environment_state[14:21]
        self._task_stage = int(round(environment_state[21]))
        self.physics.forward()

    def _sample_free_joint_qpos(self, x_range, y_range, z):
        return np.concatenate([
            np.array([
                self.np_random.uniform(*x_range),
                self.np_random.uniform(*y_range),
                z,
            ], dtype=np.float64),
            self.IDENTITY_QUATERNION,
        ])

    def reset(self, seed=None, options=None):
        _, info = AVAlohaEnv.reset(self, seed=seed, options=options)
        block_qpos = self._sample_free_joint_qpos(
            self.BLOCK_X_RANGE,
            self.BLOCK_Y_RANGE,
            self.BLOCK_Z,
        )
        for _ in range(self.MAX_RESET_SAMPLE_ATTEMPTS):
            dustpan_qpos = self._sample_free_joint_qpos(
                self.DUSTPAN_X_RANGE,
                self.DUSTPAN_Y_RANGE,
                self.DUSTPAN_Z,
            )
            lateral_offset = abs(dustpan_qpos[1] - block_qpos[1])
            if (
                self.INITIAL_LATERAL_OFFSET_RANGE[0]
                < lateral_offset
                < self.INITIAL_LATERAL_OFFSET_RANGE[1]
            ):
                break
        else:
            raise RuntimeError(
                "Could not sample the requested dustpan lateral offset"
            )

        self.physics.bind(self.dustpan_joint).qpos = dustpan_qpos
        self.physics.bind(
            self.sweeping_stick_joint
        ).qpos = self._sample_free_joint_qpos(
            self.STICK_X_RANGE,
            self.STICK_Y_RANGE,
            self.STICK_Z,
        )
        self.physics.bind(self.sweep_block_joint).qpos = block_qpos
        self._task_stage = 0
        self.physics.forward()
        return self.get_obs(), info

    def _contact_pairs(self):
        contact_pairs = set()
        model = self.physics.model
        for contact_index in range(self.physics.data.ncon):
            contact = self.physics.data.contact[contact_index]
            geom_1 = model.id2name(contact.geom1, "geom")
            geom_2 = model.id2name(contact.geom2, "geom")
            if geom_1 is None or geom_2 is None:
                continue
            contact_pairs.add((geom_1, geom_2))
            contact_pairs.add((geom_2, geom_1))
        return contact_pairs

    @staticmethod
    def _is_finger_geom(geom_name, arm_name):
        return geom_name.startswith((
            f"{arm_name}_left_",
            f"{arm_name}_right_",
        ))

    def _geom_touches_fingers(self, geom_name, arm_name, contact_pairs):
        return any(
            geom_1 == geom_name and self._is_finger_geom(geom_2, arm_name)
            for geom_1, geom_2 in contact_pairs
        )

    def _block_is_in_opening_ray_region(self):
        tray_xy = self.physics.bind(self.dustpan_tray_geom).xpos[:2]
        entry_xy = self.physics.bind(self.entry_detector_geom).xpos[:2]
        block_xy = self.physics.bind(self.sweep_block_geom).xpos[:2]
        opening_direction = entry_xy - tray_xy
        norm = np.linalg.norm(opening_direction)
        if norm < 1e-8:
            return False
        forward = opening_direction / norm
        lateral = np.array([-forward[1], forward[0]], dtype=np.float64)
        block_from_entry = block_xy - entry_xy
        return bool(
            np.dot(block_from_entry, forward) >= 0.0
            and abs(np.dot(block_from_entry, lateral))
            <= self.OPENING_RAY_HALF_WIDTH
        )

    def get_reward(self):
        contact_pairs = self._contact_pairs()
        handle_touch_left = self._geom_touches_fingers(
            self.DUSTPAN_HANDLE_NAME,
            "left",
            contact_pairs,
        )
        stick_touch_right = self._geom_touches_fingers(
            self.STICK_NAME,
            "right",
            contact_pairs,
        )
        dustpan_on_table = (
            self.DUSTPAN_TRAY_NAME,
            self.TABLE_NAME,
        ) in contact_pairs
        stick_touch_block = (
            self.STICK_NAME,
            self.BLOCK_NAME,
        ) in contact_pairs
        block_touch_entry = (
            self.BLOCK_NAME,
            self.ENTRY_DETECTOR_NAME,
        ) in contact_pairs

        if self._task_stage == 0 and handle_touch_left:
            self._task_stage = 1
        elif (
            self._task_stage == 1
            and not handle_touch_left
            and dustpan_on_table
            and self._block_is_in_opening_ray_region()
        ):
            self._task_stage = 2
        elif self._task_stage == 2 and stick_touch_right:
            self._task_stage = 3
        elif (
            self._task_stage == 3
            and stick_touch_right
            and stick_touch_block
        ):
            self._task_stage = 4
        elif self._task_stage == 4 and block_touch_entry:
            self._task_stage = 5

        return self._task_stage
