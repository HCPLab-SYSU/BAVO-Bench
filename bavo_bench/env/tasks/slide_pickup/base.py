import os

import numpy as np
from gymnasium import spaces

from bavo_bench.env.sim_config import XML_DIR
from bavo_bench.env.sim_env import AVAlohaEnv

TASK_XML_DIR = os.path.join(XML_DIR, "tasks", "slide_pickup")


class SlidePickupEnv(AVAlohaEnv):
    """Slide a handled object off a platform edge, then pick it up."""

    XML = os.path.join(TASK_XML_DIR, "task.xml")
    ENV_STATE_DIM = 8
    PROMPTS = ["slide the handled object off the platform and pick it up"]

    BLADE_NAME = "object_blade"
    HANDLE_NAME = "object_handle"
    PLATFORM_NAME = "support_platform"
    EXPOSURE_DETECTOR_NAME = "handle_exposure_detector"
    OBJECT_GEOM_NAMES = frozenset((BLADE_NAME, HANDLE_NAME))

    PLATFORM_CENTER_XY = np.array([0.0, 0.1], dtype=np.float64)
    PLATFORM_HALF_SIZE_XY = np.array([0.14, 0.14], dtype=np.float64)
    OBJECT_X_RANGE = np.array([-0.01, 0.01], dtype=np.float64)
    OBJECT_Y_OFFSET_RANGE = np.array([-0.04, 0.04], dtype=np.float64)
    OBJECT_YAW_RANGE = np.deg2rad(np.array([-5.0, 5.0], dtype=np.float64))
    OBJECT_Z = 0.055
    MIN_LIFT_CLEARANCE = 0.03
    MAX_RESET_SAMPLE_ATTEMPTS = 100

    BLADE_CENTER_XY = np.array([0.045, 0.0], dtype=np.float64)
    BLADE_HALF_SIZE_XY = np.array([0.07, 0.06], dtype=np.float64)
    HANDLE_CENTER_XY = np.array([-0.07, 0.0], dtype=np.float64)
    HANDLE_HALF_SIZE_XY = np.array([0.045, 0.015], dtype=np.float64)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_reward = 5
        self.handled_object_joint = self.mjcf_root.find(
            "joint", "handled_object_joint"
        )
        self.object_blade_geom = self.mjcf_root.find("geom", self.BLADE_NAME)
        self.object_handle_geom = self.mjcf_root.find("geom", self.HANDLE_NAME)
        self.support_platform_geom = self.mjcf_root.find(
            "geom", self.PLATFORM_NAME
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
            self.physics.bind(self.handled_object_joint).qpos.copy(),
            np.array([self._task_stage], dtype=np.float64),
        ])

    def get_obs(self):
        obs = super().get_obs()
        obs["environment_state"] = self.get_environment_state()
        return obs

    def set_state(self, state, environment_state):
        if len(environment_state) != SlidePickupEnv.ENV_STATE_DIM:
            raise ValueError(
                f"Expected environment_state with {SlidePickupEnv.ENV_STATE_DIM} "
                f"values, got {len(environment_state)}"
            )
        AVAlohaEnv.set_state(self, state, environment_state)
        self.physics.bind(self.handled_object_joint).qpos = environment_state[:7]
        self._task_stage = int(round(environment_state[7]))
        self.physics.forward()

    @staticmethod
    def _rectangle_corners(center, half_size):
        offsets = np.array([
            [-half_size[0], -half_size[1]],
            [-half_size[0], half_size[1]],
            [half_size[0], -half_size[1]],
            [half_size[0], half_size[1]],
        ], dtype=np.float64)
        return center + offsets

    def _object_footprint_inside_platform(self, position_xy, yaw):
        local_corners = np.concatenate([
            self._rectangle_corners(
                self.BLADE_CENTER_XY,
                self.BLADE_HALF_SIZE_XY,
            ),
            self._rectangle_corners(
                self.HANDLE_CENTER_XY,
                self.HANDLE_HALF_SIZE_XY,
            ),
        ])
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        rotation = np.array([
            [cos_yaw, -sin_yaw],
            [sin_yaw, cos_yaw],
        ], dtype=np.float64)
        world_corners = local_corners @ rotation.T + position_xy
        lower = self.PLATFORM_CENTER_XY - self.PLATFORM_HALF_SIZE_XY
        upper = self.PLATFORM_CENTER_XY + self.PLATFORM_HALF_SIZE_XY
        return bool(np.all((world_corners >= lower) & (world_corners <= upper)))

    def _sample_object_qpos(self):
        for _ in range(self.MAX_RESET_SAMPLE_ATTEMPTS):
            position_xy = np.array([
                self.np_random.uniform(*self.OBJECT_X_RANGE),
                self.PLATFORM_CENTER_XY[1]
                + self.np_random.uniform(*self.OBJECT_Y_OFFSET_RANGE),
            ], dtype=np.float64)
            yaw = self.np_random.uniform(*self.OBJECT_YAW_RANGE)
            if self._object_footprint_inside_platform(position_xy, yaw):
                break
        else:
            raise RuntimeError("Could not sample the handled object on the platform")

        quaternion = np.array([
            np.cos(yaw / 2.0),
            0.0,
            0.0,
            np.sin(yaw / 2.0),
        ], dtype=np.float64)
        return np.concatenate([
            position_xy,
            np.array([self.OBJECT_Z], dtype=np.float64),
            quaternion,
        ])

    def reset(self, seed=None, options=None):
        _, info = AVAlohaEnv.reset(self, seed=seed, options=options)
        self.physics.bind(self.handled_object_joint).qpos = self._sample_object_qpos()
        self._task_stage = 0
        self.physics.forward()
        return self.get_obs(), info

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

    @staticmethod
    def _is_finger_geom(geom_name, arm_name):
        return geom_name.startswith((
            f"{arm_name}_left_",
            f"{arm_name}_right_",
        ))

    def _geom_touches_fingers(self, object_geom, arm_name, contact_pairs):
        return any(
            geom_1 == object_geom and self._is_finger_geom(geom_2, arm_name)
            for geom_1, geom_2 in contact_pairs
        )

    def _object_touches_fingers(self, arm_name, contact_pairs):
        return any(
            geom_1 in self.OBJECT_GEOM_NAMES
            and self._is_finger_geom(geom_2, arm_name)
            for geom_1, geom_2 in contact_pairs
        )

    def _geom_vertical_bounds(self, geom):
        binding = self.physics.bind(geom)
        rotation = np.asarray(binding.xmat, dtype=np.float64).reshape(3, 3)
        half_extent_z = np.abs(rotation[2]) @ np.asarray(
            binding.size,
            dtype=np.float64,
        )
        center_z = float(binding.xpos[2])
        return center_z - half_extent_z, center_z + half_extent_z

    def _object_platform_vertical_clearance(self):
        platform_top = self._geom_vertical_bounds(
            self.support_platform_geom
        )[1]
        object_bottom = min(
            self._geom_vertical_bounds(self.object_blade_geom)[0],
            self._geom_vertical_bounds(self.object_handle_geom)[0],
        )
        return object_bottom - platform_top

    def get_reward(self):
        contact_pairs = self._contact_pairs()
        blade_touch_right = self._geom_touches_fingers(
            self.BLADE_NAME,
            "right",
            contact_pairs,
        )
        object_touch_right = self._object_touches_fingers("right", contact_pairs)
        handle_touch_left = self._geom_touches_fingers(
            self.HANDLE_NAME,
            "left",
            contact_pairs,
        )
        handle_exposed = (
            self.HANDLE_NAME,
            self.EXPOSURE_DETECTOR_NAME,
        ) in contact_pairs
        lift_clearance = self._object_platform_vertical_clearance()

        if self._task_stage == 0 and blade_touch_right:
            self._task_stage = 1
        elif self._task_stage == 1 and handle_exposed:
            self._task_stage = 2
        elif self._task_stage == 2 and not object_touch_right:
            self._task_stage = 3
        elif (
            self._task_stage == 3
            and handle_touch_left
            and not object_touch_right
        ):
            self._task_stage = 4
        elif (
            self._task_stage == 4
            and handle_touch_left
            and lift_clearance > self.MIN_LIFT_CLEARANCE
        ):
            self._task_stage = 5

        return self._task_stage
