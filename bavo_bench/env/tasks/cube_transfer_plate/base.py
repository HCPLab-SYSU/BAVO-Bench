import os

import numpy as np
from gymnasium import spaces

from bavo_bench.env.sim_config import XML_DIR
from bavo_bench.env.sim_env import AVAlohaEnv

TASK_XML_DIR = os.path.join(XML_DIR, "tasks", "cube_transfer_plate")


class CubeTransferPlateEnv(AVAlohaEnv):
    """Cube handoff followed by placement on a randomized plate."""

    XML = os.path.join(TASK_XML_DIR, "task.xml")
    ENV_STATE_DIM = 17
    PROMPTS = ["transfer red cube to plate"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_reward = 5
        self._grasp_right = False
        self.cube_joint = self.mjcf_root.find("joint", "cube_joint")
        self.plate_body = self.mjcf_root.find("body", "plate")
        self.plate_target_geom = self.mjcf_root.find("geom", "plate_target")
        self._plate_base_pos = np.asarray(
            self.plate_body.pos,
            dtype=np.float64,
        ).copy()

        self._cube_x_range = np.array([0.0, 0.1], dtype=np.float64)
        self._cube_y_range = np.array([0.05, 0.15], dtype=np.float64)
        self._cube_z_range = np.array([0.0, 0.0], dtype=np.float64)
        self._plate_x_range = np.array([-0.2, -0.1], dtype=np.float64)
        self._plate_y_range = self._cube_y_range.copy()
        self._plate_z = self._plate_base_pos[2]

        self.observation_space_dict["environment_state"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.ENV_STATE_DIM,),
            dtype=np.float64,
        )
        self.observation_space = spaces.Dict(self.observation_space_dict)

    def get_environment_state(self):
        return np.concatenate([
            self.physics.bind(self.cube_joint).qpos.copy(),
            self.physics.bind(self.plate_body).pos.copy(),
            self.physics.bind(self.plate_body).quat.copy(),
            self.physics.bind(self.plate_target_geom).size.copy(),
        ])

    def get_obs(self):
        obs = super().get_obs()
        obs["environment_state"] = self.get_environment_state()
        return obs

    def set_state(self, state, environment_state):
        AVAlohaEnv.set_state(self, state, environment_state)
        self.physics.bind(self.cube_joint).qpos = environment_state[:7]
        self.physics.bind(self.plate_body).pos = environment_state[7:10]
        self.physics.bind(self.plate_body).quat = environment_state[10:14]
        self.physics.bind(self.plate_target_geom).size = environment_state[14:17]
        self.physics.forward()

    def reset(self, seed=None, options=None):
        _, info = AVAlohaEnv.reset(self, seed=seed, options=options)

        cube_ranges = np.vstack([
            self._cube_x_range,
            self._cube_y_range,
            self._cube_z_range,
        ])
        cube_position = self.np_random.uniform(
            cube_ranges[:, 0],
            cube_ranges[:, 1],
        )
        cube_quat = np.array([1, 0, 0, 0])
        self.physics.bind(self.cube_joint).qpos = np.concatenate([
            cube_position,
            cube_quat,
        ])

        plate_position = np.array([
            self.np_random.uniform(*self._plate_x_range),
            self.np_random.uniform(*self._plate_y_range),
            self._plate_z,
        ], dtype=np.float64)
        self.physics.bind(self.plate_body).pos = plate_position

        self._grasp_right = False
        self.physics.forward()
        return self.get_obs(), info

    def get_reward(self):
        touch_left_gripper = False
        touch_right_gripper = False
        cube_touch_table = False
        cube_touch_plate = False

        contact_pairs = []
        for contact_index in range(self.physics.data.ncon):
            geom_id_1 = self.physics.data.contact[contact_index].geom1
            geom_id_2 = self.physics.data.contact[contact_index].geom2
            geom_1 = self.physics.model.id2name(geom_id_1, "geom")
            geom_2 = self.physics.model.id2name(geom_id_2, "geom")
            contact_pairs.append((geom_1, geom_2))
            contact_pairs.append((geom_2, geom_1))

        for geom_1, geom_2 in contact_pairs:
            if geom_1 == "cube" and geom_2.startswith("right"):
                touch_right_gripper = True
            if geom_1 == "cube" and geom_2.startswith("left"):
                touch_left_gripper = True
            if geom_1 == "table" and geom_2 == "cube":
                cube_touch_table = True
            if geom_1 == "cube" and geom_2 == "plate_target":
                cube_touch_plate = True

        reward = 0
        if touch_right_gripper:
            reward = 1
        if touch_right_gripper and not cube_touch_table and not touch_left_gripper:
            reward = 2
            self._grasp_right = True
        if (
            touch_right_gripper
            and touch_left_gripper
            and not cube_touch_table
            and self._grasp_right
        ):
            reward = 3
        if (
            touch_left_gripper
            and not cube_touch_table
            and not touch_right_gripper
            and self._grasp_right
        ):
            reward = 4
        if (
            cube_touch_plate
            and touch_left_gripper
            and not touch_right_gripper
            and self._grasp_right
        ):
            reward = 5
        return reward
