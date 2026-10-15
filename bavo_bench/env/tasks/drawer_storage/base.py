import os

import numpy as np
from gymnasium import spaces

from bavo_bench.env.sim_config import XML_DIR
from bavo_bench.env.sim_env import AVAlohaEnv

TASK_XML_DIR = os.path.join(XML_DIR, "tasks", "drawer_storage")


class DrawerStorageEnv(AVAlohaEnv):
    """Open a drawer, store an item, and close the drawer."""

    XML = os.path.join(TASK_XML_DIR, "task.xml")
    ENV_STATE_DIM = 16
    PROMPTS = ["put the item in the drawer and close it"]

    CABINET_NAME = "cabinet"
    DRAWER_JOINT_NAME = "drawer_slide_joint"
    FRONT_PANEL_NAME = "drawer_front_panel"
    HANDLE_NAME = "drawer_handle"
    DRAWER_BOTTOM_NAME = "drawer_bottom"
    OPEN_DETECTOR_NAME = "drawer_open_detector"
    CLOSED_DETECTOR_NAME = "drawer_closed_detector"
    ITEM_NAME = "storage_item"

    CABINET_X_RANGE = np.array([-0.04, 0.04], dtype=np.float64)
    CABINET_Y = 0.24
    ITEM_XY_RANDOMIZATION_HALF_WIDTH = 0.025
    DRAWER_OPEN_THRESHOLD = 0.05
    DRAWER_CLOSED_THRESHOLD = 0.015

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_reward = 5
        self.cabinet_body = self.mjcf_root.find("body", self.CABINET_NAME)
        self.drawer_slide_joint = self.mjcf_root.find(
            "joint", self.DRAWER_JOINT_NAME
        )
        self.storage_item_joint = self.mjcf_root.find(
            "joint", "storage_item_joint"
        )
        self.drawer_front_panel_geom = self.mjcf_root.find(
            "geom", self.FRONT_PANEL_NAME
        )
        self.drawer_handle_geom = self.mjcf_root.find(
            "geom", self.HANDLE_NAME
        )
        self.storage_item_geom = self.mjcf_root.find("geom", self.ITEM_NAME)
        self._cabinet_base_pos = self.physics.bind(self.cabinet_body).pos.copy()
        self._cabinet_base_quat = self.physics.bind(self.cabinet_body).quat.copy()
        self._storage_item_initial_qpos = self.physics.bind(
            self.storage_item_joint
        ).qpos.copy()
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
            self.physics.bind(self.storage_item_joint).qpos.copy(),
            np.atleast_1d(
                self.physics.bind(self.drawer_slide_joint).qpos.copy()
            ),
            self.physics.bind(self.cabinet_body).pos.copy(),
            self.physics.bind(self.cabinet_body).quat.copy(),
            np.array([self._task_stage], dtype=np.float64),
        ])

    def get_obs(self):
        obs = super().get_obs()
        obs["environment_state"] = self.get_environment_state()
        return obs

    def set_state(self, state, environment_state):
        if len(environment_state) != DrawerStorageEnv.ENV_STATE_DIM:
            raise ValueError(
                f"Expected environment_state with "
                f"{DrawerStorageEnv.ENV_STATE_DIM} values, "
                f"got {len(environment_state)}"
            )
        AVAlohaEnv.set_state(self, state, environment_state)
        self.physics.bind(self.storage_item_joint).qpos = environment_state[:7]
        self.physics.bind(self.drawer_slide_joint).qpos = environment_state[7]
        self.physics.bind(self.cabinet_body).pos = environment_state[8:11]
        self.physics.bind(self.cabinet_body).quat = environment_state[11:15]
        self._task_stage = int(round(environment_state[15]))
        self.physics.forward()

    def reset(self, seed=None, options=None):
        _, info = AVAlohaEnv.reset(self, seed=seed, options=options)
        cabinet_pos = self._cabinet_base_pos.copy()
        cabinet_pos[0] = self.np_random.uniform(*self.CABINET_X_RANGE)
        cabinet_pos[1] = self.CABINET_Y
        self.physics.bind(self.cabinet_body).pos = cabinet_pos
        self.physics.bind(self.cabinet_body).quat = self._cabinet_base_quat
        self.physics.bind(self.drawer_slide_joint).qpos = 0.0
        item_qpos = self._storage_item_initial_qpos.copy()
        item_qpos[:2] += self.np_random.uniform(
            -self.ITEM_XY_RANDOMIZATION_HALF_WIDTH,
            self.ITEM_XY_RANDOMIZATION_HALF_WIDTH,
            size=2,
        )
        self.physics.bind(self.storage_item_joint).qpos = item_qpos
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

    def _drawer_position(self):
        return float(
            np.asarray(self.physics.bind(self.drawer_slide_joint).qpos).item()
        )

    def _left_gripper_touches_item(self):
        return self._geom_touches_fingers(
            self.ITEM_NAME, "left", self._contact_pairs()
        )

    def get_reward(self):
        contact_pairs = self._contact_pairs()
        handle_touch_right = self._geom_touches_fingers(
            self.HANDLE_NAME, "right", contact_pairs
        )
        item_touch_left = self._geom_touches_fingers(
            self.ITEM_NAME, "left", contact_pairs
        )
        item_touch_right = self._geom_touches_fingers(
            self.ITEM_NAME, "right", contact_pairs
        )
        drawer_position = self._drawer_position()
        drawer_open = drawer_position > self.DRAWER_OPEN_THRESHOLD
        drawer_closed = drawer_position <= self.DRAWER_CLOSED_THRESHOLD
        item_in_drawer = (
            self.ITEM_NAME,
            self.DRAWER_BOTTOM_NAME,
        ) in contact_pairs

        if self._task_stage == 0 and handle_touch_right:
            self._task_stage = 1
        elif self._task_stage == 1 and drawer_open:
            self._task_stage = 2
        elif self._task_stage == 2 and item_touch_left:
            self._task_stage = 3
        elif (
            self._task_stage == 3
            and item_in_drawer
            and not item_touch_left
        ):
            self._task_stage = 4
        elif (
            self._task_stage == 4
            and drawer_closed
            and item_in_drawer
            and not item_touch_left
            and not item_touch_right
        ):
            self._task_stage = 5

        return self._task_stage
