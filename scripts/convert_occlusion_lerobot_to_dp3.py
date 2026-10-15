import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import gymnasium as gym
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from numcodecs import Blosc
from tqdm import tqdm

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
import bavo_bench  # noqa: F401
from bavo_bench.utils.pointcloud_utils import (
    render_point_cloud_observation,
)


DEFAULT_CROP_MIN = np.array([0.2, -0.35, -0.01], dtype=np.float32)
DEFAULT_CROP_MAX = np.array([0.8, 0.35, 0.8], dtype=np.float32)


def parse_vector(values, default: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if values is None:
        return default
    return np.asarray(values, dtype=np.float32)


def read_lerobot_v3_table(dataset_path: Path):
    info_path = dataset_path / "meta" / "info.json"
    if not info_path.is_file():
        raise FileNotFoundError(f"Missing LeRobot metadata: {info_path}")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    data_files = sorted(dataset_path.glob("data/chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {dataset_path}/data")

    table = pa.concat_tables(
        [
            pq.read_table(
                path,
                columns=[
                    "observation.state",
                    "action",
                    "observation.replay_state",
                    "episode_index",
                    "frame_index",
                ],
            )
            for path in data_files
        ]
    )
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float32)
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float32)
    replay_states = np.asarray(
        table.column("observation.replay_state").to_pylist(),
        dtype=np.float32,
    )
    episode_indices = np.asarray(table.column("episode_index").to_pylist(), dtype=np.int64)
    frame_indices = np.asarray(table.column("frame_index").to_pylist(), dtype=np.int64)
    order = np.lexsort((frame_indices, episode_indices))
    return info, states[order], actions[order], replay_states[order], episode_indices[order]


def split_episode_slices(episode_indices: np.ndarray):
    slices = []
    unique_episodes = np.unique(episode_indices)
    for episode_index in unique_episodes:
        frame_ids = np.flatnonzero(episode_indices == episode_index)
        if len(frame_ids) == 0:
            continue
        slices.append((int(episode_index), int(frame_ids[0]), int(frame_ids[-1]) + 1))
    return slices


def task_environment_state_dim(base_env) -> Optional[int]:
    observation_spaces = getattr(base_env.observation_space, "spaces", {})
    environment_space = observation_spaces.get("environment_state")
    if environment_space is None:
        return None
    return int(environment_space.shape[0])


def restore_task_environment_state(base_env, environment_state: np.ndarray) -> None:
    """Restore task-owned render state through the environment contract."""
    expected_dim = task_environment_state_dim(base_env)
    if expected_dim is None:
        raise ValueError(f"{type(base_env).__name__} has no environment_state")
    if environment_state.shape[0] != expected_dim:
        raise ValueError(
            f"Expected {type(base_env).__name__} environment_state with "
            f"{expected_dim} values, got {environment_state.shape[0]}"
        )
    restore_hook = getattr(base_env, "restore_environment_state_for_render", None)
    if restore_hook is not None:
        restore_hook(environment_state)
        return

    agent_state = np.asarray(base_env.get_obs()["agent_pos"], dtype=np.float64)
    base_env.set_state(agent_state, environment_state)


def set_replay_state(env, replay_state: np.ndarray, full_qpos_dim: int) -> str:
    base_env = env.unwrapped
    if replay_state.shape[0] <= full_qpos_dim:
        raise ValueError(
            f"Replay state has {replay_state.shape[0]} values but qpos requires "
            f"{full_qpos_dim}"
        )
    qpos = np.asarray(replay_state[:full_qpos_dim], dtype=np.float64)
    base_env.physics.data.qpos[:full_qpos_dim] = qpos
    payload_dim = replay_state.shape[0] - full_qpos_dim
    environment_dim = task_environment_state_dim(base_env)
    if environment_dim is None:
        raise ValueError(f"{type(base_env).__name__} has no environment_state")
    expected_payload_dim = environment_dim + 1
    if payload_dim != expected_payload_dim:
        raise ValueError(
            f"Unexpected replay_state payload dimension {payload_dim}; expected "
            f"environment_state_present + environment_state({environment_dim}) "
            f"after qpos({full_qpos_dim}) for "
            f"{type(base_env).__name__}"
        )

    environment_state_present = bool(round(float(replay_state[full_qpos_dim])))
    if environment_state_present:
        environment_state = np.asarray(
            replay_state[full_qpos_dim + 1 :],
            dtype=np.float64,
        )
        restore_task_environment_state(base_env, environment_state)
    replay_schema = (
        f"qpos_{full_qpos_dim}_plus_presence_plus_environment_state_{environment_dim}"
    )

    # set_state may map scalar policy grippers back onto paired finger joints.
    # Recorded all_qpos remains authoritative for every joint during rendering.
    base_env.physics.data.qpos[:full_qpos_dim] = qpos
    base_env.physics.forward()
    return replay_schema


def write_dp3_dataset(
    output_path: Path,
    states: np.ndarray,
    actions: np.ndarray,
    point_clouds: np.ndarray,
    replay_states: np.ndarray,
    episode_ends: np.ndarray,
) -> None:
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.SHUFFLE)
    root = zarr.open(str(output_path), mode="w")
    data = root.create_group("data")
    meta = root.create_group("meta")

    data.create_dataset(
        "state",
        data=states,
        chunks=(min(1024, len(states)), states.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    data.create_dataset(
        "action",
        data=actions,
        chunks=(min(1024, len(actions)), actions.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    data.create_dataset(
        "point_cloud",
        data=point_clouds,
        chunks=(min(256, len(point_clouds)), point_clouds.shape[1], point_clouds.shape[2]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    data.create_dataset(
        "replay_state",
        data=replay_states,
        chunks=(min(1024, len(replay_states)), replay_states.shape[1]),
        dtype="float32",
        overwrite=True,
        compressor=compressor,
    )
    meta.create_dataset(
        "episode_ends",
        data=episode_ends,
        chunks=(min(1024, len(episode_ends)),),
        dtype="int64",
        overwrite=True,
        compressor=compressor,
    )


def convert_dataset(args) -> None:
    dataset_path = Path(args.dataset_path).expanduser().resolve()
    output_path = Path(args.output_zarr).expanduser().resolve()
    crop_min = parse_vector(args.crop_min, DEFAULT_CROP_MIN)
    crop_max = parse_vector(args.crop_max, DEFAULT_CROP_MAX)

    info, states, actions, replay_states, episode_indices = read_lerobot_v3_table(dataset_path)
    selected_slices = split_episode_slices(episode_indices)
    selected_slices = selected_slices[args.start_episode : args.end_episode]
    if args.max_episodes is not None:
        selected_slices = selected_slices[: args.max_episodes]
    if not selected_slices:
        raise ValueError("No episodes selected for conversion.")

    env = None
    replay_schema = None
    try:
        env = gym.make(
            args.env_id,
            cameras={},
            render_camera=args.camera_name,
            render_height=args.render_height,
            render_width=args.render_width,
        )
        env.reset(seed=args.seed)
        full_qpos_dim = int(env.unwrapped.physics.data.qpos.shape[0])
        environment_state_dim = task_environment_state_dim(env.unwrapped)
        if environment_state_dim is None:
            raise ValueError(
                f"{type(env.unwrapped).__name__} has no environment_state"
            )
        expected_replay_dim = full_qpos_dim + 1 + environment_state_dim
        if replay_states.shape[1] != expected_replay_dim:
            raise ValueError(
                "Unexpected replay_state shape: "
                f"{replay_states.shape[1]} != {expected_replay_dim} for "
                f"full_qpos_dim={full_qpos_dim}, "
                f"environment_state_dim={environment_state_dim}"
            )

        all_states = []
        all_actions = []
        all_replay_states = []
        all_point_clouds = []
        episode_ends = []
        total_frames = 0
        for episode_index, start, end in tqdm(selected_slices, desc="Episodes"):
            ep_states = states[start:end]
            ep_actions = actions[start:end]
            ep_replay_states = replay_states[start:end]
            if args.max_frames_per_episode is not None:
                ep_states = ep_states[: args.max_frames_per_episode]
                ep_actions = ep_actions[: args.max_frames_per_episode]
                ep_replay_states = ep_replay_states[: args.max_frames_per_episode]

            ep_point_clouds = []
            for replay_state in tqdm(
                ep_replay_states,
                desc=f"episode_{episode_index:04d}",
                leave=False,
            ):
                replay_schema = set_replay_state(env, replay_state, full_qpos_dim)
                observation = render_point_cloud_observation(
                    physics=env.unwrapped.physics,
                    camera_name=args.camera_name,
                    camera_site_name=args.camera_site_name,
                    reference_body_name=args.reference_body_name,
                    pointcloud_frame=args.pointcloud_frame,
                    height=args.render_height,
                    width=args.render_width,
                    crop_min=crop_min,
                    crop_max=crop_max,
                    num_points=args.num_points,
                    sampling_method=args.sampling_method,
                    include_rgb=args.include_rgb,
                )
                ep_point_clouds.append(observation.point_cloud.astype(np.float32))

            all_states.append(ep_states.astype(np.float32))
            all_actions.append(ep_actions.astype(np.float32))
            all_replay_states.append(ep_replay_states.astype(np.float32))
            all_point_clouds.append(np.stack(ep_point_clouds, axis=0).astype(np.float32))
            total_frames += len(ep_states)
            episode_ends.append(total_frames)
    finally:
        if env is not None:
            env.close()

    out_states = np.concatenate(all_states, axis=0)
    out_actions = np.concatenate(all_actions, axis=0)
    out_replay_states = np.concatenate(all_replay_states, axis=0)
    out_point_clouds = np.concatenate(all_point_clouds, axis=0)
    out_episode_ends = np.asarray(episode_ends, dtype=np.int64)

    write_dp3_dataset(
        output_path=output_path,
        states=out_states,
        actions=out_actions,
        point_clouds=out_point_clouds,
        replay_states=out_replay_states,
        episode_ends=out_episode_ends,
    )

    metadata = {
        "source_dataset_path": str(dataset_path),
        "source_total_episodes": int(info.get("total_episodes", -1)),
        "source_total_frames": int(info.get("total_frames", -1)),
        "env_id": args.env_id,
        "camera_name": args.camera_name,
        "camera_site_name": args.camera_site_name,
        "reference_body_name": args.reference_body_name,
        "pointcloud_frame": args.pointcloud_frame,
        "render_height": int(args.render_height),
        "render_width": int(args.render_width),
        "num_points": int(args.num_points),
        "point_cloud_channels": int(out_point_clouds.shape[-1]),
        "include_rgb": bool(args.include_rgb),
        "sampling_method": args.sampling_method,
        "crop_min": None if crop_min is None else crop_min.tolist(),
        "crop_max": None if crop_max is None else crop_max.tolist(),
        "full_qpos_dim": int(full_qpos_dim),
        "environment_state_dim": environment_state_dim,
        "replay_state_dim": int(out_replay_states.shape[1]),
        "replay_state_schema": replay_schema,
        "num_episodes": int(len(out_episode_ends)),
        "num_frames": int(total_frames),
        "episode_indices": [item[0] for item in selected_slices],
        "script": "scripts/convert_occlusion_lerobot_to_dp3.py",
    }
    if "stage-occlusion" in args.env_id:
        metadata["stage_occlusion_episodes"] = [item[0] for item in selected_slices]
    with open(output_path.with_suffix(".json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved DP3 dataset to {output_path}")
    print(f"Metadata written to {output_path.with_suffix('.json')}")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Convert AV-Aloha occlusion LeRobot-v3 replay_state data to DP3 zarr."
    )
    parser.add_argument("--dataset-path", type=str, required=True)
    parser.add_argument("--output-zarr", type=str, required=True)
    parser.add_argument(
        "--env-id",
        type=str,
        required=True,
        help="Registered stage-occlusion environment used to restore and render replay state.",
    )
    parser.add_argument("--camera-name", type=str, default="zed_cam_right")
    parser.add_argument("--camera-site-name", type=str, default="middle_zed_cam_right_optical_frame")
    parser.add_argument("--reference-body-name", type=str, default="middle_base_link")
    parser.add_argument("--pointcloud-frame", choices=["reference", "camera"], default="reference")
    parser.add_argument("--render-height", type=int, default=224)
    parser.add_argument("--render-width", type=int, default=224)
    parser.add_argument("--num-points", type=int, default=1024)
    parser.add_argument("--sampling-method", choices=["fps", "uniform"], default="fps")
    parser.add_argument(
        "--include-rgb",
        action="store_true",
        help="Append normalized per-point RGB values, producing XYZRGB point clouds.",
    )
    parser.add_argument("--crop-min", type=float, nargs=3, default=None)
    parser.add_argument("--crop-max", type=float, nargs=3, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-episode", type=int, default=0)
    parser.add_argument("--end-episode", type=int, default=None)
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--max-frames-per-episode", type=int, default=None)
    return parser


def main() -> None:
    parser = build_argparser()
    args = parser.parse_args()
    convert_dataset(args)


if __name__ == "__main__":
    main()
