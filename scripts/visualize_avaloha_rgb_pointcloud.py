#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import matplotlib
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

matplotlib.use("Agg")
import matplotlib.pyplot as plt


OCCLUSION_PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def episode_bounds(episode_ends, episode_index):
    if episode_index < 0 or episode_index >= len(episode_ends):
        raise IndexError(
            f"episode_index={episode_index} out of range for {len(episode_ends)} episodes"
        )
    start = 0 if episode_index == 0 else int(episode_ends[episode_index - 1])
    return start, int(episode_ends[episode_index])


def read_video_metadata(dataset_path, camera_key, source_episode):
    prefix = f"videos/{camera_key}"
    columns = [
        "episode_index",
        f"{prefix}/chunk_index",
        f"{prefix}/file_index",
        f"{prefix}/from_timestamp",
    ]
    files = sorted(dataset_path.glob("meta/episodes/chunk-*/*.parquet"))
    if not files:
        raise FileNotFoundError(f"No episode metadata under {dataset_path}/meta/episodes")
    table = pa.concat_tables([pq.read_table(path, columns=columns) for path in files])
    for row in table.to_pylist():
        if int(row["episode_index"]) == source_episode:
            chunk_index = int(row[f"{prefix}/chunk_index"])
            file_index = int(row[f"{prefix}/file_index"])
            return {
                "path": dataset_path
                / "videos"
                / camera_key
                / f"chunk-{chunk_index:03d}"
                / f"file-{file_index:03d}.mp4",
                "start_time": float(row[f"{prefix}/from_timestamp"]),
            }
    raise KeyError(f"No video metadata for source episode {source_episode}")


def decode_video_segment(video_path, start_time, num_frames, width, height):
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(video_path),
        "-ss",
        f"{start_time:.6f}",
        "-frames:v",
        str(num_frames),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "rgb24",
        "pipe:1",
    ]
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE)
    frame_bytes = width * height * 3
    decoded_frames = len(result.stdout) // frame_bytes
    if decoded_frames < num_frames:
        raise RuntimeError(
            f"Decoded {decoded_frames} frames from {video_path}, expected {num_frames}"
        )
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(
        decoded_frames, height, width, 3
    )[:num_frames]


def render_frame(
    camera_rgb,
    point_cloud,
    crop_min,
    crop_max,
    title,
    elev,
    azim,
    robot_points=None,
    robot_color=(0.0, 0.85, 1.0),
    robot_point_size=12.0,
):
    xyz = np.asarray(point_cloud[:, :3], dtype=np.float32)
    rgb = np.clip(np.asarray(point_cloud[:, 3:6], dtype=np.float32), 0.0, 1.0)
    valid = np.any(xyz != 0.0, axis=-1)
    xyz = xyz[valid]
    rgb = rgb[valid]

    fig = plt.figure(figsize=(12, 4), dpi=100)
    ax_camera = fig.add_subplot(1, 3, 1)
    ax_3d = fig.add_subplot(1, 3, 2, projection="3d")
    ax_top = fig.add_subplot(1, 3, 3)

    ax_camera.imshow(camera_rgb)
    ax_camera.set_title("zed_cam_right")
    ax_camera.axis("off")

    ax_3d.scatter(
        xyz[:, 0],
        xyz[:, 1],
        xyz[:, 2],
        c=rgb,
        s=12,
        linewidths=0,
        depthshade=False,
    )
    if robot_points is not None:
        robot_points = np.asarray(robot_points, dtype=np.float32)
        ax_3d.scatter(
            robot_points[:, 0],
            robot_points[:, 1],
            robot_points[:, 2],
            c=[robot_color],
            s=robot_point_size * (14.0 / 12.0),
            edgecolors="black",
            linewidths=0.45,
            depthshade=False,
            label=f"middle arm ({len(robot_points)})",
        )
        ax_3d.legend(loc="upper right", fontsize=8)
    ax_3d.set_xlim(float(crop_min[0]), float(crop_max[0]))
    ax_3d.set_ylim(float(crop_min[1]), float(crop_max[1]))
    ax_3d.set_zlim(float(crop_min[2]), float(crop_max[2]))
    ax_3d.set_xlabel("x")
    ax_3d.set_ylabel("y")
    ax_3d.set_zlabel("z")
    point_count_text = f"scene {len(xyz)}"
    if robot_points is not None:
        point_count_text += f" + robot {len(robot_points)}"
    ax_3d.set_title(f"reference-frame point cloud ({point_count_text})")
    ax_3d.view_init(elev=elev, azim=azim)
    ax_3d.set_box_aspect(
        tuple(float(crop_max[index] - crop_min[index]) for index in range(3))
    )

    ax_top.scatter(-xyz[:, 1], xyz[:, 0], c=rgb, s=14, linewidths=0)
    if robot_points is not None:
        ax_top.scatter(
            -robot_points[:, 1],
            robot_points[:, 0],
            c=[robot_color],
            s=robot_point_size,
            edgecolors="black",
            linewidths=0.45,
        )
    ax_top.set_xlim(float(-crop_max[1]), float(-crop_min[1]))
    ax_top.set_ylim(float(crop_min[0]), float(crop_max[0]))
    ax_top.set_aspect("equal", adjustable="box")
    ax_top.set_xlabel("-y")
    ax_top.set_ylabel("x")
    ax_top.set_title("top-down XYZRGB")
    ax_top.grid(True, alpha=0.25)

    fig.suptitle(title)
    fig.tight_layout()
    fig.canvas.draw()
    rendered = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return rendered


def compute_middle_arm_points(
    root,
    start,
    num_frames,
    metadata,
    asset_path,
    expected_num_points,
):
    if "data/replay_state" not in root:
        raise KeyError("--include-middle-arm requires data/replay_state in the zarr")
    if not OCCLUSION_PACKAGE_ROOT.is_dir():
        raise FileNotFoundError(
            f"Missing local occlusion package: {OCCLUSION_PACKAGE_ROOT}"
        )
    if str(OCCLUSION_PACKAGE_ROOT) not in sys.path:
        sys.path.insert(0, str(OCCLUSION_PACKAGE_ROOT))

    import gymnasium as gym

    import bavo_bench  # noqa: F401
    from bavo_bench.utils.middle_arm_pointcloud import (
        MiddleArmPointCloud,
    )

    env_id = metadata.get("env_id")
    if env_id is None:
        raise KeyError("Dataset metadata has no env_id for replaying robot FK")
    env = gym.make(
        env_id,
        cameras={},
        render_camera=metadata.get("camera_name", "zed_cam_right"),
    )
    try:
        physics = env.unwrapped.physics
        full_qpos_dim = int(physics.data.qpos.shape[0])
        metadata_qpos_dim = metadata.get("full_qpos_dim")
        if metadata_qpos_dim is not None and int(metadata_qpos_dim) != full_qpos_dim:
            raise ValueError(
                f"Dataset full_qpos_dim={metadata_qpos_dim}, model has {full_qpos_dim}"
            )
        replay_states = root["data/replay_state"]
        if replay_states.shape[1] < full_qpos_dim:
            raise ValueError(
                f"Replay state has {replay_states.shape[1]} values, qpos needs {full_qpos_dim}"
            )
        robot_pointcloud = MiddleArmPointCloud(
            physics=physics,
            asset_path=asset_path,
            reference_body_name=metadata.get("reference_body_name", "middle_base_link"),
        )
        if robot_pointcloud.num_points != expected_num_points:
            raise ValueError(
                f"Robot asset has {robot_pointcloud.num_points} points, "
                f"expected {expected_num_points}"
            )

        frames = np.empty(
            (num_frames, robot_pointcloud.num_points, 3), dtype=np.float32
        )
        for local_frame in range(num_frames):
            replay_state = np.asarray(
                replay_states[start + local_frame], dtype=np.float64
            )
            physics.data.qpos[:full_qpos_dim] = replay_state[:full_qpos_dim]
            physics.forward()
            frames[local_frame] = robot_pointcloud.points_in_reference_frame()
        return frames, robot_pointcloud
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Render camera and true-RGB point-cloud views for an AV-Aloha zarr episode."
    )
    parser.add_argument("--zarr-path", type=Path, required=True)
    parser.add_argument("--dataset-path", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episode-index", type=int, default=0)
    parser.add_argument("--camera-key", default="observation.images.zed_cam_right")
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--frame-stride", type=int, default=1)
    parser.add_argument("--view-elev", type=float, default=24.0)
    parser.add_argument("--view-azim", type=float, default=150.0)
    parser.add_argument(
        "--include-middle-arm",
        action="store_true",
        help="Overlay fixed middle-arm surface points transformed from replay-state FK.",
    )
    parser.add_argument("--robot-points-asset", type=Path, default=None)
    parser.add_argument("--robot-num-points", type=int, default=32)
    parser.add_argument(
        "--robot-color", type=float, nargs=3, default=(0.0, 0.85, 1.0)
    )
    parser.add_argument(
        "--robot-point-size",
        type=float,
        default=12.0,
        help="Middle-arm marker size in the 3D view; defaults to the scene marker size.",
    )
    args = parser.parse_args()

    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    zarr_path = args.zarr_path.expanduser().resolve()
    metadata_path = zarr_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    source_zarr_path = metadata.get("source_zarr_path")
    if "source_dataset_path" not in metadata and source_zarr_path is not None:
        source_metadata_path = Path(source_zarr_path).with_suffix(".json")
        source_metadata = json.loads(source_metadata_path.read_text(encoding="utf-8"))
        metadata = {**source_metadata, **metadata}
    dataset_path = (
        args.dataset_path.expanduser().resolve()
        if args.dataset_path is not None
        else Path(metadata["source_dataset_path"]).expanduser().resolve()
    )
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    root = zarr.open(str(zarr_path), mode="r")
    point_clouds = root["data/point_cloud"]
    if point_clouds.shape[-1] != 6:
        raise ValueError(f"Expected XYZRGB point clouds, got shape {point_clouds.shape}")
    episode_ends = np.asarray(root["meta/episode_ends"][:], dtype=np.int64)
    start, end = episode_bounds(episode_ends, args.episode_index)
    num_frames = end - start
    if args.max_frames is not None:
        num_frames = min(num_frames, args.max_frames)
    source_episodes = metadata.get("episode_indices", list(range(len(episode_ends))))
    source_episode = int(source_episodes[args.episode_index])

    dataset_info = json.loads((dataset_path / "meta/info.json").read_text(encoding="utf-8"))
    video_feature = dataset_info["features"][args.camera_key]
    _, height, width = video_feature["shape"]
    source_fps = float(dataset_info["fps"])
    video = read_video_metadata(dataset_path, args.camera_key, source_episode)
    source_frame_offset = int(metadata.get("parameters", {}).get("trim_start_frames", 0))
    camera_frames = decode_video_segment(
        video_path=video["path"],
        start_time=video["start_time"] + source_frame_offset / source_fps,
        num_frames=num_frames,
        width=int(width),
        height=int(height),
    )

    crop_min = np.asarray(metadata["crop_min"], dtype=np.float32)
    crop_max = np.asarray(metadata["crop_max"], dtype=np.float32)
    robot_frames = None
    robot_asset = None
    if args.include_middle_arm:
        robot_frames, robot_asset = compute_middle_arm_points(
            root=root,
            start=start,
            num_frames=num_frames,
            metadata=metadata,
            asset_path=args.robot_points_asset,
            expected_num_points=args.robot_num_points,
        )
        robot_min = robot_frames.min(axis=(0, 1))
        robot_max = robot_frames.max(axis=(0, 1))
        span = np.maximum(
            np.maximum(crop_max, robot_max) - np.minimum(crop_min, robot_min),
            1e-3,
        )
        margin = 0.03 * span
        crop_min = np.minimum(crop_min, robot_min) - margin
        crop_max = np.maximum(crop_max, robot_max) + margin
    frame_indices = list(range(0, num_frames, args.frame_stride))
    suffix = f"_robot{args.robot_num_points}" if args.include_middle_arm else ""
    output_video = output_dir / f"episode_{source_episode:04d}_rgb256{suffix}.mp4"
    representative = {frame_indices[0], frame_indices[len(frame_indices) // 2], frame_indices[-1]}

    with imageio.get_writer(
        output_video,
        fps=source_fps / args.frame_stride,
        codec="libx264",
        quality=8,
        macro_block_size=16,
    ) as writer:
        for local_frame in frame_indices:
            rendered = render_frame(
                camera_rgb=camera_frames[local_frame],
                point_cloud=np.asarray(point_clouds[start + local_frame]),
                crop_min=crop_min,
                crop_max=crop_max,
                title=(
                    f"source episode {source_episode} | frame "
                    f"{source_frame_offset + local_frame}/{source_frame_offset + num_frames - 1}"
                ),
                elev=args.view_elev,
                azim=args.view_azim,
                robot_points=(
                    robot_frames[local_frame]
                    if robot_frames is not None
                    else None
                ),
                robot_color=tuple(args.robot_color),
                robot_point_size=args.robot_point_size,
            )
            writer.append_data(rendered)
            if local_frame in representative:
                imageio.imwrite(
                    output_dir
                    / f"episode_{source_episode:04d}_frame_{local_frame:04d}{suffix}.png",
                    rendered,
                )

    rgb_data = np.asarray(point_clouds[start : start + num_frames, :, 3:6])
    summary = {
        "zarr_path": str(zarr_path),
        "source_episode": source_episode,
        "source_frame_offset": source_frame_offset,
        "num_frames": num_frames,
        "point_cloud_shape": list(point_clouds.shape[1:]),
        "rgb_min": float(rgb_data.min()),
        "rgb_max": float(rgb_data.max()),
        "rgb_mean": [float(value) for value in rgb_data.mean(axis=(0, 1))],
        "video_path": str(output_video),
    }
    if robot_frames is not None:
        summary.update(
            {
                "robot_points_shape": list(robot_frames.shape[1:]),
                "robot_points_asset": str(robot_asset.asset_path),
                "robot_points_schema": robot_asset.schema_version,
                "robot_reference_body": robot_asset.reference_body_name,
                "robot_xyz_min": [float(value) for value in robot_frames.min(axis=(0, 1))],
                "robot_xyz_max": [float(value) for value in robot_frames.max(axis=(0, 1))],
                "combined_num_points": int(point_clouds.shape[1] + robot_frames.shape[1]),
            }
        )
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
