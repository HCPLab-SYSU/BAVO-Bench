"""Robola launcher for task-level MJCF editing.

This script is intentionally self-contained so it can be removed after debugging.
It keeps the task XML include structure intact, adapts asset paths for Robola's
MjSpec loader, and writes back only top-level elements owned by the task file.
"""

from __future__ import annotations

# Example:
# python robola_task_debug.py assets/tasks/block_stack/task.xml

import argparse
import contextlib
import copy
import io
import json
import logging
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
import xml.etree.ElementTree as ET

import mujoco
from robola.handlers.model_data import pack_model_data
from robola.handlers.model_save import save_model_data
from robola.server import RobolaServer


LOGGER = logging.getLogger(__name__)
WORLD_CHILD_TAGS = {"body", "geom", "site", "camera", "light"}


def _as_posix_relpath(path: Path, start: Path) -> str:
    return os.path.relpath(path, start).replace(os.sep, "/")


def _infer_assets_root(task_xml: Path) -> Path:
    for parent in [task_xml.parent, *task_xml.parents]:
        if (parent / "meshes").is_dir():
            return parent
    raise ValueError(f"Could not infer assets root from {task_xml}")


def _asset_dir(root: Path, value: str | None, fallback: str) -> Path:
    if not value:
        return root / fallback
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate
    rooted = root / candidate
    return rooted if rooted.exists() else root / fallback


def _resolve_asset_files(spec: mujoco.MjSpec, source_path: Path, assets_root: Path) -> mujoco.MjSpec:
    mesh_dir = _asset_dir(assets_root, getattr(spec.compiler, "meshdir", None), "meshes")
    texture_dir = _asset_dir(assets_root, getattr(spec.compiler, "texturedir", None), "meshes")

    spec.meshdir = _as_posix_relpath(mesh_dir, source_path.parent)
    spec.texturedir = _as_posix_relpath(texture_dir, source_path.parent)

    for collection, directory in ((spec.meshes, mesh_dir), (spec.textures, texture_dir)):
        for item in collection:
            filename = getattr(item, "file", "") or ""
            if not filename:
                continue
            path = Path(filename)
            if path.exists():
                item.file = path.name if path.parent == directory else str(path)
                continue
            candidate = directory / path.name
            if candidate.exists():
                item.file = path.name
    return spec


def load_task_spec(source_path: Path, assets_root: Path) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(source_path))
    return _resolve_asset_files(spec, source_path, assets_root)


def _strip_whitespace(element: ET.Element) -> None:
    if element.text is not None and not element.text.strip():
        element.text = None
    if element.tail is not None and not element.tail.strip():
        element.tail = None
    for child in list(element):
        _strip_whitespace(child)


def _canonical(element: ET.Element) -> str:
    clone = copy.deepcopy(element)
    _strip_whitespace(clone)
    return ET.tostring(clone, encoding="unicode", short_empty_elements=True)


def _named_key(element: ET.Element) -> tuple[str, str] | None:
    name = element.get("name")
    if not name:
        return None
    return (element.tag, name)


def _worldbody(root: ET.Element) -> ET.Element:
    world = root.find("worldbody")
    if world is None:
        raise ValueError("MJCF has no <worldbody>")
    return world


def _direct_world_children(root: ET.Element) -> list[ET.Element]:
    return [child for child in list(_worldbody(root)) if child.tag in WORLD_CHILD_TAGS]


def _task_owned_named_keys(task_xml: Path) -> set[tuple[str, str]]:
    tree = ET.parse(task_xml)
    keys: set[tuple[str, str]] = set()
    for child in _direct_world_children(tree.getroot()):
        key = _named_key(child)
        if key is None:
            raise ValueError(
                f"{task_xml} has an unnamed top-level <{child.tag}>. "
                "Name task-owned top-level elements before using Robola write-back."
            )
        keys.add(key)
    return keys


@dataclass
class BaselineOwnership:
    task_owned_keys: set[tuple[str, str]]
    scene_named: dict[tuple[str, str], str]
    scene_unnamed: Counter[str]


def _write_flat_model(spec: mujoco.MjSpec, model: mujoco.MjModel, data, target_path: Path) -> None:
    model_data = pack_model_data(spec, model, data)
    _save_model_data_quiet(spec, model, model_data, target_path)


def _save_model_data_quiet(
    spec: mujoco.MjSpec,
    model: mujoco.MjModel,
    model_data: dict,
    target_path: Path,
) -> None:
    with contextlib.redirect_stdout(io.StringIO()):
        save_model_data(spec, model, model_data, str(target_path))


def _temporary_xml_path(directory: Path, stem: str) -> Path:
    handle = tempfile.NamedTemporaryFile(
        delete=False,
        dir=str(directory),
        prefix=f".{stem}.robola_task_debug.",
        suffix=".xml",
    )
    handle.close()
    return Path(handle.name)


def build_baseline(task_xml: Path, assets_root: Path) -> BaselineOwnership:
    task_owned_keys = _task_owned_named_keys(task_xml)
    spec = load_task_spec(task_xml, assets_root)
    model = spec.compile()
    data = mujoco.MjData(model)
    temp_path = _temporary_xml_path(task_xml.parent, f"{task_xml.stem}.baseline")
    try:
        _write_flat_model(spec, model, data, temp_path)
        root = ET.parse(temp_path).getroot()
    finally:
        temp_path.unlink(missing_ok=True)

    scene_named: dict[tuple[str, str], str] = {}
    scene_unnamed: Counter[str] = Counter()
    for child in _direct_world_children(root):
        key = _named_key(child)
        if key in task_owned_keys:
            continue
        if key is None:
            scene_unnamed[_canonical(child)] += 1
        else:
            scene_named[key] = _canonical(child)

    return BaselineOwnership(
        task_owned_keys=task_owned_keys,
        scene_named=scene_named,
        scene_unnamed=scene_unnamed,
    )


def _format_key(key: tuple[str, str]) -> str:
    return f"<{key[0]} name=\"{key[1]}\">"


def extract_task_world_children(flat_xml: Path, baseline: BaselineOwnership) -> list[ET.Element]:
    root = ET.parse(flat_xml).getroot()
    matched_scene_named: set[tuple[str, str]] = set()
    scene_changes: list[str] = []
    unknown_unnamed: list[str] = []
    scene_unnamed = baseline.scene_unnamed.copy()
    task_children: list[ET.Element] = []

    for child in _direct_world_children(root):
        key = _named_key(child)
        signature = _canonical(child)
        if key is not None and key in baseline.scene_named:
            matched_scene_named.add(key)
            if signature != baseline.scene_named[key]:
                scene_changes.append(_format_key(key))
            continue
        if key is not None:
            task_children.append(copy.deepcopy(child))
            continue
        if scene_unnamed[signature] > 0:
            scene_unnamed[signature] -= 1
            continue
        unknown_unnamed.append(child.tag)

    missing_scene = sorted(set(baseline.scene_named) - matched_scene_named)
    missing_unnamed = sum(scene_unnamed.values())
    if scene_changes or missing_scene or missing_unnamed or unknown_unnamed:
        parts: list[str] = []
        if scene_changes:
            parts.append("changed scene-owned elements: " + ", ".join(scene_changes[:8]))
        if missing_scene:
            parts.append(
                "removed scene-owned elements: "
                + ", ".join(_format_key(key) for key in missing_scene[:8])
            )
        if missing_unnamed:
            parts.append(f"removed {missing_unnamed} unnamed scene-owned top-level elements")
        if unknown_unnamed:
            parts.append(
                "added unnamed top-level task elements: "
                + ", ".join(f"<{tag}>" for tag in unknown_unnamed[:8])
            )
        raise ValueError("; ".join(parts))

    if not task_children:
        raise ValueError("No task-owned top-level worldbody elements were found to write back.")
    return task_children


def _replace_worldbody(task_xml: Path, task_children: Iterable[ET.Element]) -> ET.ElementTree:
    tree = ET.parse(task_xml)
    root = tree.getroot()
    old_world = root.find("worldbody")
    new_world = ET.Element("worldbody")
    for child in task_children:
        new_world.append(copy.deepcopy(child))

    if old_world is None:
        insert_at = 0
        for index, child in enumerate(list(root)):
            if child.tag == "include":
                insert_at = index + 1
        root.insert(insert_at, new_world)
    else:
        children = list(root)
        root.remove(old_world)
        root.insert(children.index(old_world), new_world)

    ET.indent(tree, space="    ")
    return tree


def write_task_xml(task_xml: Path, task_children: list[ET.Element], assets_root: Path) -> None:
    tree = _replace_worldbody(task_xml, task_children)
    temp_path = _temporary_xml_path(task_xml.parent, f"{task_xml.stem}.write")
    try:
        tree.write(temp_path, encoding="utf-8", xml_declaration=False, short_empty_elements=True)
        with temp_path.open("ab") as handle:
            handle.write(b"\n")
        model = load_task_spec(temp_path, assets_root).compile()
        del model
        os.replace(temp_path, task_xml)
    finally:
        temp_path.unlink(missing_ok=True)


class TaskDebugRobolaServer(RobolaServer):
    def __init__(
        self,
        mjcf_path: Path,
        assets_root: Path,
        port: int = 9527,
        allowed_origin: str = "*",
        simulation_fps: float = 60.0,
    ):
        self.assets_root = assets_root
        self.baseline: BaselineOwnership | None = None
        super().__init__(
            mjcf_path=str(mjcf_path),
            port=port,
            allowed_origin=allowed_origin,
            simulation_fps=simulation_fps,
        )

    def _load_current_task_model(self) -> None:
        if not self.mjcf_path:
            raise ValueError("MJCF path is not set")
        self.spec = load_task_spec(self.mjcf_path, self.assets_root)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.baseline = build_baseline(self.mjcf_path, self.assets_root)
        if self.simulation:
            self.simulation.invalidate_metadata()

    def _preload_model(self):
        print(f"[TASK-DEBUG] Pre-loading Robola task model from: {self.mjcf_path}")
        self._load_current_task_model()
        print(f"[TASK-DEBUG] Model loaded successfully: {self.spec.modelname}")

    async def _handle_load_xml(self, websocket, message: dict):
        try:
            xml_file = message.get("xml_file")
            if xml_file:
                self.mjcf_path = Path(xml_file).resolve()
                self.work_dir = self.mjcf_path.parent
                self.assets_root = _infer_assets_root(self.mjcf_path)

            if not self.mjcf_path or not self.mjcf_path.exists():
                await websocket.send_text(
                    json.dumps({"type": "error", "message": f"File not found: {self.mjcf_path}"})
                )
                return False

            self._load_current_task_model()
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "model_loaded",
                        "message": f"Successfully loaded {self.mjcf_path.name}",
                        "model_info": {"name": self.spec.modelname, "file": str(self.mjcf_path)},
                    }
                )
            )
            await self._handle_get_model_data(websocket, {})
        except Exception as exc:
            LOGGER.exception("Failed to load task MJCF")
            await websocket.send_text(
                json.dumps({"type": "error", "message": f"Failed to load task model: {exc}"})
            )
            return False

    async def _handle_save_model(self, websocket, message: dict):
        try:
            model_data = message.get("modelData")
            if model_data is None:
                await websocket.send_text(
                    json.dumps(
                        {"type": "save_model_result", "status": "error", "message": "Missing model data"}
                    )
                )
                return False
            if self.baseline is None or self.mjcf_path is None:
                raise ValueError("No task model is loaded")

            flat_path = _temporary_xml_path(self.mjcf_path.parent, f"{self.mjcf_path.stem}.flat")
            try:
                _save_model_data_quiet(self.spec, self.model, model_data, flat_path)
                task_children = extract_task_world_children(flat_path, self.baseline)
                write_task_xml(self.mjcf_path, task_children, self.assets_root)
            finally:
                flat_path.unlink(missing_ok=True)

            self._load_current_task_model()
            await websocket.send_text(
                json.dumps(
                    {
                        "type": "save_model_result",
                        "status": "ok",
                        "body_name_map": self._build_body_name_map(),
                    },
                    separators=(",", ":"),
                )
            )
            LOGGER.info("Task XML saved successfully: %s", self.mjcf_path)
        except Exception as exc:
            LOGGER.exception("Failed to save task MJCF")
            await websocket.send_text(
                json.dumps({"type": "save_model_result", "status": "error", "message": str(exc)})
            )
            return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Serve a task-level MJCF in Robola without changing its include layout. "
            "Save writes back only task-owned top-level worldbody elements."
        )
    )
    parser.add_argument("mjcf_path", type=Path, help="Task XML under assets/tasks/...")
    parser.add_argument("--assets-root", type=Path, default=None, help="Assets root containing meshes/")
    parser.add_argument("--port", type=int, default=9527)
    parser.add_argument("--origin", default="*")
    parser.add_argument("--fps", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mjcf_path = args.mjcf_path.expanduser().resolve()
    if not mjcf_path.exists():
        raise FileNotFoundError(mjcf_path)
    assets_root = args.assets_root.expanduser().resolve() if args.assets_root else _infer_assets_root(mjcf_path)
    if not (assets_root / "meshes").is_dir():
        raise FileNotFoundError(f"Expected meshes directory under assets root: {assets_root}")

    server = TaskDebugRobolaServer(
        mjcf_path=mjcf_path,
        assets_root=assets_root,
        port=args.port,
        allowed_origin=args.origin,
        simulation_fps=args.fps,
    )
    server.run()


if __name__ == "__main__":
    main()
