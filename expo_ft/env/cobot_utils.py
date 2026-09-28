"""Load Cobot Magic LeRobot demos into the same step dicts as process_droid_dataset."""

import os
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from tqdm import tqdm

try:
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
except ImportError:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

from openpi_client import image_tools

_REQUIRED = (
    "observation.state",
    "action",
    "observation.images.cam_high",
    "observation.images.cam_right_wrist",
)
_IMAGE_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_right_wrist",
)
_IMAGE_SIZE = 224


def _to_hwc_uint8(image):
    if hasattr(image, "detach"):
        image = image.detach().cpu().numpy()
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] == 3:
        image = np.transpose(image, (1, 2, 0))
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    else:
        image = image.astype(np.uint8)
    return image_tools.resize_with_pad(image, _IMAGE_SIZE, _IMAGE_SIZE)


def _as_numpy(values):
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    first = values[0]
    if hasattr(first, "detach"):
        return np.stack([np.asarray(v.detach().cpu().numpy()) for v in values])
    return np.asarray(values)


def _load_episode_images(hf_dataset, ep_idx, start, end):
    """Images live in the episode parquet. Same source as LeRobotDataset.__getitem__."""
    rows = hf_dataset[start:end]
    n_frames = end - start
    decoded = {}
    for key in _IMAGE_KEYS:
        images = rows[key]
        if len(images) != n_frames:
            raise ValueError(
                f"Episode {ep_idx} column {key} has {len(images)} frames, expected {n_frames}"
            )
        decoded[key] = [_to_hwc_uint8(img) for img in images]
    return ep_idx, decoded


def process_cobot_dataset(datapath, task_config, num_data=None):
    """Read a LeRobot dataset root (the directory that contains meta/info.json).

    Camera features are dtype ``image``, so frames come from the parquet files,
    matching OpenPI's LeRobotDataset. Episodes are decoded in parallel. The
    small mp4 files next to the parquet are not the training images.
    """
    root = os.path.abspath(datapath)
    info = os.path.join(root, "meta", "info.json")
    if not os.path.isfile(info):
        raise FileNotFoundError(f"LeRobot dataset root must contain meta/info.json: {root}")

    parent = os.path.basename(os.path.dirname(root))
    name = os.path.basename(root)
    episodes = list(range(int(num_data))) if num_data is not None and int(num_data) > 0 else None
    dataset = LeRobotDataset(repo_id=f"{parent}/{name}", root=root, episodes=episodes)

    episode_ids = list(range(dataset.num_episodes)) if episodes is None else list(episodes)
    from_idx = np.asarray(dataset.episode_data_index["from"]).reshape(-1)
    to_idx = np.asarray(dataset.episode_data_index["to"]).reshape(-1)

    states = _as_numpy(dataset.hf_dataset["observation.state"]).astype(np.float32, copy=False)
    actions = _as_numpy(dataset.hf_dataset["action"]).astype(np.float32, copy=False)
    task_index = _as_numpy(dataset.hf_dataset["task_index"]).reshape(-1)
    tasks = dataset.meta.tasks
    fallback_prompt = getattr(task_config, "language_instruction", "") or ""

    jobs = []
    for pos, ep_idx in enumerate(episode_ids):
        start = int(from_idx[pos])
        end = int(to_idx[pos])
        jobs.append((ep_idx, start, end))

    workers = int(os.environ.get("COBOT_LOAD_WORKERS", "8"))
    workers = max(1, min(workers, len(jobs) or 1))
    decoded_by_ep = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_load_episode_images, dataset.hf_dataset, ep_idx, start, end)
            for ep_idx, start, end in jobs
        ]
        for fut in tqdm(as_completed(futures), total=len(futures), desc="cobot lerobot"):
            ep_idx, decoded = fut.result()
            decoded_by_ep[ep_idx] = decoded

    data = []
    for ep_idx, start, end in jobs:
        images = decoded_by_ep.pop(ep_idx)
        for i, frame_i in enumerate(range(start, end)):
            state = np.asarray(states[frame_i], np.float32).reshape(-1)
            action = np.asarray(actions[frame_i], np.float32).reshape(-1)
            if state.shape[0] != 7 or action.shape[0] != 7:
                raise ValueError(f"Expected 7D state/action, got {state.shape} {action.shape}")
            prompt = tasks.get(int(task_index[frame_i]), fallback_prompt) or fallback_prompt
            done = 1.0 if frame_i == end - 1 else 0.0
            data.append({
                "observations": {
                    "observation.images.cam_high": images["observation.images.cam_high"][i],
                    "observation.images.cam_right_wrist": images["observation.images.cam_right_wrist"][i],
                    "observation.state": state,
                    "prompt": str(prompt),
                },
                "actions": action,
                "rewards": np.float32(done),
                "masks": np.float32(1.0 - done),
                "dones": np.float32(done),
            })
    if not data:
        raise ValueError(f"No frames loaded from {root}")
    return data
