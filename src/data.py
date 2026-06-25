import json
import os
from pathlib import Path

import cv2
import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **_kwargs):
        return iterable


def create_mask(lanes, h_samples, image_height=720, image_width=1280, line_width=5):
    mask = np.zeros((image_height, image_width), dtype=np.uint8)

    for lane in lanes:
        points = []

        for x, y in zip(lane, h_samples):
            if x == -2:
                continue
            points.append((int(x), int(y)))

        if len(points) > 1:
            cv2.polylines(mask, [np.array(points)], isClosed=False, color=255, thickness=line_width)

    return mask


def read_tusimple_json(json_paths):
    if isinstance(json_paths, (str, os.PathLike)):
        json_paths = [json_paths]

    samples = []
    for json_path in json_paths:
        with open(json_path, "r", encoding="utf-8") as file:
            samples.extend(json.loads(line) for line in file if line.strip())

    return samples


def load_tusimple_data(
    dataset_root,
    json_paths,
    image_size=(256, 256),
    limit=200,
    line_width=5,
):
    dataset_root = Path(dataset_root)
    samples = read_tusimple_json(json_paths)

    if limit:
        samples = samples[:limit]

    images = []
    masks = []

    target_width, target_height = image_size

    for item in tqdm(samples, desc="Loading TuSimple"):
        image_path = dataset_root / item["raw_file"]

        image = cv2.imread(str(image_path))
        if image is None:
            continue

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = create_mask(
            lanes=item["lanes"],
            h_samples=item["h_samples"],
            image_height=image.shape[0],
            image_width=image.shape[1],
            line_width=line_width,
        )

        image = cv2.resize(image, (target_width, target_height))
        mask = cv2.resize(mask, (target_width, target_height), interpolation=cv2.INTER_NEAREST)

        images.append(image.astype(np.float32) / 255.0)
        masks.append(mask.astype(np.float32) / 255.0)

    return np.array(images), np.array(masks)[..., np.newaxis]
