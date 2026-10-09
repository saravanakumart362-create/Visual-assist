"""Build the door-detection training set.

    python training/prepare_doors.py

Two public sources are downloaded (about 700 MB, kept in training/datasets)
and merged into one YOLO-format dataset at training/datasets/doors:

  DoorDetect   1,213 indoor photos labelled with doors, handles, cabinet
               doors and refrigerator doors.
               https://github.com/MiguelARD/DoorDetect-Dataset
  Open Images  every validation/test photo that has a "Door" box (CC BY 2.0).
               https://storage.googleapis.com/openimages/web/index.html

Images are shrunk to at most 640 px so training on a CPU stays practical.
"""

import csv
import hashlib
import io
import sys
import urllib.request
import zipfile
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

DATASETS = Path(__file__).resolve().parent / "datasets"
OUT = DATASETS / "doors"

CLASSES = ["door", "handle", "cabinet door", "refrigerator door"]
MAX_SIDE = 640
VAL_PERCENT = 12

DOORDETECT_ZIP = DATASETS / "doordetect_src" / "doordetect.zip"
DOORDETECT_URL = "https://codeload.github.com/MiguelARD/DoorDetect-Dataset/zip/refs/heads/master"

OI_DIR = DATASETS / "openimages_src"
OI_META = "https://storage.googleapis.com/openimages/v5/"
OI_IMAGE = "https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg"
OI_SPLITS = {"validation": "val-bbox.csv", "test": "test-bbox.csv"}
OI_REMOTE = {"val-bbox.csv": "validation-annotations-bbox.csv",
             "test-bbox.csv": "test-annotations-bbox.csv",
             "classes.csv": "class-descriptions-boxable.csv"}
# Open Images class name -> our class index.
OI_CLASSES = {"Door": 0, "Door handle": 1}


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"downloading {dest.name} ...", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(dest)
    return dest


def split_of(name: str) -> str:
    """Stable train/val assignment, so re-running never moves an image."""
    bucket = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % 100
    return "val" if bucket < VAL_PERCENT else "train"


def save(name: str, image_bytes: bytes, boxes: list, counts: Counter) -> None:
    """Write one resized image and its label file. `boxes` = (class, cx, cy, w, h)."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        print(f"skipping unreadable image {name}: {exc}")
        return
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    split = split_of(name)
    image.save(OUT / "images" / split / f"{name}.jpg", quality=90)
    lines = [f"{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}" for c, cx, cy, w, h in boxes]
    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines), encoding="utf-8")
    counts[f"{split} images"] += 1
    for box in boxes:
        counts[f"{split} {CLASSES[box[0]]}"] += 1


def parse_yolo(text: str) -> list:
    boxes = []
    for line in text.splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        try:
            cls, cx, cy, w, h = int(parts[0]), *map(float, parts[1:])
        except ValueError:
            continue
        if 0 <= cls < len(CLASSES) and 0 < w <= 1 and 0 < h <= 1 and 0 <= cx <= 1 and 0 <= cy <= 1:
            boxes.append((cls, cx, cy, w, h))
    return boxes


def add_doordetect(counts: Counter) -> None:
    archive = zipfile.ZipFile(download(DOORDETECT_URL, DOORDETECT_ZIP))
    labels = {Path(n).stem: n for n in archive.namelist()
              if "/labels/" in n and n.endswith(".txt")}
    for name in archive.namelist():
        if "/images/" not in name or name.endswith("/"):
            continue
        stem = Path(name).stem
        if stem not in labels:
            continue
        boxes = parse_yolo(archive.read(labels[stem]).decode("utf-8", "replace"))
        save(f"dd_{stem}", archive.read(name), boxes, counts)


def open_images_boxes(split: str) -> dict:
    """image id -> boxes, for images whose every door is boxed individually."""
    classes = download(OI_META + OI_REMOTE["classes.csv"], OI_DIR / "classes.csv")
    with classes.open(encoding="utf-8") as handle:
        wanted = {mid: OI_CLASSES[name] for mid, name in csv.reader(handle)
                  if name in OI_CLASSES}
    table = download(OI_META + OI_REMOTE[OI_SPLITS[split]], OI_DIR / OI_SPLITS[split])
    boxes, rejected = defaultdict(list), set()
    with table.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            cls = wanted.get(row["LabelName"])
            if cls is None:
                continue
            # A "group of doors" box or a drawing would teach the wrong thing.
            if row["IsGroupOf"] == "1" or row["IsDepiction"] == "1":
                rejected.add(row["ImageID"])
                continue
            x1, x2, y1, y2 = (float(row[k]) for k in ("XMin", "XMax", "YMin", "YMax"))
            if x2 > x1 and y2 > y1:
                boxes[row["ImageID"]].append(
                    (cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1))
    return {image_id: found for image_id, found in boxes.items()
            if image_id not in rejected and any(b[0] == 0 for b in found)}


def fetch_image(split: str, image_id: str):
    cached = OI_DIR / "images" / split / f"{image_id}.jpg"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(
                    OI_IMAGE.format(split=split, image_id=image_id), timeout=60) as reply:
                cached.write_bytes(reply.read())
        except Exception as exc:
            print(f"could not fetch {image_id}: {exc}")
            return None
    return cached.read_bytes()


def add_open_images(counts: Counter) -> None:
    for split in OI_SPLITS:
        boxes = open_images_boxes(split)
        print(f"Open Images {split}: {len(boxes)} door images", flush=True)
        with ThreadPoolExecutor(16) as pool:
            images = pool.map(lambda image_id: fetch_image(split, image_id), boxes)
            for image_id, data in zip(boxes, images):
                if data:
                    save(f"oi_{image_id}", data, boxes[image_id], counts)


def main() -> None:
    for kind in ("images", "labels"):
        for split in ("train", "val"):
            (OUT / kind / split).mkdir(parents=True, exist_ok=True)
    counts = Counter()
    add_doordetect(counts)
    add_open_images(counts)

    names = "\n".join(f"  {i}: {name}" for i, name in enumerate(CLASSES))
    (OUT / "doors.yaml").write_text(
        f"path: {OUT.as_posix()}\ntrain: images/train\nval: images/val\nnames:\n{names}\n",
        encoding="utf-8")
    for key in sorted(counts):
        print(f"{key:30s} {counts[key]}")
    print(f"\nDataset ready: {OUT / 'doors.yaml'}")


if __name__ == "__main__":
    sys.exit(main())
