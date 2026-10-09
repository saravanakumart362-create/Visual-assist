"""Build the banknote-recognition training set.

    python training/prepare_doors.py       (first: supplies some "no note" photos)
    python training/prepare_currency.py

Banknotes: "Dataset of Indian (Rupees) and Thai (Baht) Banknotes with
Annotations", Meshram et al., Mendeley Data, CC BY 4.0 (540 MB download).
https://data.mendeley.com/datasets/2kfz5yc7pt/1

Its 2,000 Indian banknote photos are sorted into one folder per class at
training/datasets/currency/{train,val}/<class>, which is the layout the
YOLO classifier trains on.

A "none" class teaches the model to say "there is no note here". It must
look like whatever else the camera may be pointed at, above all the person
holding the note up: so it is made of Open Images photos with large faces
and hands, other everyday Open Images photos, and the indoor photos of the
door dataset (about 200 MB more to download).
"""

import csv
import hashlib
import io
import shutil
import sys
import urllib.request
import zipfile
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

DATASETS = Path(__file__).resolve().parent / "datasets"
OUT = DATASETS / "currency"
DOOR_PHOTOS = DATASETS / "doors" / "images"

ARCHIVE = DATASETS / "banknotes_src" / "banknotes.zip"
URL = ("https://data.mendeley.com/public-files/datasets/2kfz5yc7pt/files/"
       "e8463ff2-e55b-4d48-a985-ddaefbc23595/file_downloaded")

OI_DIR = DATASETS / "openimages_src"
OI_META = "https://storage.googleapis.com/openimages/v5/"
OI_IMAGE = "https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"

MAX_SIDE = 512
VAL_PERCENT = 20
SPLITS = {"Training": "train", "Validation": "val"}
# Old and new series look different, so they are separate classes; the
# assistant only speaks the number in front of the underscore.
CLASSES = {
    "INDIA10NEW": "10_new", "INDIA10OLD": "10_old", "INDIA20": "20",
    "INDIA50NEW": "50_new", "INDIA50OLD": "50_old",
    "INDIA100NEW": "100_new", "INDIA100OLD": "100_old",
    "INDIA200": "200", "INDIA500": "500", "INDIA2000": "2000",
}
DOOR_PHOTOS_PER_SPLIT = {"train": 320, "val": 80}
# Open Images photos for the "none" class: (label, least share of the
# picture its box must fill, how many). None = any photo at all.
EVERYDAY = [("Human face", 0.08, 300), ("Human hand", 0.04, 200), (None, 0.0, 300)]


def mixed(names) -> list:
    """Photo ids or paths in a shuffled but repeatable order."""
    return sorted(names, key=lambda name: hashlib.md5(Path(name).name.encode()).hexdigest())


def split_of(name: str) -> str:
    bucket = int(hashlib.md5(name.encode("utf-8")).hexdigest(), 16) % 100
    return "val" if bucket < VAL_PERCENT else "train"


def download(url: str, dest: Path) -> Path:
    if not dest.exists():
        print(f"downloading {dest.name} ...", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        partial = dest.with_suffix(dest.suffix + ".part")
        urllib.request.urlretrieve(url, partial)
        partial.replace(dest)
    return dest


def save(image_bytes: bytes, dest: Path) -> bool:
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        print(f"skipping unreadable image {dest.name}: {exc}")
        return False
    image.thumbnail((MAX_SIDE, MAX_SIDE))
    dest.parent.mkdir(parents=True, exist_ok=True)
    image.save(dest, quality=90)
    return True


def add_banknotes(counts: Counter) -> None:
    archive = zipfile.ZipFile(download(URL, ARCHIVE))
    for name in archive.namelist():
        parts = Path(name).parts
        if len(parts) < 4 or parts[-4] != "IndianBankNotes" or not name.lower().endswith(".jpg"):
            continue
        split, label = SPLITS.get(parts[-3]), CLASSES.get(parts[-2])
        if split and label and save(archive.read(name), OUT / split / label / parts[-1]):
            counts[f"{split} {label}"] += 1


def everyday_ids() -> list:
    """Open Images validation photos to use as "no note" examples."""
    classes = download(OI_META + "class-descriptions-boxable.csv", OI_DIR / "classes.csv")
    with classes.open(encoding="utf-8") as handle:
        code_of = {name: code for code, name in csv.reader(handle)}
    table = download(OI_META + "validation-annotations-bbox.csv", OI_DIR / "val-bbox.csv")
    largest = {}  # (image id, label code) -> biggest box, as a share of the picture
    with table.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            share = ((float(row["XMax"]) - float(row["XMin"]))
                     * (float(row["YMax"]) - float(row["YMin"])))
            key = (row["ImageID"], row["LabelName"])
            largest[key] = max(largest.get(key, 0.0), share)

    chosen = []
    for label, least, wanted in EVERYDAY:
        fits = {image_id for (image_id, code), share in largest.items()
                if (label is None or code == code_of[label]) and share >= least}
        chosen += [i for i in mixed(fits) if i not in chosen][:wanted]
    return chosen


def fetch(image_id: str):
    cached = OI_DIR / "images" / "validation" / f"{image_id}.jpg"
    if not cached.exists():
        cached.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(OI_IMAGE.format(image_id=image_id), timeout=60) as reply:
                cached.write_bytes(reply.read())
        except Exception as exc:
            print(f"could not fetch {image_id}: {exc}")
            return None
    return cached.read_bytes()


def add_no_note_photos(counts: Counter) -> None:
    ids = everyday_ids()
    print(f"Open Images: {len(ids)} everyday photos", flush=True)
    with ThreadPoolExecutor(16) as pool:
        for image_id, data in zip(ids, pool.map(fetch, ids)):
            split = split_of(image_id)
            if data and save(data, OUT / split / "none" / f"oi_{image_id}.jpg"):
                counts[f"{split} none"] += 1
    for split, wanted in DOOR_PHOTOS_PER_SPLIT.items():
        for photo in mixed((DOOR_PHOTOS / split).glob("*.jpg"))[:wanted]:
            if save(photo.read_bytes(), OUT / split / "none" / photo.name):
                counts[f"{split} none"] += 1


def main() -> int:
    if not DOOR_PHOTOS.is_dir():
        print("Run training/prepare_doors.py first: its photos are part of the 'none' class.")
        return 1
    if OUT.exists():
        shutil.rmtree(OUT)  # rebuild from scratch, so old choices cannot linger
    counts = Counter()
    add_banknotes(counts)
    add_no_note_photos(counts)
    for key in sorted(counts):
        print(f"{key:20s} {counts[key]}")
    print(f"\nDataset ready: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
