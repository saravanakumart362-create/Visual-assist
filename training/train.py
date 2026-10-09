"""Train the project's models. Works on a laptop CPU; much faster on Colab.

    python training/train.py doors       door detector       -> weights/doors.pt
    python training/train.py currency    banknote classifier -> weights/currency.pt
    python training/train.py custom --data my.yaml --name indoor
                                         your own YOLO dataset -> weights/indoor.pt

Run the matching prepare_*.py script first to build the dataset. Options
after the preset name override its defaults.

The defaults are sized for a laptop CPU: about an hour and a half for
"doors" (smaller images, fewer epochs, pretrained backbone kept frozen)
and ten minutes for "currency". With a GPU, train the whole door network
for longer:

    python training/train.py doors --imgsz 640 --epochs 100 --freeze 0
"""

import argparse
import shutil
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE.parent / "weights"

PRESETS = {
    "doors": dict(data=HERE / "datasets" / "doors" / "doors.yaml",
                  base="yolov5nu.pt", imgsz=512, epochs=30, batch=16, freeze=10),
    # Banknotes are easy to tell apart: accuracy levels off within a few epochs.
    "currency": dict(data=HERE / "datasets" / "currency",
                     base="yolov8n-cls.pt", imgsz=256, epochs=6, batch=32, freeze=0),
    "custom": dict(data=HERE / "dataset.yaml",
                   base="yolov5nu.pt", imgsz=640, epochs=100, batch=16, freeze=0),
}
# Notes are shown at any angle, so the classifier also trains on flipped ones.
EXTRA = {"currency": dict(flipud=0.5)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("preset", choices=PRESETS)
    parser.add_argument("--data", help="dataset .yaml (detection) or folder (classification)")
    parser.add_argument("--base", help="starting weights, e.g. yolov5su.pt for more accuracy")
    parser.add_argument("--name", help="output name: weights/<name>.pt")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--freeze", type=int,
                        help="keep the first N layers fixed (10 = the backbone, 0 = none)")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = parser.parse_args()

    settings = dict(PRESETS[args.preset])
    for key in ("data", "base", "epochs", "imgsz", "batch", "freeze"):
        if getattr(args, key) is not None:
            settings[key] = getattr(args, key)
    name = args.name or args.preset
    run_dir = HERE / "runs" / name

    WEIGHTS.mkdir(exist_ok=True)
    if args.resume:
        model = YOLO(run_dir / "weights" / "last.pt")
        model.train(resume=True)
    else:
        model = YOLO(WEIGHTS / settings["base"])
        model.train(
            data=str(settings["data"]), epochs=settings["epochs"],
            imgsz=settings["imgsz"], batch=settings["batch"], workers=args.workers,
            project=str(HERE / "runs"), name=name, exist_ok=True, patience=10,
            freeze=settings["freeze"] or None,
            **EXTRA.get(args.preset, {}),
        )

    best = run_dir / "weights" / "best.pt"
    shutil.copy(best, WEIGHTS / f"{name}.pt")
    metrics = YOLO(best).val(data=str(settings["data"]), imgsz=settings["imgsz"],
                             project=str(HERE / "runs"), name=f"{name}_val", exist_ok=True)
    print(f"\nSaved {WEIGHTS / (name + '.pt')}")
    if hasattr(metrics, "box"):
        print(f"mAP50 {metrics.box.map50:.3f}   mAP50-95 {metrics.box.map:.3f}")
        for index, class_name in metrics.names.items():
            if index in metrics.box.ap_class_index:
                position = list(metrics.box.ap_class_index).index(index)
                print(f"  {class_name:20s} AP50 {metrics.box.ap50[position]:.3f}")
    else:
        print(f"top-1 accuracy {metrics.top1:.3f}")


if __name__ == "__main__":
    main()
