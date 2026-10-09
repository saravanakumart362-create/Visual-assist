"""Convert the trained models for the demo page's live camera.

    pip install onnx onnxslim
    python demo/export_models.py

A browser cannot run PyTorch, so the models are converted to ONNX and
written to demo/models/, with models.json describing each one: its input
size, its confidence threshold, and its class names as the assistant
speaks them. All of that is taken from config.yaml, so the page detects
with the same settings as the device. Run this again after retraining a
model or changing those settings.
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

from visual_assist import config  # noqa: E402

OUT = HERE / "models"
OPSET = 17  # well within what onnxruntime-web supports


def convert(weights: Path, size: int) -> tuple:
    """Export one model. Returns (file name, class names by index)."""
    model = YOLO(str(weights))
    exported = Path(model.export(format="onnx", imgsz=size, opset=OPSET, simplify=True))
    target = OUT / f"{weights.stem}.onnx"
    shutil.move(str(exported), target)
    names = [model.names[index] for index in sorted(model.names)]
    print(f"{target.name}: {target.stat().st_size / 1e6:.1f} MB, {len(names)} classes")
    return target.name, names


def main() -> None:
    cfg = config.load(ROOT / "config.yaml")
    OUT.mkdir(exist_ok=True)
    manifest = {"iou": 0.7, "detectors": []}  # 0.7 is the detector's own overlap limit

    sources = [dict(weights=cfg.detector.weights, classes=None)] + list(cfg.detector.extra or [])
    for source in sources:
        size = source.get("imgsz", cfg.detector.imgsz)
        file, names = convert(cfg.path(source["weights"]), size)
        aliases = {**cfg.detector.aliases, **source.get("aliases", {})}
        names = [aliases.get(name, name) for name in names]
        keep = source.get("classes")
        manifest["detectors"].append({
            "file": file,
            "size": size,
            "confidence": source.get("confidence", cfg.detector.confidence),
            "names": names,
            # Only these classes are reported (null = all of them).
            "keep": None if not keep else [i for i, name in enumerate(names) if name in keep],
        })

    weights = cfg.path(cfg.currency.weights)
    size = YOLO(str(weights)).overrides.get("imgsz", 256)
    file, names = convert(weights, size)
    manifest["currency"] = {"file": file, "size": size, "names": names}

    (OUT / "models.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'models.json'}")


if __name__ == "__main__":
    main()
