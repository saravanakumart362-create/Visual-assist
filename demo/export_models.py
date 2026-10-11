"""Convert the trained models for the demo page's live camera.

    pip install onnx onnxslim
    python demo/export_models.py

A browser cannot run PyTorch, so the models are converted to ONNX and
written to demo/models/, with models.json describing each one: its input
size, its confidence threshold, and its class names as the assistant
speaks them. All of that is taken from config.yaml, so the page detects
with the same settings as the device. Run this again after retraining a
model or changing those settings.

The two face models are ONNX files already, so they are only copied. A
model that has not changed since it was last converted is left alone.
"""

import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))

from ultralytics import YOLO  # noqa: E402

from visual_assist import config, faces  # noqa: E402

OUT = HERE / "models"
OPSET = 17  # well within what onnxruntime-web supports


def convert(weights: Path, size: int) -> tuple:
    """Export one model. Returns (file name, class names by index)."""
    model = YOLO(str(weights))
    target = OUT / f"{weights.stem}.onnx"
    if target.exists() and target.stat().st_mtime >= weights.stat().st_mtime:
        print(f"{target.name}: up to date")
    else:
        exported = Path(model.export(format="onnx", imgsz=size, opset=OPSET, simplify=True))
        shutil.move(str(exported), target)
        print(f"{target.name}: {target.stat().st_size / 1e6:.1f} MB")
    names = [model.names[index] for index in sorted(model.names)]
    return target.name, names


def copy(model: Path) -> str:
    """Put a model that is already an ONNX file beside the others."""
    if not faces.fetch(model):
        raise SystemExit(f"{model} is missing and could not be downloaded")
    target = OUT / model.name
    if not target.exists() or target.stat().st_size != model.stat().st_size:
        shutil.copyfile(model, target)
        print(f"{target.name}: {target.stat().st_size / 1e6:.1f} MB")
    return target.name


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

    # Faces: found by one model, told apart by another (see visual_assist/faces.py).
    manifest["faces"] = {
        "detector": {"file": copy(cfg.path(cfg.faces.detector)), "size": 640},
        "recogniser": {"file": copy(cfg.path(cfg.faces.recogniser)), "size": 112},
        "score": faces.DETECTION_SCORE,  # how sure the detector must be of a face
        "min_px": cfg.faces.min_face_px,  # narrower faces are not looked at
        "most": faces.MAX_FACES,  # faces looked at in one picture
    }

    (OUT / "models.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"wrote {OUT / 'models.json'}")


if __name__ == "__main__":
    main()
