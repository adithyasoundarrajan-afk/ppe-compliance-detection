from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
TEST_IMAGES = ROOT / "dataset" / "test" / "images"
TEST_LABELS = ROOT / "dataset" / "test" / "labels"
TRAINED_WEIGHTS = (
    ROOT / "runs" / "detect" / "cca-ppe-yolov8m" / "weights" / "best.pt"
)
PRETRAINED_WEIGHTS = ROOT / "models" / "sh17_yolov8m.pt"
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def main() -> None:
    images = [
        path for path in TEST_IMAGES.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise ValueError(
            f"No test images found in '{TEST_IMAGES}'. "
            "Add a labeled, held-out test set to measure accuracy."
        )

    label_stems = {path.stem for path in TEST_LABELS.rglob("*.txt")}
    missing_labels = [path.name for path in images if path.stem not in label_stems]
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise ValueError(
            f"{len(missing_labels)} test image(s) have no matching YOLO label "
            f"in '{TEST_LABELS}' (for example: {preview})."
        )

    weights = TRAINED_WEIGHTS if TRAINED_WEIGHTS.is_file() else PRETRAINED_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights}")

    print(f"Evaluating {weights.name} on {len(images)} held-out test images.")
    model = YOLO(str(weights))
    metrics = model.val(
        data=str(DATA_YAML),
        split="test",
        imgsz=960,
        plots=True,
        project=str(ROOT / "runs" / "val"),
        name="cca-ppe-yolov8m-test",
        exist_ok=True,
    )
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP@0.50:   {metrics.box.map50:.4f}")
    print(f"mAP@0.50:0.95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
