from pathlib import Path

from ultralytics import YOLO


ROOT = Path(__file__).resolve().parent
DATA_YAML = ROOT / "dataset" / "data.yaml"
PRETRAINED_WEIGHTS = ROOT / "models" / "sh17_yolov8m.pt"
TRAIN_IMAGES = ROOT / "dataset" / "train" / "images"
TRAIN_LABELS = ROOT / "dataset" / "train" / "labels"
VAL_IMAGES = ROOT / "dataset" / "val" / "images"
VAL_LABELS = ROOT / "dataset" / "val" / "labels"

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def validate_split(images_dir: Path, labels_dir: Path) -> int:
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset split requires both '{images_dir}' and '{labels_dir}'."
        )

    images = [
        path for path in images_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    ]
    if not images:
        raise ValueError(
            f"No training images found in '{images_dir}'. "
            "Add labeled images before starting training."
        )

    labels = {path.stem for path in labels_dir.rglob("*.txt")}
    missing_labels = [path.name for path in images if path.stem not in labels]
    if missing_labels:
        preview = ", ".join(missing_labels[:5])
        raise ValueError(
            f"{len(missing_labels)} image(s) have no matching YOLO label in "
            f"'{labels_dir}' (for example: {preview})."
        )

    return len(images)


def main() -> None:
    if not DATA_YAML.is_file():
        raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")
    if not PRETRAINED_WEIGHTS.is_file():
        raise FileNotFoundError(
            f"SH17 pretrained weights not found: {PRETRAINED_WEIGHTS}"
        )

    train_count = validate_split(TRAIN_IMAGES, TRAIN_LABELS)
    val_count = validate_split(VAL_IMAGES, VAL_LABELS)
    print(f"Training on {train_count} images; validating on {val_count} images.")

    model = YOLO(str(PRETRAINED_WEIGHTS))
    model.train(
        data=str(DATA_YAML),
        epochs=100,
        imgsz=960,
        batch=4,
        patience=20,
        workers=0,
        close_mosaic=10,
        seed=42,
        deterministic=True,
        project=str(ROOT / "runs" / "detect"),
        name="cca-ppe-yolov8m",
        exist_ok=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
