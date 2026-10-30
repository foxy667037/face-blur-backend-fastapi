"""Run YuNet on the bundled sample and optionally save an annotated copy."""

import argparse
import sys
from pathlib import Path

import cv2

# Support direct execution from the repository root as well as module use.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.face_detector import DEFAULT_MODEL_PATH, FaceDetector


DEFAULT_SAMPLE_PATH = (
    DEFAULT_MODEL_PATH.parent / "example_outputs" / "largest_selfie.jpg"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        type=Path,
        default=DEFAULT_SAMPLE_PATH,
        help="Input image (defaults to the bundled YuNet sample)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for a separate annotated preview image",
    )
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    detections = FaceDetector().detect(image)
    print(f"Detected faces: {len(detections)}")
    for index, detection in enumerate(detections, start=1):
        box = detection.bounding_box
        print(
            f"{index}: bbox=({box.x:.2f}, {box.y:.2f}, "
            f"{box.width:.2f}, {box.height:.2f}), "
            f"confidence={detection.confidence:.6f}"
        )

    if args.output is not None:
        preview = image.copy()
        for detection in detections:
            box = detection.bounding_box
            start = (round(box.x), round(box.y))
            end = (round(box.x + box.width), round(box.y + box.height))
            cv2.rectangle(preview, start, end, (0, 255, 0), 2)
        if not cv2.imwrite(str(args.output), preview):
            raise OSError(f"Could not save annotated preview: {args.output}")
        print(f"Annotated preview saved to: {args.output}")


if __name__ == "__main__":
    main()
