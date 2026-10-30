"""Save before-and-after previews for blur and pixelation on the YuNet sample."""

import argparse
import sys
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.anonymizer import FaceAnonymizer
from app.services.face_detector import DEFAULT_MODEL_PATH


DEFAULT_SAMPLE = DEFAULT_MODEL_PATH.parent / "example_outputs" / "largest_selfie.jpg"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[1] / "tests" / "output"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, default=DEFAULT_SAMPLE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--strength", type=int, default=15)
    args = parser.parse_args()

    image = cv2.imread(str(args.image))
    if image is None:
        raise ValueError(f"Could not read image: {args.image}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    before_path = args.output_dir / f"{args.image.stem}_before.jpg"
    if not cv2.imwrite(str(before_path), image):
        raise OSError(f"Could not save original preview: {before_path}")

    anonymizer = FaceAnonymizer()
    for method in ("blur", "pixelate"):
        output, metadata = anonymizer.anonymize_image(
            image,
            method=method,
            strength=args.strength,
        )
        output_path = args.output_dir / f"{args.image.stem}_{method}.jpg"
        if not cv2.imwrite(str(output_path), output):
            raise OSError(f"Could not save {method} preview: {output_path}")
        print(
            f"{method}: faces={metadata.face_count}, "
            f"processed={metadata.processed_face_count}, output={output_path}"
        )
    print(f"before: {before_path}")


if __name__ == "__main__":
    main()
