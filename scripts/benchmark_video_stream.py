"""Measure sequential real-YuNet WebSocket processing on a sample frame."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import time

import cv2
import numpy as np
from websockets.sync.client import connect


SAMPLE_IMAGE = (
    Path(__file__).resolve().parents[1]
    / "tests"
    / "output"
    / "largest_selfie_before.jpg"
)
FRAME_WIDTH = 640
FRAME_HEIGHT = 360
FRAME_COUNT = 5
MAX_FRAME_BYTES = 1_000_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default="ws://127.0.0.1:8000/api/video/ws",
        help="Video WebSocket URL for the running API",
    )
    args = parser.parse_args()

    source = cv2.imread(str(SAMPLE_IMAGE), cv2.IMREAD_COLOR)
    if source is None:
        raise RuntimeError(f"Sample image could not be loaded: {SAMPLE_IMAGE}")
    frame = cv2.resize(
        source,
        (FRAME_WIDTH, FRAME_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Sample frame could not be encoded as JPEG")
    payload = encoded.tobytes()
    if len(payload) > MAX_FRAME_BYTES:
        raise RuntimeError("Encoded sample exceeds the WebSocket frame limit")

    processing_times: list[float] = []
    round_trip_times: list[float] = []
    with connect(args.url, max_size=MAX_FRAME_BYTES, max_queue=1) as websocket:
        for sequence in range(FRAME_COUNT):
            request = {
                "type": "frame",
                "sequence": sequence,
                "method": "pixelate",
                "strength": 15,
            }
            started = time.perf_counter()
            websocket.send(json.dumps(request))
            websocket.send(payload)
            metadata = json.loads(websocket.recv())
            response = websocket.recv()
            round_trip_ms = (time.perf_counter() - started) * 1000

            if (
                metadata.get("type") != "frame_result"
                or metadata.get("sequence") != sequence
            ):
                raise RuntimeError(f"Unexpected frame response: {metadata}")
            output = cv2.imdecode(
                np.frombuffer(response, dtype=np.uint8),
                cv2.IMREAD_COLOR,
            )
            if output is None or output.shape != frame.shape:
                raise RuntimeError("Processed JPEG has invalid dimensions")

            processing_times.append(float(metadata["processing_ms"]))
            round_trip_times.append(round_trip_ms)
            print(
                f"frame={sequence} faces={metadata['face_count']} "
                f"backend_ms={metadata['processing_ms']} "
                f"roundtrip_ms={round_trip_ms:.1f}"
            )

    mean_round_trip = statistics.mean(round_trip_times)
    print(
        f"Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}; "
        f"round-trip mean={mean_round_trip:.1f} ms, "
        f"median={statistics.median(round_trip_times):.1f} ms, "
        f"sequential throughput={1000 / mean_round_trip:.1f} FPS"
    )
    print(
        f"Backend processing mean={statistics.mean(processing_times):.1f} ms; "
        f"runs={FRAME_COUNT}"
    )


if __name__ == "__main__":
    main()
