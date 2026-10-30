"""FastAPI tests for the /api/anonymize-video endpoint."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock

import cv2
from fastapi.testclient import TestClient
import numpy as np

from app.main import ApiSettings, create_app
from app.services.anonymizer import FaceAnonymizer
from app.services.face_detector import BoundingBox, FaceDetection


def fake_detection(x: float, y: float, width: float, height: float) -> FaceDetection:
    return FaceDetection(
        BoundingBox(x, y, width, height),
        0.9,
        ((1.0, 1.0),) * 5,
    )


def write_test_video_bytes(frames: int = 8, fps: float = 12.0) -> bytes:
    directory = Path(tempfile.mkdtemp())
    path = directory / "input.webm"
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"VP80"),
        fps,
        (160, 120),
    )
    if not writer.isOpened():
        raise RuntimeError("test fixture writer could not open")
    try:
        for index in range(frames):
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[:] = ((index * 8) % 256, 60, 120)
            writer.write(frame)
    finally:
        writer.release()
    return path.read_bytes()


class VideoUploadApiTests(unittest.TestCase):
    def make_client(
        self,
        detector_result: list[FaceDetection] | None = None,
        settings: ApiSettings | None = None,
        factory=None,
    ) -> tuple[TestClient, Mock | None]:
        detector = None
        if factory is None:
            detector = Mock()
            detector.detect.return_value = detector_result or []
            anonymizer = FaceAnonymizer(detector)
            factory = lambda: anonymizer
        application = create_app(
            anonymizer_factory=factory,
            settings=settings,
        )
        client = TestClient(application)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client, detector

    def post_video(
        self,
        client: TestClient,
        payload: bytes | None = None,
        filename: str = "input.webm",
        method: str = "blur",
        strength: int = 15,
        margin: float = 0.0,
    ):
        return client.post(
            "/api/anonymize-video",
            files={
                "file": (
                    filename,
                    payload if payload is not None else write_test_video_bytes(),
                    "video/webm",
                )
            },
            data={"method": method, "strength": str(strength), "margin": str(margin)},
        )

    def test_processed_video_returns_video_payload_and_metadata(self) -> None:
        face = fake_detection(10, 10, 40, 40)
        client, detector = self.make_client([face])
        response = self.post_video(client, method="pixelate", strength=8)

        self.assertEqual(response.status_code, 200, response.text)
        self.assertTrue(response.headers["content-type"].startswith("video/"))
        self.assertEqual(response.headers["x-video-frame-count"], "8")
        self.assertEqual(response.headers["x-video-processed-frame-count"], "8")
        self.assertEqual(response.headers["x-video-face-count"], "8")
        self.assertEqual(response.headers["x-anonymization-method"], "pixelate")
        self.assertEqual(response.headers["x-anonymization-strength"], "8")
        self.assertEqual(response.headers["x-anonymization-margin"], "0.0")
        self.assertEqual(response.headers["x-anonymized"], "true")

        output_path = Path("decoded_video_check.webm")
        try:
            output_path.write_bytes(response.content)
            capture = cv2.VideoCapture(str(output_path))
            read_frames = 0
            while True:
                ok, _ = capture.read()
                if not ok:
                    break
                read_frames += 1
            capture.release()
            self.assertEqual(read_frames, 8)
        finally:
            output_path.unlink(missing_ok=True)
        self.assertEqual(detector.detect.call_count, 8)

    def test_zero_faces_return_valid_video_and_false_anonymized(self) -> None:
        client, _ = self.make_client([])
        response = self.post_video(client)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-video-face-count"], "0")
        self.assertEqual(response.headers["x-video-processed-frame-count"], "0")
        self.assertEqual(response.headers["x-anonymized"], "false")
        self.assertTrue(response.headers["content-type"].startswith("video/"))

    def test_multiple_faces_are_reported(self) -> None:
        client, _ = self.make_client(
            [fake_detection(5, 5, 15, 15), fake_detection(50, 35, 20, 20)]
        )
        response = self.post_video(client)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-video-face-count"], "16")
        self.assertEqual(response.headers["x-video-processed-frame-count"], "16")

    def test_invalid_options_are_rejected(self) -> None:
        client, _ = self.make_client()
        cases = (
            {"method": "mosaic"},
            {"strength": 51},
            {"strength": 0},
            {"method": "pixelate", "strength": 1},
            {"margin": 1.1},
            {"margin": -0.1},
        )
        for options in cases:
            with self.subTest(options=options):
                response = self.post_video(client, **options)
                self.assertEqual(response.status_code, 422)

    def test_missing_upload_is_rejected(self) -> None:
        client, _ = self.make_client()
        response = client.post("/api/anonymize-video", data={"method": "blur"})
        self.assertEqual(response.status_code, 422)

    def test_empty_upload_is_rejected(self) -> None:
        client, _ = self.make_client()
        response = self.post_video(client, payload=b"")
        self.assertEqual(response.status_code, 400)

    def test_corrupt_upload_is_rejected(self) -> None:
        client, _ = self.make_client()
        response = self.post_video(
            client,
            payload=b"\x1aE\xdf\xa3 not a real webm",
        )
        self.assertEqual(response.status_code, 400)

    def test_not_ready_returns_503(self) -> None:
        def fail_initialization():
            raise FileNotFoundError("private model path")

        client, _ = self.make_client(factory=fail_initialization)
        response = self.post_video(client)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private model path", response.text)

    def test_processing_error_is_safe_and_does_not_leak_details(self) -> None:
        client, detector = self.make_client([fake_detection(1, 1, 20, 20)])
        detector.detect.side_effect = RuntimeError("private decoder detail")
        response = self.post_video(client)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Video processing failed")
        self.assertNotIn("private decoder detail", response.text)

    def test_long_video_is_rejected(self) -> None:
        client, _ = self.make_client(
            settings=ApiSettings(max_video_frames=3)
        )
        response = self.post_video(client)
        self.assertEqual(response.status_code, 413)

    def test_cors_exposes_video_headers_to_vite_origin(self) -> None:
        client, _ = self.make_client()
        response = client.get(
            "/api/health",
            headers={"Origin": "http://localhost:5173"},
        )
        self.assertIn(
            "X-Video-Frame-Count",
            response.headers["access-control-expose-headers"],
        )


if __name__ == "__main__":
    unittest.main()
