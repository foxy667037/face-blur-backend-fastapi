from io import BytesIO
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image

from app.main import ApiSettings, create_app
from app.services.anonymizer import FaceAnonymizer
from app.services.face_detector import BoundingBox, FaceDetection


ROOT = Path(__file__).resolve().parents[1]
_BUNDLED_SAMPLE_IMAGE = (
    ROOT
    / "app"
    / "model"
    / "face_detection_yunet"
    / "example_outputs"
    / "largest_selfie.jpg"
)
SAMPLE_IMAGE = (
    _BUNDLED_SAMPLE_IMAGE
    if _BUNDLED_SAMPLE_IMAGE.is_file()
    else ROOT / "tests" / "output" / "largest_selfie_before.jpg"
)


def encoded_png(width: int = 80, height: int = 60) -> bytes:
    yy, xx = np.indices((height, width))
    image = np.stack(
        ((xx * 7) % 256, (yy * 9) % 256, ((xx + yy) * 5) % 256),
        axis=-1,
    ).astype(np.uint8)
    success, encoded = cv2.imencode(".png", image)
    if not success:
        raise RuntimeError("Could not construct test PNG")
    return encoded.tobytes()


def fake_detection(x: float, y: float, width: float, height: float) -> FaceDetection:
    return FaceDetection(
        BoundingBox(x, y, width, height),
        0.9,
        ((1.0, 1.0),) * 5,
    )


class FaceApiTests(unittest.TestCase):
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

    def post_image(
        self,
        client: TestClient,
        payload: bytes | None = None,
        filename: str = "input.png",
        method: str = "blur",
        strength: int = 8,
        margin: float = 0.0,
    ):
        return client.post(
            "/api/anonymize",
            files={"file": (filename, payload if payload is not None else encoded_png(), "application/octet-stream")},
            data={
                "method": method,
                "strength": str(strength),
                "margin": str(margin),
            },
        )

    def test_health_ready_when_anonymizer_initializes(self) -> None:
        client, _ = self.make_client()
        response = client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"status": "ok", "detector_ready": True},
        )

    def test_health_and_requests_are_not_ready_when_model_initialization_fails(self) -> None:
        def fail_initialization():
            raise FileNotFoundError("private model path")

        client, _ = self.make_client(factory=fail_initialization)
        health = client.get("/api/health")
        self.assertEqual(health.json()["detector_ready"], False)
        response = self.post_image(client)
        self.assertEqual(response.status_code, 503)
        self.assertNotIn("private model path", response.text)

    def test_blur_and_pixelate_return_decodable_png_with_metadata(self) -> None:
        face = fake_detection(15, 12, 30, 24)
        for method in ("blur", "pixelate"):
            with self.subTest(method=method):
                client, _ = self.make_client([face])
                response = self.post_image(client, method=method)

                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.headers["content-type"], "image/png")
                self.assertEqual(response.headers["x-anonymization-method"], method)
                self.assertEqual(response.headers["x-face-count"], "1")
                self.assertEqual(response.headers["x-processed-face-count"], "1")
                self.assertEqual(response.headers["x-anonymized"], "true")
                self.assertEqual(response.headers["x-anonymization-strength"], "8")
                self.assertEqual(response.headers["x-anonymization-margin"], "0.0")
                decoded = cv2.imdecode(
                    np.frombuffer(response.content, dtype=np.uint8),
                    cv2.IMREAD_UNCHANGED,
                )
                self.assertEqual(decoded.shape, (60, 80, 3))

    def test_multiple_faces_are_reported(self) -> None:
        client, _ = self.make_client(
            [
                fake_detection(5, 5, 15, 15),
                fake_detection(50, 35, 20, 20),
            ]
        )
        response = self.post_image(client)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-face-count"], "2")
        self.assertEqual(response.headers["x-processed-face-count"], "2")

    def test_no_faces_returns_valid_image_and_zero_metadata(self) -> None:
        client, _ = self.make_client([])
        response = self.post_image(client)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-face-count"], "0")
        self.assertEqual(response.headers["x-processed-face-count"], "0")
        self.assertEqual(response.headers["x-anonymized"], "false")
        returned = cv2.imdecode(
            np.frombuffer(response.content, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        np.testing.assert_array_equal(returned, cv2.imdecode(
            np.frombuffer(encoded_png(), dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        ))

    def test_repeated_requests_return_independent_valid_results(self) -> None:
        client, detector = self.make_client([fake_detection(15, 12, 30, 24)])

        responses = [self.post_image(client) for _ in range(2)]

        for response in responses:
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.headers["x-face-count"], "1")
            self.assertEqual(response.headers["x-processed-face-count"], "1")
            decoded = cv2.imdecode(
                np.frombuffer(response.content, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
            self.assertEqual(decoded.shape, (60, 80, 3))
        self.assertEqual(detector.detect.call_count, 2)
        self.assertIsNot(responses[0].content, responses[1].content)

    def test_detected_but_unprocessable_face_is_not_reported_anonymized(self) -> None:
        client, _ = self.make_client([fake_detection(200, 200, 10, 10)])

        response = self.post_image(client)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["x-face-count"], "1")
        self.assertEqual(response.headers["x-processed-face-count"], "0")
        self.assertEqual(response.headers["x-anonymized"], "false")

    def test_missing_upload_is_rejected(self) -> None:
        client, _ = self.make_client()
        response = client.post("/api/anonymize", data={"method": "blur"})
        self.assertEqual(response.status_code, 422)

    def test_empty_and_corrupt_uploads_are_rejected(self) -> None:
        client, _ = self.make_client()
        empty = self.post_image(client, payload=b"")
        corrupt = self.post_image(client, payload=b"\xff\xd8\xffnot-an-image")
        self.assertEqual(empty.status_code, 400)
        self.assertEqual(corrupt.status_code, 400)

    def test_unsupported_format_is_rejected(self) -> None:
        stream = BytesIO()
        Image.new("RGB", (8, 8), "red").save(stream, format="GIF")
        client, _ = self.make_client()
        response = self.post_image(
            client,
            payload=stream.getvalue(),
            filename="image.png",
        )
        self.assertEqual(response.status_code, 415)

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
                response = self.post_image(client, **options)
                self.assertEqual(response.status_code, 422)

    def test_upload_size_limit_is_enforced(self) -> None:
        client, _ = self.make_client(
            settings=ApiSettings(max_upload_bytes=128),
        )
        response = self.post_image(client, payload=encoded_png(100, 100))
        self.assertEqual(response.status_code, 413)

    def test_request_body_limit_runs_before_multipart_parsing(self) -> None:
        client, _ = self.make_client(
            settings=ApiSettings(max_upload_bytes=128),
        )
        payload = b"x" * (128 + 64 * 1024 + 1)
        response = self.post_image(client, payload=payload)
        self.assertEqual(response.status_code, 413)
        self.assertEqual(response.json()["detail"], "Request body is too large")

    def test_dimension_limit_is_checked_before_opencv_decode(self) -> None:
        client, _ = self.make_client(
            settings=ApiSettings(max_image_width=20),
        )
        with patch("app.main.cv2.imdecode") as decode:
            response = self.post_image(client, payload=encoded_png(21, 10))
        self.assertEqual(response.status_code, 413)
        decode.assert_not_called()

    def test_actual_model_anonymizes_sample_and_returns_dimensions(self) -> None:
        if not SAMPLE_IMAGE.is_file():
            self.skipTest("Bundled YuNet sample is unavailable")
        client, _ = self.make_client(factory=FaceAnonymizer)
        with SAMPLE_IMAGE.open("rb") as sample:
            response = client.post(
                "/api/anonymize",
                files={"file": ("selfie.jpg", sample, "image/jpeg")},
                data={"method": "pixelate", "strength": "15", "margin": "0"},
            )

        self.assertEqual(response.status_code, 200, response.text)
        face_count = int(response.headers["x-face-count"])
        self.assertGreater(face_count, 0)
        self.assertEqual(
            response.headers["x-processed-face-count"],
            str(face_count),
        )
        processed = cv2.imdecode(
            np.frombuffer(response.content, dtype=np.uint8),
            cv2.IMREAD_UNCHANGED,
        )
        self.assertEqual(processed.shape, (1150, 2048, 3))

    def test_processing_error_is_safe_and_does_not_leak_details(self) -> None:
        client, detector = self.make_client([fake_detection(1, 1, 20, 20)])
        detector.detect.side_effect = RuntimeError("internal details")
        response = self.post_image(client)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(response.json()["detail"], "Image processing failed")
        self.assertNotIn("internal details", response.text)

    def test_encoding_error_returns_safe_server_error(self) -> None:
        client, _ = self.make_client()
        payload = encoded_png()
        with patch("app.main.cv2.imencode", side_effect=cv2.error("private detail")):
            response = self.post_image(client, payload=payload)
        self.assertEqual(response.status_code, 500)
        self.assertEqual(
            response.json()["detail"],
            "Processed image could not be encoded",
        )
        self.assertNotIn("private detail", response.text)

    def test_cors_exposes_response_metadata_to_vite_origin(self) -> None:
        client, _ = self.make_client()
        preflight = client.options(
            "/api/anonymize",
            headers={
                "Origin": "http://localhost:5173",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type",
            },
        )
        self.assertEqual(preflight.status_code, 200)
        self.assertIn(
            "http://localhost:5173",
            preflight.headers["access-control-allow-origin"],
        )

        response = client.get(
            "/api/health",
            headers={"Origin": "http://localhost:5173"},
        )
        self.assertIn(
            "X-Face-Count",
            response.headers["access-control-expose-headers"],
        )


class ApiModelInitializationTests(unittest.TestCase):
    def test_failed_model_initialization_is_not_reported_ready(self) -> None:
        model_path = ROOT / "app" / "model" / "missing-model.onnx"

        def initialize():
            raise FileNotFoundError(f"YuNet model file not found: {model_path}")

        with TestClient(create_app(anonymizer_factory=initialize)) as client:
            response = client.get("/api/health")

        self.assertEqual(
            response.json(),
            {"status": "ok", "detector_ready": False},
        )
        self.assertNotIn(str(model_path), response.text)


if __name__ == "__main__":
    unittest.main()
