from io import BytesIO
from pathlib import Path
import threading
import time
import unittest
from unittest.mock import Mock

import cv2
from fastapi.testclient import TestClient
import numpy as np
from PIL import Image
from starlette.websockets import WebSocketDisconnect

from app.main import ApiSettings, MAX_VIDEO_FRAME_BYTES, create_app
from app.services.anonymizer import FaceAnonymizer
from app.services.face_detector import BoundingBox, FaceDetection


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_IMAGE = ROOT / "tests" / "output" / "largest_selfie_before.jpg"


def detection(x: float, y: float, width: float, height: float) -> FaceDetection:
    return FaceDetection(
        BoundingBox(x, y, width, height),
        0.9,
        ((1.0, 1.0),) * 5,
    )


def encoded_jpeg(width: int = 80, height: int = 60) -> bytes:
    yy, xx = np.indices((height, width))
    frame = np.stack(
        ((xx * 7) % 256, (yy * 9) % 256, ((xx + yy) * 5) % 256),
        axis=-1,
    ).astype(np.uint8)
    success, encoded = cv2.imencode(".jpg", frame)
    if not success:
        raise RuntimeError("Could not construct test JPEG")
    return encoded.tobytes()


class VideoWebSocketTests(unittest.TestCase):
    def make_client(self, detector: Mock | None = None) -> tuple[TestClient, Mock]:
        if detector is None:
            detector = Mock()
            detector.detect.return_value = [detection(10, 10, 30, 30)]
        app = create_app(
            anonymizer_factory=lambda: FaceAnonymizer(detector),
            settings=ApiSettings(),
        )
        client = TestClient(app)
        client.__enter__()
        self.addCleanup(client.__exit__, None, None, None)
        return client, detector

    @staticmethod
    def send_frame(
        websocket,
        sequence: int,
        method: str = "blur",
        strength: int = 15,
        payload: bytes | None = None,
    ) -> tuple[dict, bytes]:
        websocket.send_json(
            {
                "type": "frame",
                "sequence": sequence,
                "method": method,
                "strength": strength,
            }
        )
        websocket.send_bytes(payload if payload is not None else encoded_jpeg())
        return websocket.receive_json(), websocket.receive_bytes()

    def test_connection_processes_sequential_frames_with_metadata_and_jpeg(self) -> None:
        client, detector = self.make_client()
        with client.websocket_connect("/api/video/ws") as websocket:
            for sequence, method in enumerate(("blur", "pixelate")):
                metadata, payload = self.send_frame(
                    websocket,
                    sequence,
                    method=method,
                    strength=15,
                )
                self.assertEqual(metadata["type"], "frame_result")
                self.assertEqual(metadata["sequence"], sequence)
                self.assertEqual(metadata["method"], method)
                self.assertEqual(metadata["face_count"], 1)
                self.assertEqual(metadata["processed_face_count"], 1)
                self.assertTrue(metadata["anonymized"])
                self.assertGreaterEqual(metadata["processing_ms"], 0)
                decoded = cv2.imdecode(
                    np.frombuffer(payload, dtype=np.uint8),
                    cv2.IMREAD_COLOR,
                )
                self.assertEqual(decoded.shape, (60, 80, 3))
        self.assertEqual(detector.detect.call_count, 2)

    def test_zero_faces_are_reported_without_claiming_anonymization(self) -> None:
        detector = Mock()
        detector.detect.return_value = []
        client, _ = self.make_client(detector)

        with client.websocket_connect("/api/video/ws") as websocket:
            metadata, _ = self.send_frame(websocket, 0)

        self.assertEqual(metadata["face_count"], 0)
        self.assertEqual(metadata["processed_face_count"], 0)
        self.assertFalse(metadata["anonymized"])

    def test_client_disconnect_is_handled_cleanly(self) -> None:
        client, detector = self.make_client()
        with client.websocket_connect("/api/video/ws") as websocket:
            websocket.close()

        detector.detect.assert_not_called()

    def test_disconnect_while_waiting_for_frame_is_handled_cleanly(self) -> None:
        client, detector = self.make_client()
        with client.websocket_connect("/api/video/ws") as websocket:
            websocket.send_json(
                {
                    "type": "frame",
                    "sequence": 0,
                    "method": "blur",
                    "strength": 15,
                }
            )
            websocket.close()

        detector.detect.assert_not_called()

    def test_pipelined_frames_are_processed_one_at_a_time(self) -> None:
        detector = Mock()
        active = 0
        maximum_active = 0
        lock = threading.Lock()

        def detect(_frame):
            nonlocal active, maximum_active
            with lock:
                active += 1
                maximum_active = max(maximum_active, active)
            time.sleep(0.02)
            with lock:
                active -= 1
            return [detection(10, 10, 30, 30)]

        detector.detect.side_effect = detect
        client, _ = self.make_client(detector)

        with client.websocket_connect("/api/video/ws") as websocket:
            for sequence in range(3):
                websocket.send_json(
                    {
                        "type": "frame",
                        "sequence": sequence,
                        "method": "blur",
                        "strength": 15,
                    }
                )
                websocket.send_bytes(encoded_jpeg())

            sequences = []
            for _ in range(3):
                sequences.append(websocket.receive_json()["sequence"])
                websocket.receive_bytes()

        self.assertEqual(sequences, [0, 1, 2])
        self.assertEqual(detector.detect.call_count, 3)
        self.assertEqual(maximum_active, 1)

    def test_malformed_metadata_is_rejected_and_connection_closed(self) -> None:
        client, _ = self.make_client()
        with client.websocket_connect("/api/video/ws") as websocket:
            websocket.send_text("{")
            response = websocket.receive_json()

        self.assertEqual(response["type"], "error")
        self.assertEqual(response["error"], "Frame metadata is invalid")

    def test_oversized_metadata_is_rejected(self) -> None:
        client, _ = self.make_client()
        with client.websocket_connect("/api/video/ws") as websocket:
            websocket.send_text(" " * 4097)
            response = websocket.receive_json()

        self.assertEqual(response["type"], "error")
        self.assertEqual(response["error"], "Frame metadata is too large")

    def test_invalid_options_are_rejected(self) -> None:
        client, _ = self.make_client()
        with client.websocket_connect("/api/video/ws") as websocket:
            websocket.send_json(
                {
                    "type": "frame",
                    "sequence": 0,
                    "method": "pixelate",
                    "strength": 1,
                }
            )
            response = websocket.receive_json()

        self.assertEqual(response["type"], "error")
        self.assertEqual(response["error"], "Frame options are invalid")

    def test_empty_and_oversized_frames_are_rejected(self) -> None:
        for payload in (b"", b"x" * (MAX_VIDEO_FRAME_BYTES + 1)):
            with self.subTest(size=len(payload)):
                client, _ = self.make_client()
                with client.websocket_connect("/api/video/ws") as websocket:
                    websocket.send_json(
                        {
                            "type": "frame",
                            "sequence": 0,
                            "method": "blur",
                            "strength": 15,
                        }
                    )
                    websocket.send_bytes(payload)
                    response = websocket.receive_json()
                self.assertEqual(response["type"], "error")
                self.assertEqual(response["error"], "Frame size is invalid")

    def test_unsupported_format_and_dimensions_are_rejected(self) -> None:
        png_frame = np.zeros((10, 10, 3), dtype=np.uint8)
        _, png = cv2.imencode(".png", png_frame)
        oversized = Image.new("RGB", (1281, 10), "white")
        oversized_bytes = BytesIO()
        oversized.save(oversized_bytes, format="JPEG")

        for payload, expected in (
            (b"not-an-image", "Frame is not a valid JPEG image"),
            (png.tobytes(), "Frame format is not supported"),
            (oversized_bytes.getvalue(), "Frame dimensions are not supported"),
        ):
            with self.subTest(error=expected, size=len(payload)):
                client, _ = self.make_client()
                with client.websocket_connect("/api/video/ws") as websocket:
                    websocket.send_json(
                        {
                            "type": "frame",
                            "sequence": 0,
                            "method": "blur",
                            "strength": 15,
                        }
                    )
                    websocket.send_bytes(payload)
                    response = websocket.receive_json()
                self.assertEqual(response["type"], "error")
                self.assertEqual(response["error"], expected)

    def test_processing_errors_fail_closed_without_leaking_details(self) -> None:
        detector = Mock()
        detector.detect.side_effect = RuntimeError("private detector detail")
        client, _ = self.make_client(detector)

        with client.websocket_connect("/api/video/ws") as websocket:
            websocket.send_json(
                {
                    "type": "frame",
                    "sequence": 0,
                    "method": "blur",
                    "strength": 15,
                }
            )
            websocket.send_bytes(encoded_jpeg())
            response = websocket.receive_json()

        self.assertEqual(response["error"], "Frame processing failed")
        self.assertNotIn("private detector detail", str(response))

    def test_disallowed_browser_origin_is_rejected(self) -> None:
        client, _ = self.make_client()
        with self.assertRaises(WebSocketDisconnect):
            with client.websocket_connect(
                "/api/video/ws",
                headers={"origin": "https://untrusted.example"},
            ):
                pass


@unittest.skipUnless(SAMPLE_IMAGE.is_file(), "Real YuNet sample image is unavailable")
class VideoWebSocketModelIntegrationTests(unittest.TestCase):
    def test_real_yunet_processes_a_transmitted_camera_frame(self) -> None:
        frame = cv2.imread(str(SAMPLE_IMAGE))
        self.assertIsNotNone(frame)
        frame = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        success, encoded = cv2.imencode(".jpg", frame)
        self.assertTrue(success)
        app = create_app(anonymizer_factory=FaceAnonymizer)

        with TestClient(app) as client:
            with client.websocket_connect("/api/video/ws") as websocket:
                metadata, output = VideoWebSocketTests.send_frame(
                    websocket,
                    sequence=0,
                    method="pixelate",
                    strength=15,
                    payload=encoded.tobytes(),
                )

        self.assertEqual(metadata["type"], "frame_result")
        self.assertEqual(metadata["width"], 640)
        self.assertEqual(metadata["height"], 360)
        self.assertGreater(metadata["face_count"], 0)
        self.assertEqual(
            metadata["processed_face_count"],
            metadata["face_count"],
        )
        self.assertTrue(metadata["anonymized"])
        decoded = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape, frame.shape)


if __name__ == "__main__":
    unittest.main()
