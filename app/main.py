"""FastAPI application for in-memory face anonymization."""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
import io
import json
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Callable

import cv2
from fastapi import (
    FastAPI,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
import numpy as np
from PIL import Image, UnidentifiedImageError
from starlette.concurrency import run_in_threadpool
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from app.services.anonymizer import FaceAnonymizer
from app.services.video_anonymizer import (
    VideoAnonymizationResult,
    VideoDecodeError,
    VideoTooLongError,
    VideoAnonymizer,
)
from app.services.video_processor import VideoFrameProcessor


logger = logging.getLogger(__name__)
SUPPORTED_IMAGE_FORMATS = {"JPEG", "PNG", "BMP", "WEBP", "TIFF"}
PNG_COMPRESSION_LEVEL = 3
MULTIPART_OVERHEAD_LIMIT_BYTES = 64 * 1024
MAX_VIDEO_FRAME_BYTES = 1_000_000
MAX_VIDEO_FRAME_WIDTH = 1280
MAX_VIDEO_FRAME_HEIGHT = 720
MAX_VIDEO_FRAME_PIXELS = 1280 * 720
MAX_VIDEO_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_VIDEO_FRAMES = 1500
MAX_VIDEO_DIMENSION = 1280


class _RequestBodyTooLarge(Exception):
    pass


class UploadBodyLimitMiddleware:
    """Bound multipart bodies before Starlette parses or spools an upload."""

    def __init__(
        self,
        app: ASGIApp,
        image_limit_bytes: int,
        video_limit_bytes: int,
    ) -> None:
        self.app = app
        self.image_limit_bytes = image_limit_bytes
        self.video_limit_bytes = video_limit_bytes

    async def __call__(
        self,
        scope: Scope,
        receive: Receive,
        send: Send,
    ) -> None:
        if (
            scope["type"] != "http"
            or scope["method"] != "POST"
            or scope["path"] not in ("/api/anonymize", "/api/anonymize-video")
        ):
            await self.app(scope, receive, send)
            return

        headers = dict(scope["headers"])
        content_length = headers.get(b"content-length")
        max_bytes = (
            self.video_limit_bytes
            if scope["path"] == "/api/anonymize-video"
            else self.image_limit_bytes
        )
        if content_length is not None:
            try:
                declared_length = int(content_length)
            except ValueError:
                declared_length = 0
            if declared_length > max_bytes:
                await self._send_too_large(send)
                return

        received_bytes = 0

        async def limited_receive() -> Message:
            nonlocal received_bytes
            message = await receive()
            if message["type"] == "http.request":
                received_bytes += len(message.get("body", b""))
                if received_bytes > max_bytes:
                    raise _RequestBodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _RequestBodyTooLarge:
            await self._send_too_large(send)

    @staticmethod
    async def _send_too_large(send: Send) -> None:
        body = b'{"detail":"Request body is too large"}'
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(body)).encode("ascii")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": body,
                "more_body": False,
            }
        )


def _positive_int_from_environment(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be a positive integer") from exc
    if parsed < 1:
        raise RuntimeError(f"{name} must be a positive integer")
    return parsed


@dataclass(frozen=True)
class ApiSettings:
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_width: int = 8192
    max_image_height: int = 8192
    max_image_pixels: int = 25_000_000
    max_video_upload_bytes: int = 100 * 1024 * 1024
    max_video_frames: int = 1500
    max_video_dimension: int = 1280
    cors_origins: tuple[str, ...] = (
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    )

    @classmethod
    def from_environment(cls) -> ApiSettings:
        origins = os.getenv(
            "FACE_ANONYMIZATION_CORS_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        )
        return cls(
            max_upload_bytes=_positive_int_from_environment(
                "FACE_ANONYMIZATION_MAX_UPLOAD_BYTES",
                10 * 1024 * 1024,
            ),
            max_image_width=_positive_int_from_environment(
                "FACE_ANONYMIZATION_MAX_IMAGE_WIDTH",
                8192,
            ),
            max_image_height=_positive_int_from_environment(
                "FACE_ANONYMIZATION_MAX_IMAGE_HEIGHT",
                8192,
            ),
            max_image_pixels=_positive_int_from_environment(
                "FACE_ANONYMIZATION_MAX_IMAGE_PIXELS",
                25_000_000,
            ),
            max_video_upload_bytes=_positive_int_from_environment(
                "FACE_ANONYMIZATION_MAX_VIDEO_UPLOAD_BYTES",
                100 * 1024 * 1024,
            ),
            max_video_frames=_positive_int_from_environment(
                "FACE_ANONYMIZATION_MAX_VIDEO_FRAMES",
                1500,
            ),
            max_video_dimension=_positive_int_from_environment(
                "FACE_ANONYMIZATION_MAX_VIDEO_DIMENSION",
                1280,
            ),
            cors_origins=tuple(origin.strip() for origin in origins.split(",") if origin.strip()),
        )


def _validate_settings(settings: ApiSettings) -> None:
    limits = (
        settings.max_upload_bytes,
        settings.max_image_width,
        settings.max_image_height,
        settings.max_image_pixels,
        settings.max_video_upload_bytes,
        settings.max_video_frames,
        settings.max_video_dimension,
    )
    if any(isinstance(value, bool) or value < 1 for value in limits):
        raise ValueError("API upload and image limits must be positive integers")


def create_app(
    anonymizer_factory: Callable[[], FaceAnonymizer] = FaceAnonymizer,
    settings: ApiSettings | None = None,
) -> FastAPI:
    """Create an API application, allowing isolated readiness tests."""
    api_settings = settings if settings is not None else ApiSettings.from_environment()
    _validate_settings(api_settings)

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        application.state.anonymizer = None
        application.state.video_processor = None
        application.state.video_anonymizer = None
        application.state.detector_error = None
        try:
            application.state.anonymizer = anonymizer_factory()
            application.state.video_processor = VideoFrameProcessor(
                application.state.anonymizer
            )
            application.state.video_anonymizer = VideoAnonymizer(
                application.state.video_processor
            )
        except (FileNotFoundError, RuntimeError) as exc:
            application.state.detector_error = exc
            logger.error("YuNet detector initialization failed: %s", exc)
        yield

    application = FastAPI(
        title="Face Anonymization API",
        description="Detect and anonymize faces in uploaded images.",
        version="1.0.0",
        lifespan=lifespan,
    )
    application.add_middleware(
        UploadBodyLimitMiddleware,
        image_limit_bytes=api_settings.max_upload_bytes + MULTIPART_OVERHEAD_LIMIT_BYTES,
        video_limit_bytes=api_settings.max_video_upload_bytes + MULTIPART_OVERHEAD_LIMIT_BYTES,
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=list(api_settings.cors_origins),
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
        expose_headers=[
            "X-Face-Count",
            "X-Processed-Face-Count",
            "X-Anonymization-Method",
            "X-Anonymization-Strength",
            "X-Anonymization-Margin",
            "X-Anonymized",
            "X-Video-Frame-Count",
            "X-Video-Processed-Frame-Count",
            "X-Video-Face-Count",
        ],
    )
    
    @application.get("/")
    async def getStatus() -> dict[str, bool | str]:
        return {"status": "ok", "server": "server is running"}

    @application.get("/api/health")
    async def health() -> dict[str, bool | str]:
        ready = application.state.anonymizer is not None
        return {"status": "ok", "detector_ready": ready}

    @application.post("/api/anonymize")
    async def anonymize(
        file: UploadFile = File(...),
        method: str = Form("blur"),
        strength: int = Form(15, ge=1, le=50),
        margin: float = Form(0.0, ge=0.0, le=1.0),
    ) -> Response:
        if method not in ("blur", "pixelate"):
            raise HTTPException(
                status_code=422,
                detail="method must be 'blur' or 'pixelate'",
            )
        if method == "pixelate" and strength < 2:
            raise HTTPException(
                status_code=422,
                detail="strength for pixelate must be between 2 and 50",
            )

        anonymizer = application.state.anonymizer
        if anonymizer is None:
            raise HTTPException(
                status_code=503,
                detail="Face detector is not ready",
            )

        try:
            payload = await file.read(api_settings.max_upload_bytes + 1)
        finally:
            await file.close()

        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(payload) > api_settings.max_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")

        try:
            with Image.open(io.BytesIO(payload)) as image_header:
                image_format = image_header.format
                width, height = image_header.size
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            raise HTTPException(
                status_code=400,
                detail="Uploaded file is not a valid image",
            ) from exc

        if image_format not in SUPPORTED_IMAGE_FORMATS:
            raise HTTPException(
                status_code=415,
                detail="Image format is not supported",
            )
        if (
            width > api_settings.max_image_width
            or height > api_settings.max_image_height
            or width * height > api_settings.max_image_pixels
        ):
            raise HTTPException(
                status_code=413,
                detail="Image dimensions exceed the configured limit",
            )

        try:
            image = await run_in_threadpool(
                cv2.imdecode,
                np.frombuffer(payload, dtype=np.uint8),
                cv2.IMREAD_UNCHANGED,
            )
        except cv2.error as exc:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image could not be decoded",
            ) from exc
        if image is None:
            raise HTTPException(
                status_code=400,
                detail="Uploaded image could not be decoded",
            )
        if image.dtype != np.uint8:
            raise HTTPException(
                status_code=415,
                detail="Image bit depth is not supported",
            )
        if image.ndim == 3 and image.shape[2] not in (3, 4):
            raise HTTPException(
                status_code=415,
                detail="Image channel layout is not supported",
            )
        if image.ndim not in (2, 3):
            raise HTTPException(
                status_code=415,
                detail="Image channel layout is not supported",
            )

        try:
            processed, metadata = await run_in_threadpool(
                anonymizer.anonymize_image,
                image,
                method=method,
                strength=strength,
                margin=margin,
            )
        except (ValueError, RuntimeError, cv2.error) as exc:
            logger.error("Image anonymization failed: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Image processing failed",
            ) from exc

        try:
            success, encoded = await run_in_threadpool(
                cv2.imencode,
                ".png",
                processed,
                [cv2.IMWRITE_PNG_COMPRESSION, PNG_COMPRESSION_LEVEL],
            )
        except cv2.error as exc:
            logger.error("Processed image encoding failed: %s", exc)
            raise HTTPException(
                status_code=500,
                detail="Processed image could not be encoded",
            ) from exc
        if not success:
            raise HTTPException(
                status_code=500,
                detail="Processed image could not be encoded",
            )

        return Response(
            content=encoded.tobytes(),
            media_type="image/png",
            headers={
                "Content-Disposition": 'inline; filename="anonymized.png"',
                "X-Face-Count": str(metadata.face_count),
                "X-Processed-Face-Count": str(metadata.processed_face_count),
                "X-Anonymization-Method": metadata.method,
                "X-Anonymization-Strength": str(metadata.strength),
                "X-Anonymization-Margin": str(metadata.margin),
                "X-Anonymized": str(metadata.anonymized).lower(),
            },
        )

    @application.post("/api/anonymize-video")
    async def anonymize_video(
        file: UploadFile = File(...),
        method: str = Form("blur"),
        strength: int = Form(15, ge=1, le=50),
        margin: float = Form(0.0, ge=0.0, le=1.0),
    ) -> Response:
        if method not in ("blur", "pixelate"):
            raise HTTPException(
                status_code=422,
                detail="method must be 'blur' or 'pixelate'",
            )
        if method == "pixelate" and strength < 2:
            raise HTTPException(
                status_code=422,
                detail="strength for pixelate must be between 2 and 50",
            )

        anonymizer = application.state.video_anonymizer
        if anonymizer is None:
            raise HTTPException(
                status_code=503,
                detail="Face detector is not ready",
            )

        try:
            payload = await file.read(api_settings.max_video_upload_bytes + 1)
        finally:
            await file.close()

        if not payload:
            raise HTTPException(status_code=400, detail="Uploaded file is empty")
        if len(payload) > api_settings.max_video_upload_bytes:
            raise HTTPException(status_code=413, detail="Uploaded file is too large")

        upload_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.filename or "upload").suffix
            ) as upload_file:
                upload_file.write(payload)
                upload_path = Path(upload_file.name)

            try:
                result = await run_in_threadpool(
                    anonymizer.anonymize_video,
                    upload_path,
                    Path(tempfile.gettempdir()),
                    method=method,
                    strength=strength,
                    margin=margin,
                    max_frames=api_settings.max_video_frames,
                    max_dimension=api_settings.max_video_dimension,
                )
            except VideoTooLongError:
                raise HTTPException(
                    status_code=413,
                    detail="Video is longer than the configured frame limit",
                )
            except VideoDecodeError as exc:
                logger.error("Video upload could not be decoded: %s", exc)
                raise HTTPException(status_code=400, detail="Uploaded video could not be decoded")
            except (ValueError, RuntimeError, cv2.error) as exc:
                logger.error("Video anonymization failed: %s", exc)
                raise HTTPException(
                    status_code=500,
                    detail="Video processing failed",
                ) from exc
        finally:
            if upload_path is not None:
                upload_path.unlink(missing_ok=True)

        return Response(
            content=result.content,
            media_type=result.media_type,
            headers={
                "Content-Disposition": f'inline; filename="{result.filename}"',
                "X-Video-Frame-Count": str(result.frame_count),
                "X-Video-Processed-Frame-Count": str(result.processed_frame_count),
                "X-Video-Face-Count": str(result.face_count),
                "X-Anonymization-Method": result.method,
                "X-Anonymization-Strength": str(result.strength),
                "X-Anonymization-Margin": str(result.margin),
                "X-Anonymized": str(result.anonymized).lower(),
            },
        )

    @application.websocket("/api/video/ws")
    async def process_video(websocket: WebSocket) -> None:
        origin = websocket.headers.get("origin")
        if origin and origin not in api_settings.cors_origins:
            await websocket.close(code=1008)
            return

        processor = application.state.video_processor
        if processor is None:
            await websocket.close(code=1013)
            return

        await websocket.accept()
        last_sequence = -1
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                text = message.get("text")
                if text is None:
                    await websocket.send_json(
                        {"type": "error", "error": "Expected frame metadata"}
                    )
                    await websocket.close(code=1003)
                    return
                if len(text) > 4096:
                    await websocket.send_json(
                        {"type": "error", "error": "Frame metadata is too large"}
                    )
                    await websocket.close(code=1009)
                    return
                try:
                    frame_request = json.loads(text)
                except ValueError:
                    await websocket.send_json(
                        {"type": "error", "error": "Frame metadata is invalid"}
                    )
                    await websocket.close(code=1003)
                    return

                if not isinstance(frame_request, dict):
                    await websocket.send_json(
                        {"type": "error", "error": "Frame metadata is invalid"}
                    )
                    await websocket.close(code=1003)
                    return
                sequence = frame_request.get("sequence")
                method = frame_request.get("method")
                strength = frame_request.get("strength")
                if (
                    frame_request.get("type") != "frame"
                    or isinstance(sequence, bool)
                    or not isinstance(sequence, int)
                    or sequence <= last_sequence
                    or method not in ("blur", "pixelate")
                    or isinstance(strength, bool)
                    or not isinstance(strength, int)
                    or not (2 if method == "pixelate" else 1) <= strength <= 50
                ):
                    await websocket.send_json(
                        {"type": "error", "error": "Frame options are invalid"}
                    )
                    await websocket.close(code=1008)
                    return

                frame_message = await websocket.receive()
                if frame_message["type"] == "websocket.disconnect":
                    return
                payload = frame_message.get("bytes")
                if payload is None:
                    await websocket.send_json(
                        {"type": "error", "error": "Expected a JPEG frame"}
                    )
                    await websocket.close(code=1003)
                    return
                if not payload or len(payload) > MAX_VIDEO_FRAME_BYTES:
                    await websocket.send_json(
                        {"type": "error", "error": "Frame size is invalid"}
                    )
                    await websocket.close(code=1009)
                    return

                try:
                    with Image.open(io.BytesIO(payload)) as image_header:
                        width, height = image_header.size
                        image_format = image_header.format
                except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
                    await websocket.send_json(
                        {"type": "error", "error": "Frame is not a valid JPEG image"}
                    )
                    await websocket.close(code=1003)
                    return

                if image_format != "JPEG":
                    await websocket.send_json(
                        {"type": "error", "error": "Frame format is not supported"}
                    )
                    await websocket.close(code=1003)
                    return
                if (
                    width < 1
                    or height < 1
                    or width > MAX_VIDEO_FRAME_WIDTH
                    or height > MAX_VIDEO_FRAME_HEIGHT
                    or width * height > MAX_VIDEO_FRAME_PIXELS
                ):
                    await websocket.send_json(
                        {"type": "error", "error": "Frame dimensions are not supported"}
                    )
                    await websocket.close(code=1009)
                    return

                try:
                    frame = await run_in_threadpool(
                        cv2.imdecode,
                        np.frombuffer(payload, dtype=np.uint8),
                        cv2.IMREAD_COLOR,
                    )
                    if frame is None or frame.shape[:2] != (height, width):
                        raise ValueError("Frame decoding failed")
                    started_at = time.perf_counter()
                    result = await run_in_threadpool(
                        processor.process_frame,
                        frame,
                        method=method,
                        strength=strength,
                    )
                    processing_ms = round(
                        (time.perf_counter() - started_at) * 1000,
                        1,
                    )
                    success, encoded = await run_in_threadpool(
                        cv2.imencode,
                        ".jpg",
                        result.frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 78],
                    )
                    if not success:
                        raise RuntimeError("Frame encoding failed")
                except (cv2.error, RuntimeError, ValueError) as exc:
                    logger.error("Video frame processing failed: %s", exc)
                    await websocket.send_json(
                        {"type": "error", "error": "Frame processing failed"}
                    )
                    await websocket.close(code=1011)
                    return

                await websocket.send_json(
                    {
                        "type": "frame_result",
                        "sequence": sequence,
                        "width": width,
                        "height": height,
                        "face_count": result.metadata.face_count,
                        "processed_face_count": result.metadata.processed_face_count,
                        "anonymized": result.metadata.anonymized,
                        "method": result.metadata.method,
                        "strength": result.metadata.strength,
                        "processing_ms": processing_ms,
                    }
                )
                await websocket.send_bytes(encoded.tobytes())
                last_sequence = sequence
        except WebSocketDisconnect:
            return

    return application


app = create_app()