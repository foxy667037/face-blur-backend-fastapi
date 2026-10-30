# Face Anonymization Backend

The backend accepts an image upload, detects faces with the bundled YuNet
model, and returns a PNG processed with Gaussian blur or pixelation. Uploaded
images are processed in memory and are not saved by the API.

Blur and pixelation modify face regions but are not guaranteed to make people
unidentifiable.

## Local setup

From the backend project root in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m uvicorn app.main:app --reload --ws-max-size 1000000 --ws-max-queue 1
```

The existing virtual environment can also be used without activation:

```powershell
& '.\.venv\Scripts\python.exe' -m uvicorn app.main:app --reload --ws-max-size 1000000 --ws-max-queue 1
```

OpenAPI documentation is available at <http://127.0.0.1:8000/docs>; the
health endpoint is <http://127.0.0.1:8000/api/health>.

Run all tests from the project root:

```powershell
& '.\.venv\Scripts\python.exe' -m unittest discover -s tests -v
```

The Uvicorn WebSocket transport is configured with a 1,000,000-byte message
limit and a one-message receive queue. The endpoint also validates the JPEG
payload size, dimensions, and decoded frame before processing.

To measure five sequential real-YuNet round trips on the included sample
frame while the API is running, use:

```powershell
& '.\.venv\Scripts\python.exe' scripts\benchmark_video_stream.py
```

Pass `--url ws://127.0.0.1:PORT/api/video/ws` to measure a non-default API
port. The reported throughput uses a local sample image, not a camera stream.

## API

### `GET /api/health`

Returns `{"status":"ok","detector_ready":true}` only after the anonymizer and
YuNet model have initialized successfully. If initialization failed, the API
remains available for health checks and reports `detector_ready:false`;
anonymization requests return HTTP 503.

### `POST /api/anonymize`

Send `multipart/form-data` with:

| Field | Required | Default | Validation |
|---|---:|---:|---|
| `file` | yes | — | Supported JPEG, PNG, BMP, WebP, or TIFF image |
| `method` | no | `blur` | `blur` or `pixelate` |
| `strength` | no | `15` | Integer 1–50; pixelation requires 2–50 |
| `margin` | no | `0` | Fraction from 0 to 1, relative to each detected box |

On success, the response body is a PNG. Metadata is returned in
`X-Face-Count`, `X-Processed-Face-Count`, `X-Anonymization-Method`,
`X-Anonymization-Strength`, `X-Anonymization-Margin`, and `X-Anonymized`
headers. These headers are exposed to the configured browser origins by CORS.
If no faces are detected, the unchanged image is returned with zero counts
and `X-Anonymized: false`.

The default upload limit is 10 MiB. Images are limited to 8192 pixels per
dimension and 25 million total pixels. CORS defaults to Vite development
origins `http://localhost:5173` and `http://127.0.0.1:5173`.

Override limits and origins with:

- `FACE_ANONYMIZATION_MAX_UPLOAD_BYTES`
- `FACE_ANONYMIZATION_MAX_IMAGE_WIDTH`
- `FACE_ANONYMIZATION_MAX_IMAGE_HEIGHT`
- `FACE_ANONYMIZATION_MAX_IMAGE_PIXELS`
- `FACE_ANONYMIZATION_CORS_ORIGINS` (comma-separated origins)

The API does not accept model paths from clients and does not write uploads or
processed output to disk.

## Live video processing

The existing `VideoFrameProcessor`, `FaceAnonymizer`, and loaded YuNet model
are reused by `WS /api/video/ws`. The browser opens the connection only after
the user enables the camera. It downsizes frames to at most 640 x 360 and
targets one frame every 200 ms. Each request is sent as a JSON text message
followed by one JPEG binary message:

```json
{"type":"frame","sequence":0,"method":"blur","strength":15}
```

The server responds with JSON metadata followed by the processed JPEG binary
frame. Metadata includes sequence, dimensions, detected and processed face
counts, method, strength, `anonymized`, and backend processing time. Sequence
numbers must increase. Only one request is in flight in the bundled React
client; it does not build a backlog or send independent HTTP requests.

Frames are JPEG-only, at most 1,000,000 encoded bytes, 1280 x 720 pixels per
dimension, and 921,600 total pixels. JSON metadata is limited to 4 KiB. The
service processes frame data in memory; it does not write raw frames to disk
or log image contents. Invalid input and processing errors return a safe
error message and close the connection. The client clears the processed
preview and stops the camera instead of showing a raw-frame fallback.

The video page displays only frames returned by the API. The hidden camera
video is used only as a source for frame capture. Disabling the camera or
leaving video mode closes the WebSocket and stops all media tracks. Camera
permission is requested only after the user selects **Enable camera**.

For live tests, install `requirements.txt` so Uvicorn has its WebSocket
protocol dependency (`websockets`). The backend tests include mocked
WebSocket cases and a real-YuNet test that transmits a resized sample image.
An integrated browser test used a generated canvas stream (not a physical
webcam) with the real local API: 640 x 360 frames, 28.1 ms backend processing
time and 4.7 received frames per second in that run; the generated scene had
zero face detections. A real webcam could not be exercised because the
browser denied camera permission. These figures are a single synthetic-source
smoke measurement, not a real-camera performance guarantee.

YuNet can miss faces. A zero-face frame is reported accurately but does not
prove that no face is present. Blur and pixelation do not guarantee anonymity.
Recording/download is not included in this milestone; the next task must
record only the processed canvas stream.
