#!/usr/bin/env python3
"""
Flask web server for YOLO object detection/tracking video stream.
Streams processed video frames to a web browser.
Supports webcam, video files, and RTSP/HLS streams via ffmpeg.
"""

import time
import os
import subprocess
import signal
import atexit
from threading import Lock

import cv2
import torch
from flask import Flask, Response, render_template

from ultralytics import YOLO
from ultralytics.utils import LOGGER
from ultralytics.utils.plotting import Annotator, colors

app = Flask(__name__)

# Configuration from environment or defaults
MODEL_PATH = os.environ.get("MODEL_PATH", "yolo26n.pt")
DETECT_STREAM_URL = os.environ.get("DETECT_STREAM_URL", None)  # HLS/HTTP stream URL
RTSP_LOCAL_URL = "rtsp://localhost:8554/test"  # Local RTSP endpoint for ffmpeg output
video_source = os.environ.get("VIDEO_SOURCE", "0")  # 0 for webcam, or path/URL

# Try to parse video_source as int for webcam device index
try:
    video_source = int(video_source)
except ValueError:
    pass  # Keep as string (file path or URL)

# CUDA settings
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if USE_CUDA else "cpu"

# Detection settings
show_fps = True
show_conf = False
conf = 0.3
iou = 0.3
max_det = 20
SKIP_FRAMES = int(os.environ.get("SKIP_FRAMES", "1"))  # Process every Nth frame

tracker = "bytetrack.yaml"
track_args = {
    "persist": True,
    "verbose": False,
}

# If DETECT_STREAM_URL is set, use ffmpeg to convert to local RTSP
if DETECT_STREAM_URL:
    video_source = RTSP_LOCAL_URL
    LOGGER.info(f"Using remote stream via ffmpeg: {DETECT_STREAM_URL} -> {RTSP_LOCAL_URL}")

# Global state
frame_lock = Lock()
latest_frame = None
ffmpeg_proc = None
frame_counter = 0

# Initialize model
LOGGER.info("🚀 Initializing YOLO model...")
LOGGER.info(f"Torch CUDA Version: {torch.version.cuda}")
model = YOLO(MODEL_PATH)

if USE_CUDA:
    LOGGER.info(f"Using CUDA on {DEVICE}")
    model.to(DEVICE)
else:
    LOGGER.info("Using CPU")

classes = model.names


def start_ffmpeg():
    """Start ffmpeg to pull HLS/HTTP stream and publish to local RTSP."""
    global ffmpeg_proc
    if ffmpeg_proc is not None and ffmpeg_proc.poll() is None:
        return
    cmd = [
        "ffmpeg",
        "-re",
        "-i", DETECT_STREAM_URL,
        "-c", "copy",
        "-f", "rtsp",
        RTSP_LOCAL_URL,
    ]
    LOGGER.info(f"Starting ffmpeg: {' '.join(cmd)}")
    ffmpeg_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)


def stop_ffmpeg():
    """Stop the ffmpeg subprocess."""
    global ffmpeg_proc
    if ffmpeg_proc is None:
        return
    try:
        if ffmpeg_proc.poll() is None:
            LOGGER.info(f"Terminating ffmpeg (pid={ffmpeg_proc.pid})")
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
    except Exception:
        try:
            ffmpeg_proc.kill()
        except Exception:
            pass
    ffmpeg_proc = None


# Ensure ffmpeg is stopped on exit
atexit.register(stop_ffmpeg)


def _handle_sigterm(signum, frame):
    stop_ffmpeg()
    raise SystemExit(0)


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def get_center(x1: int, y1: int, x2: int, y2: int) -> tuple[int, int]:
    """Calculate the center point of a bounding box."""
    return (x1 + x2) // 2, (y1 + y2) // 2


def generate_frames():
    """Generator function that yields video frames with YOLO detections."""
    global latest_frame, frame_counter
    
    # Start ffmpeg if using remote stream
    if DETECT_STREAM_URL:
        start_ffmpeg()
        # Wait for ffmpeg to start and RTSP stream to be available
        time.sleep(2)
    
    # Open video capture with retries for RTSP
    cap = None
    max_retries = 15 if DETECT_STREAM_URL else 3
    
    for attempt in range(1, max_retries + 1):
        if attempt > 1:
            time.sleep(1.0)
        LOGGER.info(f"Opening video source (attempt {attempt}/{max_retries})")
        
        if isinstance(video_source, str) and video_source.startswith("rtsp://"):
            cap = cv2.VideoCapture(video_source, cv2.CAP_FFMPEG)
        else:
            cap = cv2.VideoCapture(video_source)
        
        if cap.isOpened():
            LOGGER.info("Video source opened successfully")
            break
        else:
            if cap:
                try:
                    cap.release()
                except Exception:
                    pass
            cap = None
            LOGGER.warning("Video source not ready, retrying...")
    
    if cap is None or not cap.isOpened():
        LOGGER.error(f"Failed to open video source after {max_retries} attempts")
        stop_ffmpeg()
        return
    
    fps_counter, fps_timer, fps_display = 0, time.time(), 0
    
    try:
        while True:
            success, im = cap.read()
            if not success:
                # Loop video if file, or break if webcam/stream fails
                if isinstance(video_source, str) and not video_source.startswith("rtsp://"):
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                break
            
            frame_counter += 1

            # Run YOLO tracking to keep stable IDs across frames
            # Limit detections to car and truck when possible
            allowed_names = {"car", "truck"}
            allowed_ids = [i for i, n in classes.items() if n.lower() in allowed_names]
            if allowed_ids:
                results = model.track(
                    im,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    device=DEVICE,
                    half=USE_CUDA,
                    classes=allowed_ids,
                    tracker=tracker,
                    vid_stride=SKIP_FRAMES,
                    **track_args,
                )
            else:
                LOGGER.warning("No class ids found for car/truck; running full detection")
                results = model.track(
                    im,
                    conf=conf,
                    iou=iou,
                    max_det=max_det,
                    device=DEVICE,
                    half=USE_CUDA,
                    tracker=tracker,
                    vid_stride=SKIP_FRAMES,
                    **track_args,
                )
            annotator = Annotator(im)
            detections = results[0].boxes.data if results[0].boxes is not None else []

            # Draw detections (support both detection and tracked box formats)
            for det in detections:
                track = det.tolist()
                if len(track) < 6:
                    continue

                x1, y1, x2, y2 = map(int, track[:4])
                # Detection format: [x1,y1,x2,y2,score,class]
                # Track format: [x1,y1,x2,y2,track_id,score,class]
                if len(track) == 6:
                    conf_score = float(track[4])
                    class_id = int(track[5])
                    track_id = -1
                else:
                    track_id = int(track[4])
                    conf_score = float(track[5])
                    class_id = int(track[6])

                color = colors(track_id, True)
                txt_color = annotator.get_txt_color(color)
                class_name = classes.get(class_id, str(class_id))
                # Include track id only when available
                label = f"{class_name}"
                if track_id >= 0:
                    label += f" ID {track_id}"
                if show_conf:
                    label += f" ({conf_score:.2f})"

                annotator.box_label([x1, y1, x2, y2], label=label, color=color)
            
            # FPS counter
            if show_fps:
                fps_counter += 1
                if time.time() - fps_timer >= 1.0:
                    fps_display = fps_counter
                    fps_counter = 0
                    fps_timer = time.time()

                fps_text = f"FPS: {fps_display}"
                # Use the same smaller font for the FPS overlay to keep UI consistent
                fps_font = cv2.FONT_HERSHEY_SIMPLEX
                fps_scale = 0.5
                fps_thickness = 1
                (tw, th), bl = cv2.getTextSize(fps_text, fps_font, fps_scale, fps_thickness)
                cv2.rectangle(im, (10 - 5, 25 - th - 5), (10 + tw + 5, 25 + bl), (255, 255, 255), -1)
                cv2.putText(im, fps_text, (10, 25), fps_font, fps_scale, (104, 31, 17), fps_thickness, cv2.LINE_AA)
            
            # Store frame for access by other routes
            with frame_lock:
                latest_frame = im.copy()
            
            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', im)
            if not ret:
                continue
            
            frame_bytes = buffer.tobytes()
            
            # Yield frame in multipart format
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    
    finally:
        cap.release()
        stop_ffmpeg()


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    """Video streaming route."""
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    LOGGER.info("🌐 Starting Flask server...")
    LOGGER.info("Access the stream at: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
