from ultralytics import YOLO
import cv2
import torch
from flask import Flask, Response, render_template_string
import threading
import time
import os
import subprocess
import signal
import atexit

# Settings
MODEL_PATH = "yolo26n.pt"
DETECT_STREAM_URL = os.environ.get("DETECT_STREAM_URL", "https://publicstreamer2.cotrip.org/rtplive/225S00995CAM1NWC/chunklist_w423332375.m3u8")
SOURCE = "rtsp://localhost:8554/test"
SKIP_FRAMES = 1         # process every Nth frame
USE_CUDA = torch.cuda.is_available()
DEVICE = "cuda:0" if USE_CUDA else "cpu"

model = YOLO(MODEL_PATH)

print(f"Torch Version CUDA: {torch.version.cuda}")
# move model to device and use half precision if on GPU
if USE_CUDA:
    print("Using CUDA")
    print(DEVICE)
    model.to(DEVICE)
    # let Ultralytics manage half precision via the `half` flag
else:
    print("Using CPU")

cap = None
ffmpeg_proc = None

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
        "rtsp://127.0.0.1:8554/test",
    ]
    print("Starting ffmpeg:", " ".join(cmd))
    # Start ffmpeg detached but keep a handle for shutdown
    ffmpeg_proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)

def stop_ffmpeg():
    global ffmpeg_proc
    if ffmpeg_proc is None:
        return
    try:
        if ffmpeg_proc.poll() is None:
            print("Terminating ffmpeg (pid=%s)" % ffmpeg_proc.pid)
            ffmpeg_proc.terminate()
            ffmpeg_proc.wait(timeout=5)
    except Exception:
        try:
            ffmpeg_proc.kill()
        except Exception:
            pass
    ffmpeg_proc = None

# ensure ffmpeg is stopped on exit
atexit.register(stop_ffmpeg)
def _handle_sigterm(signum, frame):
    stop_ffmpeg()
    raise SystemExit(0)

signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)
frame_i = 0

# Flask app to serve MJPEG stream
app = Flask(__name__)
last_frame = None
frame_lock = threading.Lock()

def detection_loop():
    global last_frame, frame_i
    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(0.1)
            continue
        frame_i += 1
        if SKIP_FRAMES > 1 and (frame_i % SKIP_FRAMES) != 0:
            continue

        # native frame used (no scaling)
        results = model.track(source=[frame], device=DEVICE, conf=0.3, stream=False, half=USE_CUDA)

        # draw boxes
        if results:
            r = results[0]
            boxes = getattr(r, "boxes", None)
            if boxes is not None:
                for box in boxes:
                    xyxy = box.xyxy.cpu().numpy().astype(int)[0]
                    conf = float(box.conf.cpu().numpy()[0])
                    cls = int(box.cls.cpu().numpy()[0])
                    # Resolve class name from the model if available
                    names = None
                    if hasattr(model, 'names'):
                        names = model.names
                    elif hasattr(model, 'model') and hasattr(model.model, 'names'):
                        names = model.model.names

                    label = str(cls)
                    if names is not None and cls in names:
                        label = names[cls]

                    # color-code common vehicle types
                    if label in ("car", "truck"):
                        box_color = (0, 0, 255)  # red for vehicles
                    else:
                        box_color = (0, 255, 0)  # green otherwise

                    cv2.rectangle(frame, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), box_color, 2)

                    # draw filled label background for readability
                    text = f"{label}:{conf:.2f}"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.5
                    thickness = 1
                    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, thickness)
                    # make a little padding
                    pad_x, pad_y = 4, 4
                    y1 = max(0, xyxy[1] - text_h - pad_y)
                    y2 = xyxy[1]
                    x1 = xyxy[0]
                    x2 = xyxy[0] + text_w + pad_x
                    cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, -1)
                    # put white text on the filled rectangle
                    text_org = (x1 + 2, y2 - 3)
                    cv2.putText(frame, text, text_org, font, font_scale, (255,255,255), thickness)

        # encode frame as JPEG
        ok, jpeg = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        with frame_lock:
            last_frame = jpeg.tobytes()

def gen():
    while True:
        with frame_lock:
            frame = last_frame
        if frame is None:
            time.sleep(0.05)
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        time.sleep(0.01)

@app.route('/')
def index():
    return render_template_string('<html><body><h3>YOLO Stream</h3><img src="/stream"/></body></html>')

@app.route('/stream')
def stream():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    # start ffmpeg to publish the remote HLS to local RTSP
    start_ffmpeg()
    # allow ffmpeg to warm up and retry opening the RTSP source until available
    cap = None
    max_retries = 15
    for attempt in range(1, max_retries + 1):
        time.sleep(1.0)
        print(f"Attempting to open VideoCapture (attempt {attempt}/{max_retries})")
        cap = cv2.VideoCapture(SOURCE, cv2.CAP_FFMPEG)
        if cap.isOpened():
            print("VideoCapture opened successfully")
            break
        else:
            try:
                cap.release()
            except Exception:
                pass
            cap = None
            print("VideoCapture not open yet, retrying...")

    if cap is None or not cap.isOpened():
        print(f"Failed to open VideoCapture after {max_retries} attempts. Stopping ffmpeg and exiting.")
        stop_ffmpeg()
        raise SystemExit(1)

    # start detection thread
    t = threading.Thread(target=detection_loop, daemon=True)
    t.start()
    print('Serving stream at http://0.0.0.0:5000')
    try:
        app.run(host='0.0.0.0', port=5000, threaded=True)
    finally:
        stop_ffmpeg()