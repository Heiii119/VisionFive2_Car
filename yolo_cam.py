#!/usr/bin/env python3
import argparse
import threading
import time
import socket
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

# ----------------------------
# MJPEG Streaming (latest-frame, drop-old-frames)
# ----------------------------
_latest_jpeg = None
_latest_seq = 0
_latest_ts = 0.0
_latest_cond = threading.Condition()
_latest_frame = None
_frame_lock = threading.Lock()

BOUNDARY = "frame"

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

def _detect_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

class StreamHandler(BaseHTTPRequestHandler):
    server_version = "YOLOMJPEG/1.0"

    def log_message(self, fmt, *args):
        return

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            html = f"""<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>YOLOv5 MJPEG Stream</title>
    <style>
      body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 12px; }}
      img {{ max-width: 100%; height: auto; background: #000; }}
      code {{ background: #f2f2f2; padding: 2px 6px; border-radius: 6px; }}
    </style>
  </head>
  <body>
    <h2>YOLOv5 MJPEG Stream (Latest-Frame, Low-Latency)</h2>
    <p>Stream endpoint: <code>/mjpeg</code></p>
    <img src="/mjpeg" />
  </body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path == "/mjpeg":
            # Reduce latency on some networks (disable Nagle)
            try:
                self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except Exception:
                pass

            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-store, no-cache, must-revalidate, max-age=0")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={BOUNDARY}")
            self.end_headers()

            last_seq = -1

            try:
                while True:
                    # Wait for a NEW frame (sequence number changes)
                    with _latest_cond:
                        if _latest_jpeg is None:
                            _latest_cond.wait(timeout=1.0)

                        while _latest_seq == last_seq:
                            _latest_cond.wait(timeout=1.0)

                        jpg = _latest_jpeg
                        seq = _latest_seq

                    if jpg is None:
                        continue

                    last_seq = seq

                    self.wfile.write(f"--{BOUNDARY}\r\n".encode("ascii"))
                    self.wfile.write(b"Content-Type: image/jpeg\r\n")
                    self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode("ascii"))
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    self.wfile.flush()

            except (BrokenPipeError, ConnectionResetError):
                return
            except Exception:
                return

        self.send_response(404)
        self.end_headers()

def start_stream_server(host, port):
    server = ThreadedHTTPServer((host, port), StreamHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

def prompt_port(default_port=9090):
    while True:
        try:
            s = input(f"Enter streaming port (1-65535) [default {default_port}]: ").strip()
        except EOFError:
            # If running non-interactively, fall back to default
            return int(default_port)

        if s == "":
            return int(default_port)

        if s.isdigit():
            p = int(s)
            if 1 <= p <= 65535:
                return p

        print("Invalid port. Example valid ports: 8080, 9090, 5000.")

# ----------------------------
# YOLOv5 helpers
# ----------------------------
def load_class_names(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            names = [line.strip() for line in f.readlines() if line.strip()]
        return names if names else None
    except FileNotFoundError:
        return None

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114)):
    h, w = im.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / h, new_shape[1] / w)
    new_w = int(round(w * r))
    new_h = int(round(h * r))

    dw = new_shape[1] - new_w
    dh = new_shape[0] - new_h
    dw /= 2
    dh /= 2

    if (w, h) != (new_w, new_h):
        im = cv2.resize(im, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    top = int(round(dh - 0.1))
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1))
    right = int(round(dw + 0.1))

    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return im, r, (left, top)

def xywh_to_xyxy(xywh):
    x = xywh[:, 0]
    y = xywh[:, 1]
    w = xywh[:, 2]
    h = xywh[:, 3]
    x1 = x - w / 2.0
    y1 = y - h / 2.0
    x2 = x + w / 2.0
    y2 = y + h / 2.0
    return np.stack([x1, y1, x2, y2], axis=1)

def detect_yolov5_opencv(net, frame_bgr, conf_thres=0.25, iou_thres=0.45, input_size=640):
    img, r, (padw, padh) = letterbox(frame_bgr, (input_size, input_size))
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=1.0 / 255.0, size=(input_size, input_size), swapRB=True, crop=False
    )
    net.setInput(blob)
    out = net.forward()

    if out.ndim == 3:
        pred = out[0]
    elif out.ndim == 2:
        pred = out
    else:
        pred = np.squeeze(out)

    if pred.ndim != 2 or pred.shape[1] < 6:
        raise RuntimeError(f"Unexpected output shape from YOLOv5 ONNX: {out.shape}")

    boxes_xywh = pred[:, 0:4].astype(np.float32)
    obj = pred[:, 4].astype(np.float32)
    cls_scores = pred[:, 5:].astype(np.float32)

    cls_id = np.argmax(cls_scores, axis=1)
    cls_conf = cls_scores[np.arange(cls_scores.shape[0]), cls_id]
    scores = obj * cls_conf

    keep = scores >= conf_thres
    if not np.any(keep):
        return []

    boxes_xywh = boxes_xywh[keep]
    scores = scores[keep]
    cls_id = cls_id[keep]

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - padw) / r
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - padh) / r

    h0, w0 = frame_bgr.shape[:2]
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w0 - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h0 - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w0 - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h0 - 1)

    boxes_xywh_nms = []
    for b in boxes_xyxy:
        x1, y1, x2, y2 = b
        boxes_xywh_nms.append([int(x1), int(y1), int(x2 - x1), int(y2 - y1)])

    indices = cv2.dnn.NMSBoxes(
        bboxes=boxes_xywh_nms,
        scores=scores.tolist(),
        score_threshold=conf_thres,
        nms_threshold=iou_thres,
    )

    results = []
    if len(indices) > 0:
        for i in indices.flatten():
            x1, y1, w, h = boxes_xywh_nms[i]
            x2 = x1 + w
            y2 = y1 + h
            results.append((x1, y1, x2, y2, float(scores[i]), int(cls_id[i])))

    return results

def draw_detections(frame, detections, class_names=None):
    for (x1, y1, x2, y2, score, cid) in detections:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if class_names and 0 <= cid < len(class_names):
            label = f"{class_names[cid]} {score:.2f}"
        else:
            label = f"id:{cid} {score:.2f}"
        y = max(0, y1 - 7)
        cv2.putText(frame, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
    return frame

def capture_worker(cap):
    global _latest_frame
    while True:
        ok, frame = cap.read()
        if not ok or frame is None:
            time.sleep(0.005)
            continue
        with _frame_lock:
            _latest_frame = frame
            
# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    # Requested defaults:
    ap.add_argument("--model", default="yolov5n.onnx", help="Path to yolov5n.onnx")
    ap.add_argument("--source", default="/dev/video4", help="Camera index like 0, or a video path, or a URL")
    ap.add_argument("--host", default="0.0.0.0", help="Streaming bind host")

    # Ask port at runtime unless user provides --port
    ap.add_argument("--port", type=int, default=None, help="Streaming port (if omitted, prompt at startup)")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size (usually 640)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--names", default="", help="Optional path to class names file (coco.names)")
    ap.add_argument("--jpeg_quality", type=int, default=60, help="JPEG quality 1-100")
    ap.add_argument("--show_local", action="store_true", help="Show local preview window (needs GUI)")
    ap.add_argument("--cap_width", type=int, default=320, help="Optional capture width (0 = default)")
    ap.add_argument("--cap_height", type=int, default=240, help="Optional capture height (0 = default)")
    ap.add_argument("--cap_fps", type=int, default=0, help="Optional capture fps (0 = default)")
    args = ap.parse_args()

    port = args.port if args.port is not None else prompt_port(default_port=9090)

    class_names = load_class_names(args.names) if args.names else None

    net = cv2.dnn.readNetFromONNX(args.model)

    # Allow camera as index "0" or as device path "/dev/video4"
    s = str(args.source)

    if s.startswith("/dev/video"):
        src = s
    elif s.isdigit():
        src = int(s)
    else:
        src = s

    # Prefer V4L2 on Linux (reduces GStreamer issues)
    cap = cv2.VideoCapture(src, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")
        t_cap = threading.Thread(target=capture_worker, args=(cap,), daemon=True)
        t_cap.start()

    # Reduce camera buffering if supported
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
    if args.cap_width:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.cap_width)
    if args.cap_height:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.cap_height)
    if args.cap_fps:
        cap.set(cv2.CAP_PROP_FPS, args.cap_fps)

    # Start server
    start_stream_server(args.host, port)

    ip_for_print = _detect_local_ip() if args.host == "0.0.0.0" else args.host
    print(f"Streaming page: http://{ip_for_print}:{port}/")
    print(f"MJPEG endpoint: http://{ip_for_print}:{port}/mjpeg")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    fps_t0 = time.time()
    fps_frames = 0
    last_fps = 0.0

    global _latest_jpeg, _latest_seq, _latest_ts

    try:
        while True:
            with _frame_lock:
                frame = None if _latest_frame is None else _latest_frame.copy()

            if frame is None:
                time.sleep(0.005)
                continue

            detections = detect_yolov5_opencv(
                net,
                frame,
                conf_thres=args.conf,
                iou_thres=args.iou,
                input_size=args.imgsz,
            )

            annotated = frame.copy()
            draw_detections(annotated, detections, class_names=class_names)

            # FPS overlay
            fps_frames += 1
            dt = time.time() - fps_t0
            if dt >= 1.0:
                last_fps = fps_frames / dt
                fps_t0 = time.time()
                fps_frames = 0

            cv2.putText(
                annotated,
                f"FPS: {last_fps:.1f}  det: {len(detections)}",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 0, 0),
                2,
            )

            # Encode once; all clients receive this latest JPEG
            ok2, jpg = cv2.imencode(".jpg", annotated, encode_params)
            if ok2:
                payload = jpg.tobytes()
                with _latest_cond:
                    _latest_jpeg = payload
                    _latest_seq += 1
                    _latest_ts = time.time()
                    _latest_cond.notify_all()

            if args.show_local:
                cv2.imshow("YOLOv5 Stream", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        cap.release()
        if args.show_local:
            cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
