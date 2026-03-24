#!/usr/bin/env python3
import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

import cv2
import numpy as np

# ----------------------------
# MJPEG Streaming Server
# ----------------------------
_latest_jpeg = None
_latest_lock = threading.Lock()

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/", "/index.html"):
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            html = """
<!doctype html>
<html>
  <head><title>YOLO Stream</title></head>
  <body>
    <h2>YOLOv5 MJPEG Stream</h2>
    <p>Stream endpoint: <code>/mjpeg</code></p>
    <img src="/mjpeg" style="max-width: 100%; height: auto;" />
  </body>
</html>
"""
            self.wfile.write(html.encode("utf-8"))
            return

        if self.path == "/mjpeg":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.end_headers()

            try:
                while True:
                    with _latest_lock:
                        jpg = _latest_jpeg

                    if jpg is None:
                        time.sleep(0.05)
                        continue

                    self.wfile.write(b"--frame\r\n")
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Content-Length", str(len(jpg)))
                    self.end_headers()
                    self.wfile.write(jpg)
                    self.wfile.write(b"\r\n")
                    time.sleep(0.02)
            except BrokenPipeError:
                return
            except ConnectionResetError:
                return

        self.send_response(404)
        self.end_headers()

def start_stream_server(host, port):
    server = ThreadedHTTPServer((host, port), StreamHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server

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
    # Resize + pad to keep aspect ratio (YOLO-style)
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
    # xywh is Nx4 in center format
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
    # Preprocess
    img, r, (padw, padh) = letterbox(frame_bgr, (input_size, input_size))
    blob = cv2.dnn.blobFromImage(
        img, scalefactor=1.0 / 255.0, size=(input_size, input_size), swapRB=True, crop=False
    )
    net.setInput(blob)

    # Forward
    out = net.forward()

    # Normalize output shape
    # Common: (1, 25200, 85)
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

    # Undo letterbox: from input coords -> original frame coords
    boxes_xyxy[:, [0, 2]] = (boxes_xyxy[:, [0, 2]] - padw) / r
    boxes_xyxy[:, [1, 3]] = (boxes_xyxy[:, [1, 3]] - padh) / r

    # Clip
    h0, w0 = frame_bgr.shape[:2]
    boxes_xyxy[:, 0] = np.clip(boxes_xyxy[:, 0], 0, w0 - 1)
    boxes_xyxy[:, 1] = np.clip(boxes_xyxy[:, 1], 0, h0 - 1)
    boxes_xyxy[:, 2] = np.clip(boxes_xyxy[:, 2], 0, w0 - 1)
    boxes_xyxy[:, 3] = np.clip(boxes_xyxy[:, 3], 0, h0 - 1)

    # Prepare for NMSBoxes (expects xywh ints)
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

# ----------------------------
# Main
# ----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolov5n.onnx", help="Path to yolov5n.onnx")
    ap.add_argument("--source", default="0", help="Camera index like 0, or a video path, or a URL")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference size (usually 640)")
    ap.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    ap.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    ap.add_argument("--names", default="", help="Optional path to class names file (coco.names)")
    ap.add_argument("--host", default="0.0.0.0", help="Streaming bind host")
    ap.add_argument("--port", type=int, default=8080, help="Streaming port")
    ap.add_argument("--jpeg_quality", type=int, default=80, help="JPEG quality 1-100")
    ap.add_argument("--show_local", action="store_true", help="Show local preview window (needs GUI)")
    args = ap.parse_args()

    class_names = load_class_names(args.names) if args.names else None

    # Load network
    net = cv2.dnn.readNetFromONNX(args.model)

    # If your OpenCV build supports it, these can help (safe to keep commented if unsure):
    # net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    # net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    # Open source
    src = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {args.source}")

    # Start streaming server
    start_stream_server(args.host, args.port)
    print(f"Streaming page:   http://{args.host}:{args.port}/")
    print(f"MJPEG endpoint:   http://{args.host}:{args.port}/mjpeg")
    print("Tip: if args.host is 0.0.0.0, use your board IP on the other device.")

    encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), int(args.jpeg_quality)]

    fps_t0 = time.time()
    fps_frames = 0
    last_fps = 0.0

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.02)
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

            # FPS counter (overlay)
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

            # Update JPEG for web stream
            ok2, jpg = cv2.imencode(".jpg", annotated, encode_params)
            if ok2:
                with _latest_lock:
                    global _latest_jpeg
                    _latest_jpeg = jpg.tobytes()

            # Optional local preview
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
    if cap.isOpened():
        return cap

    # Fallback: V4L2 (may or may not honor MJPG)
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)
    if not cap.isOpened():
        return cap
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def main():
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    cap = open_camera_mjpeg(DEV)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera {DEV}")

    frame_i = 0
    t0 = time.time()
    fps_count = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed")
            break

        img, scale, pad_x, pad_y = letterbox(frame, INP_SIZE)
        blob = cv2.dnn.blobFromImage(img, scalefactor=1/255.0,
                                     size=(INP_SIZE, INP_SIZE),
                                     swapRB=True, crop=False)
        net.setInput(blob)

        pred = net.forward()

        # Typical YOLOv8 ONNX: (1, 84, 8400) -> transpose to (8400, 84)
        pred = np.squeeze(pred)
        if pred.ndim == 2 and pred.shape[0] in (84, 85):
            pred = pred.T  # (N, C)

        boxes = []
        confs = []
        class_ids = []

        for row in pred:
            # row: [cx, cy, w, h, cls0..]
            cx, cy, w, h = row[0:4]
            scores = row[4:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf < CONF_THRES:
                continue

            # Convert from letterboxed coords back to original frame coords
            x = (cx - 0.5 * w - pad_x) / scale
            y = (cy - 0.5 * h - pad_y) / scale
            ww = w / scale
            hh = h / scale

            boxes.append([int(x), int(y), int(ww), int(hh)])
            confs.append(conf)
            class_ids.append(class_id)

        idxs = cv2.dnn.NMSBoxes(boxes, confs, CONF_THRES, IOU_THRES)

        # Draw detections
        if len(idxs) > 0:
            for j in idxs.flatten():
                x, y, ww, hh = boxes[j]
                x = max(0, x); y = max(0, y)
                ww = max(1, ww); hh = max(1, hh)

                cv2.rectangle(frame, (x, y), (x + ww, y + hh), (0, 255, 0), 2)
                label = f"id={class_ids[j]} conf={confs[j]:.2f}"
                cv2.putText(frame, label, (x, max(0, y - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        fps_count += 1
        if fps_count >= 30:
            t1 = time.time()
            fps = fps_count / (t1 - t0)
            print(f"Approx FPS: {fps:.2f} | detections: {len(idxs) if len(idxs)>0 else 0}")
            t0 = t1
            fps_count = 0

        # Save an annotated frame every 30 frames (headless-friendly)
        if frame_i % 30 == 0:
            cv2.imwrite("annotated_latest.jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            print("Wrote annotated_latest.jpg")

        frame_i += 1

    cap.release()

if __name__ == "__main__":
    main()
