#!/usr/bin/env python3
import cv2
import numpy as np
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn

# ===============================
# CONFIG
# ===============================
MODEL_PATH = "best.onnx"
CAMERA_DEVICE = "/dev/video4"
IMG_SIZE = 320          # 320 is much faster than 640 on VF2
CONF_THRES = 0.4
IOU_THRES = 0.45
STREAM_PORT = 9090
JPEG_QUALITY = 40
DETECT_EVERY = 3        # Run detection every 3 frames (important for speed)

# ===============================
# Globals
# ===============================
latest_frame = None
latest_jpeg = None
frame_lock = threading.Lock()
cond = threading.Condition()

# ===============================
# Letterbox
# ===============================
def letterbox(im, new_shape=(320, 320)):
    h, w = im.shape[:2]
    r = min(new_shape[0] / h, new_shape[1] / w)
    nw, nh = int(w * r), int(h * r)

    resized = cv2.resize(im, (nw, nh))
    canvas = np.full((new_shape[0], new_shape[1], 3), 114, dtype=np.uint8)

    top = (new_shape[0] - nh) // 2
    left = (new_shape[1] - nw) // 2

    canvas[top:top+nh, left:left+nw] = resized
    return canvas, r, left, top

def xywh_to_xyxy(x):
    y = np.zeros_like(x)
    y[:,0] = x[:,0] - x[:,2]/2
    y[:,1] = x[:,1] - x[:,3]/2
    y[:,2] = x[:,0] + x[:,2]/2
    y[:,3] = x[:,1] + x[:,3]/2
    return y

# ===============================
# Detection
# ===============================
def detect(net, frame):
    img, r, padw, padh = letterbox(frame, (IMG_SIZE, IMG_SIZE))

    blob = cv2.dnn.blobFromImage(img, 1/255.0, (IMG_SIZE, IMG_SIZE), swapRB=True)
    net.setInput(blob)
    pred = net.forward()[0]

    boxes = pred[:, :4]
    obj = pred[:, 4]
    cls = pred[:, 5:]

    cls_id = np.argmax(cls, axis=1)
    cls_conf = cls[np.arange(len(cls)), cls_id]
    scores = obj * cls_conf

    mask = scores > CONF_THRES
    boxes = boxes[mask]
    scores = scores[mask]
    cls_id = cls_id[mask]

    if len(boxes) == 0:
        return []

    boxes = xywh_to_xyxy(boxes)

    boxes[:, [0,2]] = (boxes[:, [0,2]] - padw) / r
    boxes[:, [1,3]] = (boxes[:, [1,3]] - padh) / r

    boxes = boxes.astype(int)

    boxes_nms = []
    for b in boxes:
        x1,y1,x2,y2 = b
        boxes_nms.append([x1,y1,x2-x1,y2-y1])

    idx = cv2.dnn.NMSBoxes(boxes_nms, scores.tolist(), CONF_THRES, IOU_THRES)

    results = []
    if len(idx) > 0:
        for i in idx.flatten():
            x,y,w,h = boxes_nms[i]
            results.append((x,y,x+w,y+h,scores[i],cls_id[i]))

    return results

# ===============================
# Camera Thread
# ===============================
def capture_thread(cap):
    global latest_frame
    while True:
        ret, frame = cap.read()
        if ret:
            with frame_lock:
                latest_frame = frame

# ===============================
# HTTP Server
# ===============================
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path != "/mjpeg":
            self.send_response(200)
            self.send_header("Content-Type","text/html")
            self.end_headers()
            self.wfile.write(b'<img src="/mjpeg">')
            return

        self.send_response(200)
        self.send_header("Content-Type","multipart/x-mixed-replace; boundary=frame")
        self.end_headers()

        while True:
            with cond:
                cond.wait()
                jpg = latest_jpeg

            self.wfile.write(b"--frame\r\n")
            self.wfile.write(b"Content-Type: image/jpeg\r\n\r\n")
            self.wfile.write(jpg)
            self.wfile.write(b"\r\n")

# ===============================
# MAIN
# ===============================
def main():
    global latest_jpeg

    net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    cap = cv2.VideoCapture(CAMERA_DEVICE, cv2.CAP_V4L2)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    cap.set(cv2.CAP_PROP_FPS, 15)

    threading.Thread(target=capture_thread, args=(cap,), daemon=True).start()

    server = ThreadedHTTPServer(("0.0.0.0", STREAM_PORT), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    print(f"Open browser: http://YOUR_BOARD_IP:{STREAM_PORT}")

    frame_i = 0
    detections = []

    while True:
        with frame_lock:
            if latest_frame is None:
                continue
            frame = latest_frame.copy()

        frame_i += 1

        if frame_i % DETECT_EVERY == 0:
            detections = detect(net, frame)

        for x1,y1,x2,y2,score,cid in detections:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

        ok, jpg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if ok:
            with cond:
                latest_jpeg = jpg.tobytes()
                cond.notify_all()

        time.sleep(0.01)

if __name__ == "__main__":
    main()
