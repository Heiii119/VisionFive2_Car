import cv2
import time
import numpy as np

MODEL_PATH = "yolov8n.onnx"
DEV = "/dev/video4"

CONF_THRES = 0.35
IOU_THRES  = 0.45
INP_SIZE   = 640  # YOLOv8n default

def letterbox(im, new_shape=640, color=(114, 114, 114)):
    h, w = im.shape[:2]
    scale = min(new_shape / w, new_shape / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    im_resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w = new_shape - nw
    pad_h = new_shape - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left

    out = cv2.copyMakeBorder(im_resized, top, bottom, left, right,
                             cv2.BORDER_CONSTANT, value=color)
    return out, scale, left, top

def open_camera_mjpeg(dev):
    # Most reliable: force MJPEG via GStreamer
    pipeline = (
        f"v4l2src device={dev} ! "
        f"image/jpeg,width=640,height=480,framerate=30/1 ! "
        "jpegdec ! videoconvert ! appsink drop=true sync=false"
    )
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
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
