import cv2
import numpy as np

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant",
           "sheep", "sofa", "train", "tvmonitor"]

net = cv2.dnn.readNetFromCaffe(
    "MobileNetSSD_deploy.prototxt",
    "MobileNetSSD_deploy.caffemodel"
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Camera not detected!")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(
        cv2.resize(frame, (300, 300)),
        0.007843,
        (300, 300),
        127.5
    )

    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]

        if confidence > 0.5:
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            label = f"{CLASSES[idx]}: {confidence:.2f}"

            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

            cv2.putText(frame, label,
                        (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 2)

    cv2.imshow("MobileNet-SSD Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
