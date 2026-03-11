import cv2

def main():
    dev = "/dev/video4"   # try "/dev/video5" if needed
    cap = cv2.VideoCapture(dev, cv2.CAP_V4L2)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {dev}. Try /dev/video5 or check v4l2-ctl formats.")

    # Many webcams work best with MJPG on small boards
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        cv2.imshow("USB Webcam Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
