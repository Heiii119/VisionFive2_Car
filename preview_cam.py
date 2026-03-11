import cv2

def main():
    cap = cv2.VideoCapture(0)  # change to /dev/video1 etc if needed
    if not cap.isOpened():
        raise RuntimeError("Cannot open webcam. Try another index or check /dev/video*")

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Failed to read frame")
            break

        cv2.imshow("USB Webcam Preview", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
