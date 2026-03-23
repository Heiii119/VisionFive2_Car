import cv2
import time
import argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0, help="V4L2 device index (e.g., 0 for /dev/video0)")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera index {args.device} (try --device 1,2,...)")

    # Request settings (camera may choose the nearest supported mode)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    cap.set(cv2.CAP_PROP_FPS, args.fps)

    win = "StarFive Webcam Preview (press q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)

    last = time.time()
    fps_smooth = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            print("Frame read failed; retrying...")
            time.sleep(0.1)
            continue

        now = time.time()
        inst = 1.0 / max(1e-6, (now - last))
        last = now
        fps_smooth = 0.9 * fps_smooth + 0.1 * inst if fps_smooth else inst

        cv2.putText(frame, f"FPS: {fps_smooth:.1f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        cv2.imshow(win, frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
