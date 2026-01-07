import os
import cv2
import time

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def main():
    name = input("Enter person name: ").strip()
    n_samples = int(input("How many images to capture? (e.g., 20-50): ").strip())

    out_dir = os.path.join("data", "enrolled", name)
    ensure_dir(out_dir)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    print("Press SPACE to capture an image, Q to quit early.")
    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        display = frame.copy()
        cv2.putText(display, f"{name} | Captured: {count}/{n_samples}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
        cv2.putText(display, "SPACE=capture  Q=quit",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Enroll", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key == 32:  # SPACE
            filename = os.path.join(out_dir, f"{int(time.time()*1000)}.jpg")
            cv2.imwrite(filename, frame)
            count += 1
            print("Saved:", filename)

        if count >= n_samples:
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Done. Images saved in:", out_dir)

if __name__ == "__main__":
    main()