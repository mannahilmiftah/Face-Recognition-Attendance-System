import os
import cv2
import numpy as np
import time
import csv
from datetime import datetime
import joblib

from mtcnn import MTCNN
from keras_facenet import FaceNet

MODEL_DIR = "models"
SVM_PATH = os.path.join(MODEL_DIR, "svm_face_model.joblib")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

ATT_DIR = "Attendance"
os.makedirs(ATT_DIR, exist_ok=True)

IMG_SIZE = (160, 160)
UNKNOWN_THRESHOLD = 0.65

# cooldown: don't mark the same person again within this many seconds
COOLDOWN_SECONDS = 60

detector = MTCNN()
embedder = FaceNet()

def write_attendance(name: str, timestamp: str, date: str):
    csv_path = os.path.join(ATT_DIR, f"Attendance_{date}.csv")
    exists = os.path.isfile(csv_path)

    with open(csv_path, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["NAME", "TIME"])
        w.writerow([name, timestamp])

def main():
    clf = joblib.load(SVM_PATH)
    le = joblib.load(LE_PATH)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    # Store last marked times to avoid duplicates
    last_marked = {}  # {name: unix_time}

    print("Auto attendance is ON. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.detect_faces(img_rgb)

        display_text = "No face"

        if results:
            # choose largest face
            results = sorted(results, key=lambda r: r["box"][2]*r["box"][3], reverse=True)
            x, y, w, h = results[0]["box"]
            x, y = max(0, x), max(0, y)

            face = img_rgb[y:y+h, x:x+w]
            if face.size != 0:
                face = cv2.resize(face, IMG_SIZE)
                emb = embedder.embeddings([face])[0].astype(np.float32).reshape(1, -1)

                probs = clf.predict_proba(emb)[0]
                idx = int(np.argmax(probs))
                conf = float(probs[idx])

                if conf < UNKNOWN_THRESHOLD:
                    name = "Unknown"
                else:
                    name = le.inverse_transform([idx])[0]

                display_text = f"{name} ({conf:.2f})"

                # Draw UI box
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.rectangle(frame, (x, y-35), (x+w, y), (0, 0, 255), -1)
                cv2.putText(frame, display_text, (x, y-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

                # ✅ AUTO MARK LOGIC
                if name != "Unknown":
                    now = time.time()
                    last_time = last_marked.get(name, 0)

                    if (now - last_time) >= COOLDOWN_SECONDS:
                        date = datetime.fromtimestamp(now).strftime("%d-%m-%Y")
                        timestamp = datetime.fromtimestamp(now).strftime("%H:%M:%S")

                        write_attendance(name, timestamp, date)
                        last_marked[name] = now

                        print(f"[AUTO] Marked: {name} at {timestamp}")

        cv2.putText(frame, "Auto attendance ON | q=quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        cv2.imshow("Auto Attendance (FaceNet + SVM)", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()