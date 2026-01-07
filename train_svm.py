import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from mtcnn import MTCNN
from keras_facenet import FaceNet

DATA_DIR = os.path.join("data", "enrolled")
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

SVM_PATH = os.path.join(MODEL_DIR, "svm_face_model.joblib")
LE_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")

IMG_SIZE = (160, 160)  # FaceNet expects 160x160

detector = MTCNN()
embedder = FaceNet()

def detect_and_crop_face(img_bgr):
    """Returns aligned-ish cropped face resized to 160x160, or None if no face found."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = detector.detect_faces(img_rgb)
    if not results:
        return None

    # pick the largest face
    results = sorted(results, key=lambda r: r["box"][2]*r["box"][3], reverse=True)
    x, y, w, h = results[0]["box"]
    x, y = max(0, x), max(0, y)
    face = img_rgb[y:y+h, x:x+w]
    if face.size == 0:
        return None

    face = cv2.resize(face, IMG_SIZE)
    return face

def face_embedding(face_rgb_160):
    """FaceNet embedding for one 160x160 RGB face."""
    emb = embedder.embeddings([face_rgb_160])[0]  # shape: (512,)
    return emb.astype(np.float32)

def load_dataset():
    X, y = [], []
    people = [d for d in os.listdir(DATA_DIR) if os.path.isdir(os.path.join(DATA_DIR, d))]
    if not people:
        raise RuntimeError(f"No enrolled people found in {DATA_DIR}")

    for person in people:
        person_dir = os.path.join(DATA_DIR, person)
        for fn in os.listdir(person_dir):
            if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            path = os.path.join(person_dir, fn)
            img = cv2.imread(path)
            if img is None:
                continue

            face = detect_and_crop_face(img)
            if face is None:
                continue

            emb = face_embedding(face)
            X.append(emb)
            y.append(person)

    return np.array(X), np.array(y)

def main():
    print("Loading dataset + extracting embeddings...")
    X, y = load_dataset()
    print("Embeddings:", X.shape, "Labels:", y.shape)

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Split for evaluation (optional but recommended)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y_enc, test_size=0.2, random_state=42, stratify=y_enc
    )

    # SVM classifier
    base_svm = SVC(kernel="rbf", probability=True, class_weight="balanced")

    # Calibrate probabilities (more reliable confidence)
    clf = CalibratedClassifierCV(base_svm, method="sigmoid", cv=3)
    clf.fit(X_train, y_train)

    # Evaluate
    y_pred = clf.predict(X_val)
    print("\nValidation report:\n")
    print(classification_report(y_val, y_pred, target_names=le.classes_))

    # Save
    joblib.dump(clf, SVM_PATH)
    joblib.dump(le, LE_PATH)
    print("\nSaved model:", SVM_PATH)
    print("Saved label encoder:", LE_PATH)

if __name__ == "__main__":
    main()