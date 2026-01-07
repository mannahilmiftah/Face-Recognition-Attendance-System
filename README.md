# Face Recognition Attendance System

## Project Overview
This project is a fully automatic face recognition based attendance system that uses deep learning face embeddings (FaceNet) combined with a Support Vector Machine (SVM) classifier to identify individuals in real time and record attendance without any manual intervention.

Unlike traditional pixel-based methods (e.g., KNN on raw images), this system follows a real-world biometric recognition pipeline:
- Deep CNN–based feature extraction
- Robust classification
- Confidence-based decision logic
- Real-time inference
- Automatic logging with duplicate prevention

## 🎯 Key Features
- Automatic attendance marking (no button press)
- Deep face embeddings (FaceNet – 512D)
- SVM classifier with probability calibration
- Unknown person detection using confidence threshold
- Anti-duplicate logic (cooldown-based)
- Real-time webcam inference
- Date-wise CSV attendance logs

## 🛠️ Technologies Used

- Python
- OpenCV – real-time video capture & visualization
- FaceNet (keras-facenet) – deep face embeddings
- MTCNN – robust face detection
- scikit-learn (SVM) – classification
- NumPy – numerical operations
- Joblib – model persistence
- CSV – attendance storage
- Text-to-Speech (Windows SAPI) – voice feedback

## 📂 Project Structure
```text
├── data/
│   └── enrolled/
│       ├── Mannahil Miftah/
│       ├── Muhammad Rayyan/
│       └── ...
├── models/
│   ├── svm_face_model.joblib
│   └── label_encoder.joblib
├── Attendance/
│   └── Attendance_DD-MM-YYYY.csv
├── enroll.py
├── train_svm.py
├── attendance_embedded.py
```

## 🔍 How the System Works
### 1️⃣ Dataset Creation (enroll.py)
- Captures face images using a webcam
- Stores images per person in a structured directory
- Supports adding new people or more samples for existing people
 
### 2️⃣ Model Training (train_svm.py)
- Detects faces using MTCNN
- Extracts 512-dimensional FaceNet embeddings
- Encodes labels using LabelEncoder
- Trains an SVM (RBF kernel) with probability calibration
- Saves trained model and encoder for reuse

### Automatic Attendance (attendance_embedded.py)
- Detects faces in real time
- Extracts FaceNet embeddings
- Predicts identity using SVM
- Applies confidence threshold for unknown detection
- Automatically records attendance with timestamp
- Prevents duplicate entries using cooldown logic
  
### ▶️ How to Run the Project

Create virtual environment
```python
python -m venv myvenv # for windows
```
Activate the environment
```python
.\myvenv\Scripts\Activate #for windows
```

#### Step 1: Install Dependencies
```python
pip install -r requirements.txt
```

#### Step 2: Collect Face Data
```python
python enroll.py
```
- Enter your name
- Enter number of samples (e.g., 30, 50, 100)
- Repeat for all individuals

#### Step 3: Train the Model
```python
python train_svm.py
```
This will create:
- models/svm_face_model.joblib
- models/label_encoder.joblib

#### Step 4: Run Automatic Attendance
```python
python attendance_embedded.py
```
##### Controls
- q → Quit program

Attendance will be saved as:
```text
Attendance/Attendance_DD-MM-YYYY.csv
```

### 📊 Output Example
CSV file format:
```text
NAME, TIME
Mannahil, 10:32:15
Sara, 10:33:02
```

### ⚙️ Configuration Options
#### 🔹 Unknown Person Threshold
```python
UNKNOWN_THRESHOLD = 0.65
```
- Increase → stricter recognition
- Decrease → more permissive recognition

#### 🔹 Duplicate Prevention
```python
COOLDOWN_SECONDS = 60
```
Prevents the same person from being marked repeatedly

### 🚀 Future Enhancements
- Mark attendance once per day instead of cooldown
- Multi-face attendance in a single frame
- CNN-based face alignment
- Database-backed attendance storage
- ROS / robot perception integration
- Edge deployment
