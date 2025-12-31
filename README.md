# Face Recognition Attendance System

## Project Overview
This project is a real-time face recognition based attendance system built using Computer Vision and Machine Learning techniques. It uses a webcam to detect and recognize faces, and automatically records attendance with timestamps into CSV files.

The system demonstrates a complete computer vision pipeline:
- Face detection
- Feature extraction
- Machine learning–based classification
- Real-time inference
- Automated attendance logging

This project was designed as a foundational step toward intelligent systems that perceive their environment and make decisions autonomously.

## 🛠️ Technologies Used

- Python
- OpenCV – face detection & image processing
- scikit-learn (KNN) – face classification
- NumPy – numerical operations
- Pickle – dataset persistence
- CSV – attendance storage
- Text-to-Speech (Windows SAPI) – voice feedback

## 📂 Project Structure
```text
├── Dataset.py              # Collects and stores face samples
├── Attendance.py           # Real-time face recognition & attendance
├── data/
│   ├── faces_data.pkl      # Flattened face feature vectors
│   └── names.pkl           # Corresponding labels (names)
├── Attendance/
│   └── Attendance_DD-MM-YYYY.csv
├── haarcascade_frontalface_default.xml
├── bg.png                  # UI background (optional)
```

## 🔍 How the System Works
### 1️⃣ Dataset Creation (Dataset.py)
- Captures face images using a webcam
- Detects faces using Haar Cascade
- Converts images to grayscale
- Resizes faces to 50×50
- Flattens each face into a 2500-dimensional feature vector
- Stores:
  - Face data → faces_data.pkl
  - Labels (names) → names.pkl
- Supports:
  - Multiple people
  - Custom number of samples per person
  - Adding more samples for an existing person
 
### 2️⃣ Model Training & Recognition (Attendance.py)
- Loads stored face features and labels
- Trains a K-Nearest Neighbors (KNN) classifier
- Performs real-time face recognition
- Displays predicted name on screen
- On key press (o):
  - Records attendance with timestamp
  - Saves it to a date-wise CSV file
  - Announces attendance using voice feedback

### ▶️ How to Run the Project
#### Step 1: Install Dependencies
```python
pip install -r requirements.txt
```

#### Step 2: Collect Face Data
```python
python Dataset.py
```
- Enter your name
- Enter number of samples (e.g., 30, 50, 100)
- Face the camera until capture completes

#### Step 3: Run Attendance System
```python
python Attendance.py
```
##### Controls
- o → Take attendance
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
