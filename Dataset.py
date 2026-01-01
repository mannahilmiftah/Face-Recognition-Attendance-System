# import cv2
# import pickle
# import numpy as np
# import os
# video=cv2.VideoCapture(0)
# facedetect=cv2.CascadeClassifier('haarcascade_frontalface_default .xml')

# faces_data=[]

# i=0

# name=input("Enter Your Name: ")

# while True:
#     ret,frame=video.read()
#     gray=cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     faces=facedetect.detectMultiScale(gray, 1.3 ,5)
#     for (x,y,w,h) in faces:
#         crop_img=frame[y:y+h, x:x+w, :]
#         resized_img=cv2.resize(crop_img, (50,50))
#         if len(faces_data)<=100 and i%10==0:
#             faces_data.append(resized_img)
#         i=i+1
#         cv2.putText(frame, str(len(faces_data)), (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (50,50,255), 1)
#         cv2.rectangle(frame, (x,y), (x+w, y+h), (50,50,255), 1)
#     cv2.imshow("Frame",frame)
#     k=cv2.waitKey(1)
#     if len(faces_data)==50:
#         break
# video.release()
# cv2.destroyAllWindows()

# faces_data=np.asarray(faces_data)
# faces_data=faces_data.reshape(100, -1)


# if 'names.pkl' not in os.listdir('data/'):
#     names=[name]*100
#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(names, f)
# else:
#     with open('data/names.pkl', 'rb') as f:
#         names=pickle.load(f)
#     names=names+[name]*100
#     with open('data/names.pkl', 'wb') as f:
#         pickle.dump(names, f)

# if 'faces_data.pkl' not in os.listdir('data/'):
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(faces_data, f)
# else:
#     with open('data/faces_data.pkl', 'rb') as f:
#         faces=pickle.load(f)
#     faces=np.append(faces, faces_data, axis=0)
#     with open('data/faces_data.pkl', 'wb') as f:
#         pickle.dump(faces, f)

import cv2
import pickle
import numpy as np
import os

video = cv2.VideoCapture(0)
facedetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # remove extra space

faces_data = []
i = 0

name = input("Enter Your Name: ")

TARGET_SAMPLES = 100
FACE_SIZE = (50, 50)  # width, height

while True:
    ret, frame = video.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        crop = gray[y:y+h, x:x+w]                 # grayscale crop
        resized = cv2.resize(crop, FACE_SIZE)     # 50x50
        if len(faces_data) < TARGET_SAMPLES and i % 10 == 0:
            faces_data.append(resized)
        i += 1

        cv2.putText(frame, str(len(faces_data)), (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (50, 50, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 1)

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    if len(faces_data) >= TARGET_SAMPLES:
        break

video.release()
cv2.destroyAllWindows()

faces_data = np.asarray(faces_data)                 # (N, 50, 50)
faces_data = faces_data.reshape(len(faces_data), -1)  # (N, 2500)

# labels length MUST match number of face samples
names = [name] * len(faces_data)

os.makedirs("data", exist_ok=True)

# Save / append names
names_path = "data/names.pkl"
faces_path = "data/faces_data.pkl"

if not os.path.exists(names_path):
    with open(names_path, "wb") as f:
        pickle.dump(names, f)
else:
    with open(names_path, "rb") as f:
        old = pickle.load(f)
    with open(names_path, "wb") as f:
        pickle.dump(old + names, f)

# Save / append faces
if not os.path.exists(faces_path):
    with open(faces_path, "wb") as f:
        pickle.dump(faces_data, f)
else:
    with open(faces_path, "rb") as f:
        old_faces = pickle.load(f)
    new_faces = np.append(old_faces, faces_data, axis=0)
    with open(faces_path, "wb") as f:
        pickle.dump(new_faces, f)

print("Saved:", faces_data.shape, "and labels:", len(names))