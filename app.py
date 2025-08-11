from flask import Flask, render_template, Response
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import pickle
import os
from datetime import datetime
import pandas as pd

app = Flask(__name__)

# Load model and label dictionary
model = load_model("face_attendance/model/facenet_cnn.keras")
with open("face_attendance/model/label_dict.pkl", "rb") as f:
    label_dict = pickle.load(f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

recognized = set()

# Thresholds
CONFIDENCE_THRESHOLD = 0.55
MARGIN_THRESHOLD = 0.10

def prewhiten(x):
    mean = np.mean(x)
    std = np.std(x)
    std_adj = np.maximum(std, 1.0/np.sqrt(x.size))
    y = (x - mean) / std_adj
    return y

def mark_attendance(name):
    now = datetime.now()
    dt_string = now.strftime('%Y-%m-%d %H:%M:%S')
    file_path = 'face_attendance/attendance.csv'

    if not os.path.exists(file_path):
        pd.DataFrame(columns=['Name', 'Time']).to_csv(file_path, index=False)

    df = pd.read_csv(file_path)
    if name not in df['Name'].values:
        new_row = pd.DataFrame({'Name': [name], 'Time': [dt_string]})
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(file_path, index=False)

def gen_frames():
    cap = cv2.VideoCapture(0)
    global recognized
    recognized.clear()

    while True:
        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for x, y, w, h in faces:
            face_img = frame[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (100, 100)) / 255.0
            face_img = prewhiten(face_img)
            face_img = np.expand_dims(face_img, axis=0)

            pred = model.predict(face_img, verbose=0)[0]
            label = np.argmax(pred)
            confidence = pred[label]
            top2 = np.sort(pred)[-2:]
            margin = top2[-1] - top2[-2]

            if confidence >= CONFIDENCE_THRESHOLD and margin >= MARGIN_THRESHOLD:
                name = label_dict[label]
                if name not in recognized:
                    mark_attendance(name)
                    recognized.add(name)
                cv2.putText(frame, name, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "Unknown", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def show_attendance():
    df = pd.read_csv('face_attendance/attendance.csv')
    return df.to_html(index=False)

if __name__ == '__main__':
    app.run(debug=True)
