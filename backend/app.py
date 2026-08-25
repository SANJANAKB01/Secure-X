from flask import Flask, Response, send_from_directory
import cv2
from flask_cors import CORS
import time
import mysql.connector
import os

app = Flask(__name__)
# === NEW === Allow CORS from your Render URL instead of just localhost
CORS(app, origins=["http://localhost:3000", "https://secure-x.onrender.com"])

# MySQL database configuration
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'sanchi01',  
    'database': 'securex'
}

def get_db_connection():
    return mysql.connector.connect(**db_config)

# Initialize the video capture and Haar cascades
cap = None # We keep this None so it doesn't crash looking for a cloud webcam
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')

# Initialize warning state
warning_displayed = False
warning_start_time = None

def generate_frames():
    global warning_displayed, warning_start_time
    # === NEW === If camera is None (cloud), just send a black dummy frame so the app doesn't crash
    if cap is None:
        while True:
            dummy_frame = cv2.putText(np.zeros((480, 640, 3), dtype=np.uint8), "Camera Off (Cloud Mode)", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            _, buffer = cv2.imencode('.jpg', dummy_frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            time.sleep(0.1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        face_count = len(faces)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 255, 0), 5)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]
            eyes = eye_cascade.detectMultiScale(roi_gray, 1.3, 5)
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 5)

        if face_count > 1:
            if not warning_displayed:
                warning_displayed = True
                warning_start_time = time.time()
        else:
            if warning_displayed and (time.time() - warning_start_time) >= 1:
                warning_displayed = False

        if warning_displayed:
            elapsed_time = time.time() - warning_start_time
            if elapsed_time < 360:
                warning_text = "Warning!!"
                cv2.putText(frame, warning_text, (frame.shape[1] // 2 - 150, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3, cv2.LINE_AA)

        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# ==========================================
# === NEW: THE MAGIC REACT SERVING CODE ===
# ==========================================
# CHANGE 'frontend' TO THE EXACT NAME OF YOUR REACT FOLDER!!!
REACT_BUILD_DIR = os.path.join(os.path.dirname(__file__), '..', 'securex-main', 'build')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve_react(path):
    # If the file exists in the React build folder, send it (like CSS, JS files)
    if path != "" and os.path.exists(os.path.join(REACT_BUILD_DIR, path)):
        return send_from_directory(REACT_BUILD_DIR, path)
    # Otherwise, send the index.html (this handles React Router)
    else:
        return send_from_directory(REACT_BUILD_DIR, 'index.html')
# ==========================================

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

import atexit
atexit.register(lambda: cap.release() if cap is not None else None)
