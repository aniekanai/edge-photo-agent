import cv2
import numpy as np

def compute_brightness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)

def compute_sharpness(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def interpret_brightness(brightness):
    if brightness < 80:
        return "Too Dark"
    elif brightness > 180:
        return "Too Bright"
    else:
        return "OK"

def interpret_sharpness(sharpness):
    return "Sharp" if sharpness > 100 else "Blurry"

def compute_quality_score(brightness, sharpness):
    score = 0
    if 80 <= brightness <= 180:
        score += 50
    if sharpness > 100:
        score += 50
    return score

def detect_faces(frame, face_cascade):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

def is_face_centered(face, frame_width, tolerance=0.15):
    x, y, w, h = face
    face_center_x = x + w / 2
    frame_center_x = frame_width / 2
    offset = abs(face_center_x - frame_center_x) / frame_center_x
    return offset < tolerance
