import cv2
import numpy as np

def compute_brightness(frame):
    """
    Compute the average brightness of a frame.

    We convert the frame to grayscale and take the mean pixel
    intensity. Higher values indicate a brighter scene.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return np.mean(gray)


def compute_sharpness(frame):
    """
    Compute image sharpness using the variance of the Laplacian.

    A higher variance indicates a sharper image.
    A low variance suggests blur or motion.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def interpret_brightness(brightness):
    """
    Convert raw brightness value into a human-readable label.
    """
    if brightness < 80:
        return "Too Dark"
    elif brightness > 180:
        return "Too Bright"
    else:
        return "OK"


def interpret_sharpness(sharpness):
    """
    Convert sharpness score into a readable label.
    """
    return "Sharp" if sharpness > 100 else "Blurry"


def compute_quality_score(brightness, sharpness):
    """
    Compute a simple quality score (0–100) based on
    brightness and sharpness thresholds.
    """
    score = 0

    if 80 <= brightness <= 180:
        score += 50

    if sharpness > 100:
        score += 50

    return score
def detect_faces(frame, face_cascade):
    """
    Detect faces in a frame using a Haar cascade.
    Returns a list of bounding boxes.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )
    return faces


def is_face_centered(face, frame_width, tolerance=0.15):
    """
    Check whether a detected face is roughly centered
    within the frame width.
    """
    x, y, w, h = face
    face_center_x = x + w / 2
    frame_center_x = frame_width / 2

    offset = abs(face_center_x - frame_center_x) / frame_center_x
    return offset < tolerance
