import cv2
import time
import os

from metrics import (
    compute_brightness,
    compute_sharpness,
    compute_quality_score,
    detect_faces,
    is_face_centered
)

from agent import nemotron_guidance


def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened")
        return

    cv2.namedWindow("Edge Photo Agent", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Edge Photo Agent", 1000, 700)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    captures_dir = os.path.join(os.path.dirname(__file__), "captures")
    os.makedirs(captures_dir, exist_ok=True)

    last_guidance = "Initializing..."
    last_nemotron_time = 0
    nemotron_interval = 3

    capture_sequence = False
    sequence_start = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frame = frame.copy()

        faces = detect_faces(frame, face_cascade)
        face_detected = len(faces) > 0
        face_centered = False

        if face_detected:
            x, y, w, h = faces[0]
            face_centered = is_face_centered(faces[0], frame.shape[1])

        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)
        score = compute_quality_score(brightness, sharpness)

        photo_ready = (
            face_detected and
            face_centered and
            80 <= brightness <= 180 and
            sharpness > 150
        )

        now = time.time()

        # Nemotron guidance (throttled)
        if (now - last_nemotron_time > nemotron_interval) and not capture_sequence:
            metrics = {
                "brightness": brightness,
                "sharpness": sharpness,
                "face_detected": face_detected,
                "face_centered": face_centered,
                "quality_score": score,
            }

            last_guidance = nemotron_guidance(frame, metrics)
            last_nemotron_time = now

        # Capture sequence logic
        if photo_ready and not capture_sequence:
            capture_sequence = True
            sequence_start = now

        if capture_sequence:
            if now - sequence_start < 1:
                last_guidance = "Look at camera"
            elif now - sequence_start < 2:
                last_guidance = "Smile"
            else:
                filename = os.path.join(
                    captures_dir,
                    f"captured_{int(now)}.jpg"
                )
                cv2.imwrite(filename, raw_frame)
                print(f"[CAPTURED] {filename}")
                capture_sequence = False

        # Minimal clean overlay
        status_text = "Ready" if photo_ready else "Adjusting"
        status_color = (0, 255, 0) if photo_ready else (0, 0, 255)

        cv2.putText(frame, status_text,
                    (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    status_color,
                    3)

        cv2.putText(frame, last_guidance,
                    (50, 650),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (0, 255, 255),
                    3)

        cv2.imshow("Edge Photo Agent", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
