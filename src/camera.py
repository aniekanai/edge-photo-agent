import cv2
import time
from metrics import (
    compute_brightness,
    compute_sharpness,
    interpret_brightness,
    interpret_sharpness,
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

    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    last_capture_time = 0
    capture_cooldown = 3  # seconds between captures

    while True:

        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        faces = detect_faces(frame, face_cascade)

        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)

        brightness_label = interpret_brightness(brightness)
        sharpness_label = interpret_sharpness(sharpness)

        score = compute_quality_score(brightness, sharpness)

        face_detected = len(faces) > 0
        face_centered = False

        face_status = "No Face Detected"

        if face_detected:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            face_centered = is_face_centered(faces[0], frame.shape[1])
            face_status = "Centered" if face_centered else "Not Centered"

        photo_ready = (
            face_detected and
            face_centered and
            80 <= brightness <= 180 and
            sharpness > 150
        )

        # ===============================
        # AUTO CAPTURE
        # ===============================

        current_time = time.time()

        if photo_ready and (current_time - last_capture_time > capture_cooldown):
            filename = f"captures/captured_{int(current_time)}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Captured: {filename}")
            last_capture_time = current_time

        # ===============================
        # NEMOTRON GUIDANCE
        # ===============================

        metrics = {
            "brightness": brightness,
            "sharpness": sharpness,
            "face_detected": face_detected,
            "face_centered": face_centered,
            "quality_score": score
        }

        guidance_text = nemotron_guidance(frame, metrics)

        # ===============================
        # OVERLAYS
        # ===============================

        cv2.putText(frame, f"Brightness: {brightness_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Sharpness: {sharpness_label}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Face: {face_status}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"Score: {score}/100", (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.putText(frame, f"Guidance: {guidance_text}", (10, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        cv2.imshow("Edge Photo Intelligence Agent", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
