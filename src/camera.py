import cv2
import time
import os

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

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Make sure captures folder exists (relative to where you run)
    captures_dir = "captures"
    os.makedirs(captures_dir, exist_ok=True)

    # --- Nemotron throttling settings ---
    last_nemotron_time = 0.0
    nemotron_interval = 3.0  # seconds (increase if still slow)
    last_guidance = "Initializing guidance..."

    # Optional: only request Nemotron when close to capture
    nemotron_min_score = 70

    # Auto-capture cooldown (optional if you’re saving multiple)
    last_capture_time = 0.0
    capture_cooldown = 3.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # --- Face detection ---
        faces = detect_faces(frame, face_cascade)
        face_detected = len(faces) > 0
        face_centered = False
        face_status = "No Face Detected"

        if face_detected:
            x, y, w, h = faces[0]
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            face_centered = is_face_centered(faces[0], frame.shape[1])
            face_status = "Centered" if face_centered else "Not Centered"

        # --- Metrics ---
        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)
        brightness_label = interpret_brightness(brightness)
        sharpness_label = interpret_sharpness(sharpness)
        score = compute_quality_score(brightness, sharpness)

        # --- Photo readiness (keep your current logic) ---
        photo_ready = (
            face_detected and
            face_centered and
            (80 <= brightness <= 180) and
            (sharpness > 150)
        )

        # --- Auto-capture ---
        now = time.time()
        if photo_ready and (now - last_capture_time) > capture_cooldown:
            filename = os.path.join(captures_dir, f"captured_{int(now)}.jpg")
            cv2.imwrite(filename, frame)
            print(f"[CAPTURED] {filename}")
            last_capture_time = now

        # --- Nemotron guidance (THROTTLED + EVENT-DRIVEN) ---
        should_call_nemotron = (
            (now - last_nemotron_time) >= nemotron_interval and
            (score >= nemotron_min_score or photo_ready) and
            face_detected  # only if something to reason about
        )

        if should_call_nemotron:
            metrics = {
                "brightness": brightness,
                "sharpness": sharpness,
                "face_detected": face_detected,
                "face_centered": face_centered,
                "quality_score": score,
            }
            # This may take time, but it happens only once per interval
            last_guidance = nemotron_guidance(frame, metrics)
            last_nemotron_time = now

        # --- Overlays (WITH NUMBERS BACK) ---
        cv2.putText(frame, f"Brightness: {brightness:.1f} ({brightness_label})",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Sharpness: {sharpness:.1f} ({sharpness_label})",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(frame, f"Face: {face_status}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.putText(frame, f"Score: {score}/100",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        status_text = "PHOTO READY" if photo_ready else "Adjusting..."
        status_color = (0, 255, 0) if photo_ready else (0, 0, 255)
        cv2.putText(frame, status_text,
                    (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        # Show last guidance immediately (no waiting)
        cv2.putText(frame, f"Guidance: {last_guidance}",
                    (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        cv2.imshow("Edge Photo Intelligence Agent", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
