import cv2
import os
from datetime import datetime

from metrics import (
    compute_brightness,
    compute_sharpness,
    interpret_brightness,
    interpret_sharpness,
    compute_quality_score,
    detect_faces,
    is_face_centered
)

def run_camera():
    # --- Setup camera ---
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened")
        return

    # --- Load face detector ---
    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    # --- Setup capture output directory ---
    output_dir = os.path.join("..", "demo", "captures")
    os.makedirs(output_dir, exist_ok=True)

    print(f"[INFO] Captured images will be saved to: {output_dir}")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # --- Compute metrics ---
        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)

        brightness_label = interpret_brightness(brightness)
        sharpness_label = interpret_sharpness(sharpness)

        score = compute_quality_score(brightness, sharpness)

        # --- Face detection ---
        faces = detect_faces(frame, face_cascade)
        face_status = "No Face Detected"
        face_centered = False

        if len(faces) > 0:
            face = faces[0]
            x, y, w, h = face

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            face_centered = is_face_centered(face, frame.shape[1])
            face_status = "Centered" if face_centered else "Not Centered"

        # --- Photo readiness logic ---
        photo_ready = (
            score >= 85 and
            face_centered and
            brightness_label == "OK" and
            sharpness_label == "Sharp"
        )

        # --- Status text ---
        status_text = "Adjusting..."
        status_color = (0, 0, 255)

        if photo_ready:
            status_text = "PHOTO READY"
            status_color = (0, 255, 0)

        # --- Overlays ---
        cv2.putText(frame, f"Brightness: {brightness:.1f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"Sharpness: {sharpness:.1f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        cv2.putText(frame, f"Brightness Status: {brightness_label}",
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.putText(frame, f"Sharpness Status: {sharpness_label}",
                    (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.putText(frame, f"Quality Score: {score}/100",
                    (10, 160), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)

        cv2.putText(frame, f"Face Status: {face_status}",
                    (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

        cv2.putText(frame, status_text,
                    (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.9, status_color, 3)

        cv2.imshow("Edge Photo Intelligence Agent", frame)

        # --- Auto-capture ---
        if photo_ready:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_{timestamp}.jpg"
            filepath = os.path.join(output_dir, filename)

            cv2.imwrite(filepath, frame)
            print(f"[CAPTURED] {filepath}")
            break

        # Manual exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
