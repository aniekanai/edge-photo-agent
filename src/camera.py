import cv2
import time
import os
import threading

from metrics import (
    compute_brightness,
    compute_sharpness,
    detect_faces,
    is_face_centered
)

from agent import nemotron_refinement


def local_guidance(face_detected, face_centered, brightness, sharpness):
    """
    Instant CV-based guidance for photographer.
    """
    if not face_detected:
        return "No subject detected"

    if not face_centered:
        return "Subject not centered"

    if brightness < 90:
        return "Lighting too low"

    if brightness > 180:
        return "Lighting too strong"

    if sharpness < 400:  # your updated threshold
        return "Image unstable"

    return "Optimal Composition"


def draw_multiline_text(frame, text, y_start):
    """
    Draw centered multi-line wrapped text so nothing gets cut off.
    """
    words = text.split(" ")
    lines = []
    current_line = ""

    max_width = frame.shape[1] - 100  # padding from edges

    for word in words:
        test_line = current_line + " " + word if current_line else word
        text_size = cv2.getTextSize(
            test_line,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2
        )[0]

        if text_size[0] > max_width:
            lines.append(current_line)
            current_line = word
        else:
            current_line = test_line

    if current_line:
        lines.append(current_line)

    y = y_start
    for line in lines:
        text_size = cv2.getTextSize(
            line,
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            2
        )[0]
        x = (frame.shape[1] - text_size[0]) // 2

        cv2.putText(
            frame,
            line,
            (x, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 255, 255),
            2
        )
        y += 45


def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened")
        return

    # 🔥 TRUE FULLSCREEN MODE
    cv2.namedWindow("AI Photography Co-Pilot", cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        "AI Photography Co-Pilot",
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN
    )

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    captures_dir = os.path.join(os.path.dirname(__file__), "captures")
    os.makedirs(captures_dir, exist_ok=True)

    refinement_text = ""
    refinement_lock = threading.Lock()
    worker_running = False
    last_worker_start = 0
    nemotron_interval = 4.0  # seconds between Nemotron calls

    def worker(frame_copy, metrics_copy):
        nonlocal refinement_text, worker_running
        try:
            result = nemotron_refinement(frame_copy, metrics_copy)
            with refinement_lock:
                refinement_text = result
        finally:
            worker_running = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frame = frame.copy()

        faces = detect_faces(frame, face_cascade)
        face_detected = len(faces) > 0
        face_centered = False

        if face_detected:
            face_centered = is_face_centered(faces[0], frame.shape[1])

        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)

        guidance = local_guidance(
            face_detected,
            face_centered,
            brightness,
            sharpness
        )

        photo_ready = guidance == "Optimal Composition"

        now = time.time()

        # Run Nemotron in background
        if (
            face_detected
            and (now - last_worker_start > nemotron_interval)
            and not worker_running
        ):
            worker_running = True
            last_worker_start = now

            metrics_copy = {
                "brightness": brightness,
                "sharpness": sharpness,
                "face_detected": face_detected,
                "face_centered": face_centered,
            }

            t = threading.Thread(
                target=worker,
                args=(raw_frame.copy(), metrics_copy),
                daemon=True
            )
            t.start()

        # Merge refinement
        with refinement_lock:
            refine = refinement_text

        if refine:
            final_guidance = f"{guidance} | {refine}"
        else:
            final_guidance = guidance

        # Status indicator
        status_color = (0, 255, 0) if photo_ready else (0, 0, 255)
        status_text = (
            "Optimal Composition - Press SPACE"
            if photo_ready
            else "Adjusting"
        )

        cv2.putText(
            frame,
            status_text,
            (40, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            status_color,
            3
        )

        # 🔥 Wrapped guidance text (bottom center)
        draw_multiline_text(
            frame,
            final_guidance,
            frame.shape[0] - 120
        )

        if worker_running:
            cv2.putText(
                frame,
                "Nemotron analyzing...",
                (40, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2
            )

        cv2.imshow("AI Photography Co-Pilot", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(" "):  # SPACE to capture
            if photo_ready:
                filename = os.path.join(
                    captures_dir,
                    f"captured_{int(time.time())}.jpg"
                )
                cv2.imwrite(filename, raw_frame)
                print(f"[CAPTURED] {filename}")

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()