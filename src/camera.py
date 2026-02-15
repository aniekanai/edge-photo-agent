import cv2
import os
import time
from metrics import (
    compute_brightness,
    compute_sharpness,
    detect_faces,
    is_face_centered,
)
from agent import local_guidance, start_nemotron_background, get_latest_refinement


def run_camera():
    window_name = "AI Photography Co-Pilot"

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    if not cap.isOpened():
        print("Camera not opened")
        return

    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1200, 700)

    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    capture_dir = "captures"
    os.makedirs(capture_dir, exist_ok=True)

    last_nemotron_time = 0
    nemotron_interval = 3.0

    frame_count = 0
    face_detected = False
    face_centered = False

    print("AI Photography Co-Pilot Running...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frame = frame.copy()

        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)

        # Run face detection every 5 frames (performance)
        frame_count += 1
        if frame_count % 5 == 0:
            faces = detect_faces(frame, face_cascade)
            face_detected = len(faces) > 0

            if face_detected:
                face_centered = is_face_centered(faces[0], frame.shape[1])
            else:
                face_centered = False

        metrics = {
            "brightness": brightness,
            "sharpness": sharpness,
            "face_detected": face_detected,
            "face_centered": face_centered,
        }

        primary_instruction = local_guidance(metrics)

        # Background Nemotron (NON BLOCKING)
        if primary_instruction == "READY":
            current_time = time.time()
            if current_time - last_nemotron_time > nemotron_interval:
                start_nemotron_background(raw_frame, metrics)
                last_nemotron_time = current_time

            refinement = get_latest_refinement()
        else:
            refinement = ""

        if primary_instruction != "READY":
            final_instruction = primary_instruction
            optimal = False
        else:
            if refinement:
                final_instruction = refinement
                optimal = False
            else:
                final_instruction = "Optimal Composition"
                optimal = True

        # ---------- UI ----------
        font = cv2.FONT_HERSHEY_SIMPLEX

        if optimal:
            cv2.putText(frame, "Optimal Composition", (30, 50), font, 0.9, (0, 255, 0), 2)
        else:
            cv2.putText(frame, "Adjusting", (30, 50), font, 0.9, (0, 165, 255), 2)

        instruction = "Press SPACE when ready" if optimal else final_instruction

        text_size = cv2.getTextSize(instruction, font, 0.9, 2)[0]
        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 40

        cv2.putText(frame, instruction, (text_x, text_y), font, 0.9, (255, 255, 255), 2)

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        if key == 32 and optimal:
            filename = f"capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(os.path.join(capture_dir, filename), raw_frame)
            print("Captured:", filename)

    cap.release()
    cv2.destroyAllWindows()