import cv2
import os
import time
from metrics import (
    compute_brightness,
    compute_sharpness,
    detect_faces,
    is_face_centered,
)
from agent import local_guidance, nemotron_refinement


def run_camera():
    window_name = "AI Photography Co-Pilot"

    cap = cv2.VideoCapture(0)

    # Moderate resolution (no extra optimization)
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

    print("AI Photography Co-Pilot Running...")

    # Nemotron throttling
    last_nemotron_time = 0
    nemotron_interval = 2.0
    cached_refinement = ""

    # Face detection throttling
    frame_count = 0
    face_detected = False
    face_centered = False

    optimal = False
    nemotron_status = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        raw_frame = frame.copy()

        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)

        # --------------------------------
        # Face detection every 5 frames
        # --------------------------------
        frame_count += 1

        if frame_count % 5 == 0:
            faces = detect_faces(frame, face_cascade)
            face_detected = len(faces) > 0

            if face_detected:
                face_centered = is_face_centered(
                    faces[0], frame.shape[1]
                )
            else:
                face_centered = False

        metrics = {
            "brightness": brightness,
            "sharpness": sharpness,
            "face_detected": face_detected,
            "face_centered": face_centered,
        }

        # --------------------------------
        # Local Guidance (fast)
        # --------------------------------
        primary_instruction = local_guidance(metrics)

        # --------------------------------
        # Nemotron Refinement (throttled)
        # --------------------------------
        if primary_instruction == "READY":
            current_time = time.time()

            if current_time - last_nemotron_time > nemotron_interval:
                nemotron_status = "Nemotron analyzing..."
                cached_refinement = nemotron_refinement(frame, metrics)
                last_nemotron_time = current_time

            refinement = cached_refinement
        else:
            cached_refinement = ""
            nemotron_status = ""
            refinement = ""

        # --------------------------------
        # Final Instruction
        # --------------------------------
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

        font = cv2.FONT_HERSHEY_SIMPLEX

        # --------------------------------
        # Top Left Status
        # --------------------------------
        if optimal:
            cv2.putText(
                frame,
                "Optimal Composition",
                (30, 50),
                font,
                0.9,
                (0, 255, 0),
                2,
            )
        else:
            cv2.putText(
                frame,
                "Adjusting",
                (30, 50),
                font,
                0.9,
                (0, 165, 255),
                2,
            )

        if nemotron_status:
            cv2.putText(
                frame,
                nemotron_status,
                (30, 85),
                font,
                0.7,
                (255, 255, 255),
                2,
            )

        # --------------------------------
        # Bottom Instruction
        # --------------------------------
        if optimal:
            instruction_text = "Press SPACE when ready"
        else:
            instruction_text = final_instruction

        text_size = cv2.getTextSize(
            instruction_text, font, 0.9, 2
        )[0]

        text_x = (frame.shape[1] - text_size[0]) // 2
        text_y = frame.shape[0] - 40

        cv2.putText(
            frame,
            instruction_text,
            (text_x, text_y),
            font,
            0.9,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # Manual Capture (clean frame)
        if key == 32 and optimal:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(capture_dir, filename)
            cv2.imwrite(filepath, raw_frame)
            print(f"Captured: {filename}")

    cap.release()
    cv2.destroyAllWindows()