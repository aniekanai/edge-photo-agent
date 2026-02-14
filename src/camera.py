import cv2
import os
import time
import subprocess
from metrics import (
    compute_brightness,
    compute_sharpness,
    detect_faces,
    is_face_centered,
)
from agent import  nemotron_refinement


def run_camera():
    window_name = "AI Photography Co-Pilot"

    cap = cv2.VideoCapture(0)

    # Set camera resolution (if supported)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    if not cap.isOpened():
        print("Camera not opened")
        return

    # Create fullscreen window
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(
        window_name,
        cv2.WND_PROP_FULLSCREEN,
        cv2.WINDOW_FULLSCREEN,
    )

    # Get screen resolution safely (Jetson compatible)
    try:
        output = subprocess.check_output(
            "xrandr | grep '*'", shell=True
        ).decode()
        resolution = output.split()[0]
        screen_width, screen_height = map(int, resolution.split("x"))
    except:
        screen_width, screen_height = 1920, 1080

    face_cascade = cv2.CascadeClassifier(
        "haarcascade_frontalface_default.xml"
    )

    last_capture_time = 0
    capture_cooldown = 3  # seconds

    capture_dir = os.path.join("captures")
    os.makedirs(capture_dir, exist_ok=True)

    print("AI Photography Co-Pilot Running...")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # ----------------------------
        # Compute CV Metrics
        # ----------------------------
        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)

        faces = detect_faces(frame, face_cascade)
        face_detected = len(faces) > 0

        face_centered = False

        if face_detected:
            x, y, w, h = faces[0]
            face_centered = is_face_centered(
                faces[0], frame.shape[1]
            )
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2,
            )

        metrics = {
            "brightness": brightness,
            "sharpness": sharpness,
            "face_detected": face_detected,
            "face_centered": face_centered,
        }

        # ----------------------------
        # Local CV Guidance (FAST)
        # ----------------------------
        primary_instruction = local_guidance(metrics)

        # ----------------------------
        # Nemotron Refinement (SMART)
        # ----------------------------
        refinement = ""

        if primary_instruction == "READY":
            refinement = nemotron_refinement(frame, metrics)

        # ----------------------------
        # Final Instruction Logic
        # ----------------------------
        if primary_instruction != "READY":
            final_instruction = primary_instruction
            status_color = (0, 165, 255)  # Orange
        else:
            if refinement:
                final_instruction = refinement
                status_color = (0, 165, 255)
            else:
                final_instruction = "READY - HOLD STILL"
                status_color = (0, 255, 0)

        # ----------------------------
        # Auto Capture
        # ----------------------------
        current_time = time.time()

        if (
            primary_instruction == "READY"
            and not refinement
            and current_time - last_capture_time > capture_cooldown
            and sharpness >= 400
        ):
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            filepath = os.path.join(capture_dir, filename)
            cv2.imwrite(filepath, frame)
            print(f"Captured: {filename}")
            last_capture_time = current_time

        # ----------------------------
        # Resize to Fullscreen
        # ----------------------------
        frame = cv2.resize(frame, (screen_width, screen_height))

        # ----------------------------
        # Overlay UI
        # ----------------------------
        font = cv2.FONT_HERSHEY_SIMPLEX

        # Status Top Left
        cv2.putText(
            frame,
            "AI Photography Co-Pilot",
            (50, 70),
            font,
            1.2,
            (255, 255, 255),
            3,
        )

        # Main Instruction Center
        text_size = cv2.getTextSize(
            final_instruction, font, 1.2, 3
        )[0]

        text_x = (screen_width - text_size[0]) // 2
        text_y = screen_height - 120

        cv2.putText(
            frame,
            final_instruction,
            (text_x, text_y),
            font,
            1.2,
            status_color,
            3,
        )

        cv2.imshow(window_name, frame)

        # Exit
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()