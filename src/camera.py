import cv2
import time
import os
import threading

from metrics import (
    compute_brightness,
    compute_sharpness,
    compute_quality_score,
    detect_faces,
    is_face_centered
)

from agent import nemotron_refinement


def local_guidance(face_detected, face_centered, brightness, sharpness):
    """
    Instant, reliable guidance based on CV metrics.
    This runs every frame and never blocks.
    """
    if not face_detected:
        return "Point camera at face"

    if not face_centered:
        return "Move to center"

    if brightness < 90:
        return "Increase lighting"

    if brightness > 180:
        return "Reduce lighting"

    if sharpness < 500:
        return "Hold still"

    return "Ready"


def run_camera():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Camera not opened")
        return

    # Big, resizable window for demo
    cv2.namedWindow("Edge Photo Agent", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Edge Photo Agent", 1100, 750)

    face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

    # Save captures inside src/captures (stable)
    captures_dir = os.path.join(os.path.dirname(__file__), "captures")
    os.makedirs(captures_dir, exist_ok=True)

    # --- Nemotron background worker state ---
    refinement_text = ""        # last Nemotron refinement
    refinement_lock = threading.Lock()
    worker_running = False
    last_worker_start = 0.0
    nemotron_interval = 4.0     # seconds between Nemotron calls (tune: 3-6)

    # Capture sequence state
    capture_sequence = False
    sequence_start = 0.0
    capture_cooldown = 2.0
    last_capture_time = 0.0

    def worker(frame_copy, metrics_copy):
        nonlocal refinement_text, worker_running
        try:
            r = nemotron_refinement(frame_copy, metrics_copy)  # may take time
            with refinement_lock:
                refinement_text = r
        finally:
            worker_running = False

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        raw_frame = frame.copy()

        # --- CV metrics (fast) ---
        faces = detect_faces(frame, face_cascade)
        face_detected = len(faces) > 0
        face_centered = False

        if face_detected:
            # Use first face
            x, y, w, h = faces[0]
            face_centered = is_face_centered(faces[0], frame.shape[1])

        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)
        score = compute_quality_score(brightness, sharpness)

        # Photo readiness (with your preferred sharpness threshold)
        photo_ready = (
            face_detected and
            face_centered and
            90 <= brightness <= 180 and
            sharpness >= 500
        )

        now = time.time()

        # --- Start Nemotron worker occasionally (never blocks) ---
        # Only run Nemotron when a face is present OR near-ready.
        should_spawn = (
            (now - last_worker_start) >= nemotron_interval and
            (face_detected or score >= 70) and
            not worker_running and
            not capture_sequence
        )

        if should_spawn:
            worker_running = True
            last_worker_start = now

            metrics_copy = {
                "brightness": brightness,
                "sharpness": sharpness,
                "face_detected": face_detected,
                "face_centered": face_centered,
                "quality_score": score,
            }
            frame_copy = raw_frame.copy()

            t = threading.Thread(target=worker, args=(frame_copy, metrics_copy), daemon=True)
            t.start()

        # --- Local guidance (instant) ---
        instant = local_guidance(face_detected, face_centered, brightness, sharpness)

        # --- Combine guidance: local first, Nemotron refines ---
        with refinement_lock:
            refine = refinement_text

        # Only show refinement if it’s relevant (non-empty) and we're not already "Ready"
        if instant != "Ready" and refine:
            guidance = f"{instant} | {refine}"
        else:
            guidance = instant

        # --- Capture sequence: Look at camera -> Smile -> Capture (smooth) ---
        if photo_ready and (now - last_capture_time) >= capture_cooldown and not capture_sequence:
            capture_sequence = True
            sequence_start = now

        if capture_sequence:
            elapsed = now - sequence_start
            if elapsed < 1.0:
                guidance = "Look at camera"
            elif elapsed < 2.0:
                guidance = "Smile"
            else:
                filename = os.path.join(captures_dir, f"captured_{int(now)}.jpg")
                cv2.imwrite(filename, raw_frame)  # save clean image (no overlays)
                print(f"[CAPTURED] {filename}")
                last_capture_time = now
                capture_sequence = False

        # --- UI overlay (clean + readable) ---
        status_text = "Ready" if photo_ready else "Adjusting"
        status_color = (0, 255, 0) if photo_ready else (0, 0, 255)

        cv2.putText(frame, status_text, (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, status_color, 3)

        # Center guidance along bottom
        text_size = cv2.getTextSize(guidance, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 3)[0]
        text_x = max(20, (frame.shape[1] - text_size[0]) // 2)
        text_y = frame.shape[0] - 50
        cv2.putText(frame, guidance, (text_x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 3)

        # Small indicator if background Nemotron is running
        if worker_running:
            cv2.putText(frame, "Nemotron: thinking...", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        cv2.imshow("Edge Photo Agent", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()