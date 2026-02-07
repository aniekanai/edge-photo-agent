import cv2
from metrics import (
    compute_brightness,
    compute_sharpness,
    interpret_brightness,
    interpret_sharpness,
    compute_quality_score
)

def run_camera():
    """
    Main camera loop for the Edge Photo Intelligence Agent.

    Captures live frames from the Jetson Nano camera,
    computes visual quality metrics, overlays results,
    and displays real-time feedback.
    """
    cap = cv2.VideoCapture(0)

    # Ensure camera opens correctly
    if not cap.isOpened():
        print("Camera not opened")
        return

    while True:
        # Capture a single frame
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # --- Compute visual metrics ---
        brightness = compute_brightness(frame)
        sharpness = compute_sharpness(frame)

        brightness_label = interpret_brightness(brightness)
        sharpness_label = interpret_sharpness(sharpness)

        score = compute_quality_score(brightness, sharpness)

        # --- Overlay numeric metrics ---
        cv2.putText(
            frame,
            "Brightness: {:.1f}".format(brightness),
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Sharpness: {:.1f}".format(sharpness),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        # --- Overlay interpreted feedback ---
        cv2.putText(
            frame,
            "Brightness Status: {}".format(brightness_label),
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        cv2.putText(
            frame,
            "Sharpness Status: {}".format(sharpness_label),
            (10, 120),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        # --- Overlay quality score ---
        cv2.putText(
            frame,
            "Quality Score: {}/100".format(score),
            (10, 160),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        # Display the annotated camera feed
        cv2.imshow("Edge Photo Intelligence Agent", frame)

        # Exit when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up resources
    cap.release()
    cv2.destroyAllWindows()
