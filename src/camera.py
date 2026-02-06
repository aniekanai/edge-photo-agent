import cv2 #OpenCV library for computer vision

#Open the default camera (index 0 = first USB Camera)
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Camera not opened")
    exit()

while True:
    #Read one frame from the camera
    ret, frame = cap.read()

    # If frame not read correctly, stop
    if not ret:
        print("Failed to grab frame")
        break

    # Display the frame in a window (requires local display)
    cv2.imshow("Jetson Nano Camera Test", frame)

    #Wait 1 ms, exit loop if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera resource
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()

    