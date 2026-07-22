import picamera2
import time
import cv2

# Initialise the camera
camera = picamera2.Picamera2()
config = camera.create_preview_configuration(
    main={'format': 'RGB888', 'size': (640, 480)}
)
camera.configure(config)
camera.start()

# Allow the camera to warm up
time.sleep(0.1)

# Capture frames
while True:
    image = camera.capture_array("main")
    
    # Rotate 180 degrees
    image = cv2.rotate(image, cv2.ROTATE_180)
    
    # Show the frame
    cv2.imshow("Frame", image)
    
    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

