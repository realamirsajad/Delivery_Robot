import cv2
import numpy as np

# --- CONFIGURATION ---
CHECKERBOARD = (6, 9) # Number of inner corners (width, height)
SQUARE_SIZE = 0.025   # Physical size of one square in meters (e.g., 25mm = 0.025)

# Termination criteria for refining corner coordinates
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare 3D object points based on the real-world measurements
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
objp = objp * SQUARE_SIZE

# Arrays to store object points and image points
objpoints = [] # 3D points in real world space
imgpoints = [] # 2D points in image plane
image_size = None

# Start webcam capture
cap = cv2.VideoCapture(0)

print("--- CAMERA CALIBRATION ---")
print("Press 'c' to capture a frame when the checkerboard is in view.")
print("Press 'q' to finish capturing and compute the calibration.")
print("Aim for at least 15-20 good captures from different angles and distances.\n")

captured_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame.")
        break

    # Create a copy for the display so we don't draw on the raw frame
    display_frame = frame.copy()
    
    # Add a UI overlay
    cv2.putText(display_frame, f"Captured: {captured_count} (Need 15+)", 
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(display_frame, "Press 'c' to capture, 'q' to finish", 
                (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    cv2.imshow('Live Calibration Feed', display_frame)

    key = cv2.waitKey(1) & 0xFF

    # If 'c' is pressed, try to find the checkerboard
    if key == ord('c'):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Save image size for the calibration function later
        if image_size is None:
            image_size = gray.shape[::-1]

        # Find the chess board corners
        ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

        if ret_corners:
            # Refine the corners for higher accuracy
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            
            objpoints.append(objp)
            imgpoints.append(corners2)
            captured_count += 1
            
            print(f"[{captured_count}] Checkerboard successfully detected and captured!")

            # Draw the corners on the display frame to give visual confirmation
            cv2.drawChessboardCorners(display_frame, CHECKERBOARD, corners2, ret_corners)
            cv2.imshow('Live Calibration Feed', display_frame)
            cv2.waitKey(500) # Freeze the frame for 0.5 seconds so you can see it worked
        else:
            print("❌ Checkerboard NOT fully visible. Make sure the whole board is in frame, well-lit, and flat.")

    # If 'q' is pressed, break the loop and start the math
    elif key == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()

# --- COMPUTE CALIBRATION ---
print("\n--- CAPTURE COMPLETE ---")
if captured_count < 10:
    print("Warning: You captured fewer than 10 images. Calibration might be inaccurate.")

if captured_count > 0:
    print("Calculating calibration matrix... This may take a moment.")
    
    # Perform the actual camera calibration
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None, None)

    # Save the resulting data
    np.savez('calibration_data.npz', mtx=mtx, dist=dist)

    print("\n✅ Calibration successful!")
    print("Camera Matrix:\n", mtx)
    print("Distortion Coefficients:\n", dist)
    print("Saved to 'calibration_data.npz'. You can now run the ArUco script.")
else:
    print("No valid images captured. Calibration cancelled.")