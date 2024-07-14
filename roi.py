import cv2

# Path to your MP4 video file
video_path = 'video.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)

# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

# Read the first frame of the video
ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame.")
    exit()

# Resize the frame to fit on the screen if necessary
scale_percent = 100  # Percent of original size, adjust if frame is too large
width = int(frame.shape[1] * scale_percent / 100)
height = int(frame.shape[0] * scale_percent / 100)
dim = (width, height)

# Resize image
resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)

# Display the frame and select ROI
print("Select the region of interest (ROI) and press Enter or Space.")
roi = cv2.selectROI("Frame", resized_frame, fromCenter=False, showCrosshair=True)
cv2.destroyAllWindows()

# Adjust ROI coordinates based on resizing
roi_x = int(roi[0] * 100 / scale_percent)
roi_y = int(roi[1] * 100 / scale_percent)
roi_w = int(roi[2] * 100 / scale_percent)
roi_h = int(roi[3] * 100 / scale_percent)
print(f"ROI selected: x={roi_x}, y={roi_y}, width={roi_w}, height={roi_h}")

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
