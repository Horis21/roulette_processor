import time
import cv2
import pytesseract
import os
import numpy as np

# Ensure pytesseract is correctly installed on your system
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Update with your Tesseract path



# Function to extract text from a frame
def extract_text_from_frame(frame):
    # Convert the frame to grayscale
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Use pytesseract to extract text
    text = pytesseract.image_to_string(gray_frame)
    return text


def calculate_angle(p1, p2, p3):
    """
    Calculate the angle between three points.
    """
    a = np.array(p1)
    b = np.array(p2)
    c = np.array(p3)
    ab = b - a
    cb = b - c
    angle = np.arctan2(ab[1], ab[0]) - np.arctan2(cb[1], cb[0])
    return np.degrees(angle)

def detect_rotation_direction(points):
    """
    Determine if points are moving clockwise or counterclockwise.
    """
    angles = []
    for i in range(2, len(points)):
        angle = calculate_angle(points[i-2], points[i-1], points[i])
        angles.append(angle)
    
    # Sum of angles to determine overall direction
    total_angle = sum(angles)
    
    if total_angle > 0:
        return "counterclockwise"
    else:
        return "clockwise"

# Path to your MP4 video file
video_path = 'video.mp4'

# Open the video file
cap = cv2.VideoCapture(video_path)


# Check if the video file opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()


movement_history = []

frame_count = 0

# Define the ROI coordinates (top-left x, top-left y, width, height)
roi_x, roi_y, roi_w, roi_h = 622, 170, 657, 389  # Adjust these values as needed

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    print("Frame count: ", frame_count)

    # Extract text from the current frame
    text = extract_text_from_frame(frame)

     # Define the region of interest (ROI)
    roi = frame[roi_x:roi_x + roi_w, roi_y:roi_y + roi_h]

     # Convert the ROI to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    
    start_time = time.time()

    # Detect circles using HoughCircles
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=100,
                               param1=100, param2=30, minRadius=0, maxRadius=0)

    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Consider only the first detected circle
        i = circles[0, 0]

        # Draw the outer circle (adjust coordinates to original frame)
        cv2.circle(frame, (i[0] + roi_x, i[1] + roi_y), i[2], (0, 255, 0), 2)
        # Draw the center of the circle (adjust coordinates to original frame)
        cv2.circle(frame, (i[0] + roi_x, i[1] + roi_y), 2, (0, 0, 255), 3)
        
        # Track a point on the circumference of the circle
        point = (i[0] + i[2] + roi_x, i[1] + roi_y)
        movement_history.append(point)

        # Draw the tracked point
        cv2.circle(frame, point, 5, (255, 0, 0), -1)

    if(frame_count > 3):
        direction = detect_rotation_direction(movement_history)
        print(f"The object is spinning {direction}.")

    # Draw the ROI rectangle on the frame for visualization
    cv2.rectangle(frame, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), 2)

    # Show the frame
    cv2.imshow('Frame', frame)

    end_time = time.time()
    print(f"Time taken for circle detection in this frame: {end_time - start_time:.4f} seconds")

    #print(f"Frame {frame_count}: {text}")

    # Optional: Save or process the text as needed
    # For example, write the text to a file or database


# Analyze the movement path to determine rotation direction
if len(movement_history) >= 3:
    direction = detect_rotation_direction(movement_history)
    print(f"The object is spinning {direction}.")
else:
    print("Not enough data to determine the direction.")

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
