import cv2
import math
from ultralytics import YOLO
import cvzone

# --- CONFIGURATION ---
# Set to 0 for Webcam. 
# If you have a video file, replace 0 with the filename like 'crowd.mp4'
VIDEO_SOURCE = 0 

# ALERT SETTING: If more than this many people are seen, show RED ALERT
CROWD_THRESHOLD = 4  

# --- INITIALIZATION ---
# Load the YOLOv8 Nano model (It is fast and light for laptops)
# Note: It will automatically download 'yolov8n.pt' the first time you run this.
print("Loading SurgeGuard AI Model...")
model = YOLO('yolov8n.pt')

# Class names (YOLO detects many things, we only care about 'person' which is ID 0)
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

# Start Video Capture
cap = cv2.VideoCapture(VIDEO_SOURCE)
cap.set(3, 1280) # Set Width
cap.set(4, 720)  # Set Height

print("System Ready. Opening Camera...")

while True:
    success, img = cap.read()
    if not success:
        break

    # Run YOLOv8 detection on the current frame
    results = model(img, stream=True)
    
    person_count = 0

    # Process Detections
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Check Class ID (0 is Person)
            cls = int(box.cls[0])
            if classNames[cls] == "person":
                person_count += 1
                
                # Get Bounding Box Coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Dynamic Color Logic: Green = Safe, Red = Alert
                color = (0, 255, 0) # Green
                if person_count > CROWD_THRESHOLD:
                    color = (0, 0, 255) # Red
                
                # Draw Bounding Box
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                
                # Draw Label (Optional)
                cvzone.putTextRect(img, f'Person', (max(0, x1), max(35, y1)), 
                                   scale=1, thickness=1, colorR=color)

    # --- DASHBOARD UI (The Top Bar) ---
    # 1. Draw a black background bar at the top
    cv2.rectangle(img, (0, 0), (1280, 80), (0, 0, 0), -1) 
    
    # 2. Determine Status Text
    if person_count > CROWD_THRESHOLD:
        status_text = "CRITICAL: OVERCROWDED"
        status_color = (0, 0, 255) # Red Text
    else:
        status_text = "STATUS: SAFE"
        status_color = (0, 255, 0) # Green Text

    # 3. Put Text on Screen
    cv2.putText(img, f'Count: {person_count}', (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(img, status_text, (300, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 3)

    # Show the Frame
    cv2.imshow("SurgeGuard AI - Crowd Monitor", img)
    
    # Press 'q' key to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()