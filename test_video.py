import cv2
import time
from app import VideoStreamProcessor

processor = VideoStreamProcessor()
cap = cv2.VideoCapture(0)

print("Reading frame from camera...")
ret, frame = cap.read()
if ret:
    print(f"Frame shape: {frame.shape}")
    print("Sending to process_frame...")
    try:
        result = processor.process_frame(frame)
        print(f"Result success: {result['success']}")
        if not result['success']:
            if 'error' in result:
                print(f"Caught error: {result['error']}")
    except Exception as e:
        print(f"Caught top level exception: {e}")
else:
    print("Could not read frame")

cap.release()
