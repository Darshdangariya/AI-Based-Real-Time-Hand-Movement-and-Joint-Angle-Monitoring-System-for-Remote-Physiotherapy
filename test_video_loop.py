import cv2
import time
from app import VideoStreamProcessor

processor = VideoStreamProcessor()
cap = cv2.VideoCapture(0)

print("Starting loop...")
frame_count = 0
try:
    while frame_count < 10:
        ret, frame = cap.read()
        if not ret: break
        result = processor.process_frame(frame)
        print(f"Frame {frame_count} success: {result['success']}")
        if not result['success']:
            if 'error' in result: print(f"Error: {result['error']}")
        frame_count += 1
        time.sleep(0.1)
except Exception as e:
    print(f"Exception caught in loop: {e}")

cap.release()
print("Done")
