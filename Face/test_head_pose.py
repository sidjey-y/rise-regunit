#!/usr/bin/env python3
"""
Head Pose Detection Test Script
Use this to test and calibrate head pose detection for liveness verification.
"""

import cv2
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_detector import FaceDetector
from config_manager import ConfigManager

def main():
    print("Head Pose Detection Test")
    print("=" * 50)
    print("This script will help you test head pose detection.")
    print("Turn your head left and right to see the yaw values.")
    print("Press 'Q' to quit, 'R' to reset thresholds.")
    print("=" * 50)
    
    try:
        # Initialize components
        config_manager = ConfigManager("config.yaml")
        face_detector = FaceDetector(config_manager)
        
        if not face_detector.initialize():
            print("Failed to initialize face detector")
            return
        
        # Get camera
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Failed to open camera")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get threshold from config
        threshold = config_manager.get('liveness.head_movement_threshold', 8.0)
        print(f"Current threshold: {threshold}°")
        print("Turn your head left (should show negative yaw) and right (should show positive yaw)")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces, gray = face_detector.detect_faces(frame)
            
            if len(faces) > 0:
                landmarks = face_detector.get_landmarks(gray, faces[0])
                
                # Get head pose
                pitch, yaw, roll = face_detector.get_head_pose(landmarks, frame.shape)
                
                # Draw landmarks
                frame = face_detector.draw_landmarks(frame, landmarks)
                
                # Display head pose values
                if yaw is not None:
                    # Color code based on direction
                    if yaw < -threshold:
                        color = (0, 0, 255)  # Red for left
                        direction = "LEFT"
                    elif yaw > threshold:
                        color = (255, 0, 0)  # Blue for right
                        direction = "RIGHT"
                    else:
                        color = (0, 255, 0)  # Green for center
                        direction = "CENTER"
                    
                    # Display values
                    cv2.putText(frame, f"Yaw: {yaw:.2f}°", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    cv2.putText(frame, f"Pitch: {pitch:.2f}°", (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(frame, f"Roll: {roll:.2f}°", (10, 110), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
                    cv2.putText(frame, f"Direction: {direction}", (10, 150), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
                    cv2.putText(frame, f"Threshold: ±{threshold}°", (10, 190), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                    
                    # Draw threshold lines
                    h, w = frame.shape[:2]
                    center_x = w // 2
                    center_y = h // 2
                    
                    # Draw center line
                    cv2.line(frame, (center_x, 0), (center_x, h), (0, 255, 0), 2)
                    
                    # Draw threshold indicators
                    if abs(yaw) > threshold:
                        # Draw success indicator
                        cv2.circle(frame, (center_x, center_y), 50, color, 3)
                        cv2.putText(frame, "THRESHOLD REACHED!", (center_x - 150, center_y + 80), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                else:
                    cv2.putText(frame, "No head pose data", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Turn head LEFT/RIGHT to test detection", (10, frame.shape[0] - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press Q to quit, R to reset", (10, frame.shape[0] - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Head Pose Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q'):
                break
            elif key == ord('r') or key == ord('R'):
                # Reset threshold
                threshold = 8.0
                print(f"Reset threshold to: {threshold}°")
        
        cap.release()
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()





