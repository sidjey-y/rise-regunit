#!/usr/bin/env python3

import cv2
import numpy as np
from face_detector import FaceDetector
from config_manager import ConfigManager

def debug_head_pose():
    """Debug script to test head pose detection"""
    
    print("Initializing face detector...")
    config_manager = ConfigManager("config.yaml")
    face_detector = FaceDetector(config_manager)
    
    if not face_detector.initialize():
        print("Failed to initialize face detector")
        return
    
    print("Opening camera...")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Failed to open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    print("\n=== HEAD POSE DEBUG MODE ===")
    print("Press 'Q' to quit")
    print("Turn your head left/right to see detection values")
    print("=" * 40)
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces, gray = face_detector.detect_faces(frame)
            
            if len(faces) > 0:
                face = faces[0]
                landmarks = face_detector.get_landmarks(gray, face)
                
                if landmarks is not None:
                    # Get face direction
                    direction = face_detector.get_face_direction(landmarks)
                    
                    # Get head pose angles
                    pitch, yaw, roll = face_detector.get_head_pose(landmarks, frame.shape)
                    
                    # Get detailed head pose analysis
                    head_pose_analysis = face_detector._analyze_head_pose(landmarks, frame.shape)
                    
                    # Display information on frame
                    info_text = [
                        f"Direction: {direction}",
                        f"Pitch: {pitch:.1f}°" if pitch is not None else "Pitch: None",
                        f"Yaw: {yaw:.1f}°" if yaw is not None else "Yaw: None", 
                        f"Roll: {roll:.1f}°" if roll is not None else "Roll: None",
                        f"Threshold: ±15°",
                        f"Looking Right: {'YES' if yaw and yaw > 15 else 'NO' if yaw else 'UNKNOWN'}",
                        f"Looking Left: {'YES' if yaw and yaw < -15 else 'NO' if yaw else 'UNKNOWN'}"
                    ]
                    
                    # Draw landmarks
                    frame = face_detector.draw_landmarks(frame, landmarks)
                    
                    # Draw face boundary
                    frame = face_detector.draw_face_boundary(frame, face)
                    
                    # Draw text
                    y_offset = 30
                    for text in info_text:
                        cv2.putText(frame, text, (10, y_offset), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                        y_offset += 25
                    
                    # Print to console for debugging
                    print(f"\rDirection: {direction}, Yaw: {yaw:.1f}°, Threshold: ±15°", end="")
                    
                    # Check if thresholds are met
                    if yaw is not None:
                        if yaw > 15:
                            print(f" ✅ RIGHT TURN DETECTED (Yaw: {yaw:.1f}°)")
                        elif yaw < -15:
                            print(f" ✅ LEFT TURN DETECTED (Yaw: {yaw:.1f}°)")
                        else:
                            print(f" ⚠️  Within center range (Yaw: {yaw:.1f}°)")
            
            cv2.imshow("Head Pose Debug", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nStopped by user")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        face_detector.cleanup()
        print("\nDebug session ended")

if __name__ == "__main__":
    debug_head_pose()

