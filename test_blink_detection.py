#!/usr/bin/env python3
"""
Simple blink detection test script
Run this to test if your camera and blink detection are working properly
"""

import cv2
import numpy as np
import time
from face_detector import FaceDetector

def test_blink_detection():
    """Test blink detection with live camera feed"""
    
    # Initialize face detector
    print("Initializing face detector...")
    face_detector = FaceDetector()
    
    # Initialize camera
    print("Initializing camera...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    print("✅ Camera initialized successfully")
    print("Press 'q' to quit, 'b' to test blink detection")
    
    # Variables for blink tracking
    last_ear_values = []
    blink_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Detect faces and landmarks
            faces, gray = face_detector.detect_faces(frame)
            
            if len(faces) > 0:
                # Get landmarks for the first face
                landmarks = face_detector.get_landmarks(gray, faces[0])
                
                if landmarks is not None and len(landmarks) >= 48:
                    # Draw face boundary
                    x, y, w, h = faces[0].left(), faces[0].top(), faces[0].width(), faces[0].height()
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Get blink status and EAR
                    is_blinking, ear = face_detector.is_blinking(landmarks)
                    
                    if ear is not None:
                        # Add EAR to tracking
                        last_ear_values.append(ear)
                        if len(last_ear_values) > 10:
                            last_ear_values.pop(0)
                        
                        # Display EAR value
                        cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                        
                        # Display blink status
                        if is_blinking:
                            status_color = (0, 0, 255)  # Red for blinking
                            status_text = "BLINKING"
                        else:
                            status_color = (0, 255, 0)  # Green for eyes open
                            status_text = "EYES OPEN"
                        
                        cv2.putText(frame, f"Status: {status_text}", (10, 70), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                        
                        # Display recent EAR values
                        if len(last_ear_values) > 0:
                            recent_text = f"Recent: {[f'{v:.3f}' for v in last_ear_values[-5:]]}"
                            cv2.putText(frame, recent_text, (10, 110), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        
                        # Simple blink detection logic
                        if len(last_ear_values) >= 3:
                            current_ear = last_ear_values[-1]
                            previous_ear = last_ear_values[-2]
                            older_ear = last_ear_values[-3]
                            
                            # Check for blink pattern
                            if (older_ear > 0.21 and  # Eyes were open
                                previous_ear < 0.19 and  # Eyes closed
                                current_ear > 0.21):     # Eyes opened again
                                
                                blink_count += 1
                                print(f"✅ Blink #{blink_count} detected! Pattern: {older_ear:.3f} -> {previous_ear:.3f} -> {current_ear:.3f}")
                                
                                # Clear buffer to avoid multiple detections
                                last_ear_values.clear()
                        
                        # Display blink count
                        cv2.putText(frame, f"Blinks detected: {blink_count}", (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
                        
                        # Display thresholds
                        cv2.putText(frame, "Open: >0.21, Closed: <0.19", (10, 190), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
                        
                    else:
                        cv2.putText(frame, "EAR calculation failed", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No valid landmarks", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            # Display instructions
            cv2.putText(frame, "Press 'q' to quit, 'b' to test blink", (10, 450), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Blink Detection Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('b'):
                # Test blink detection manually
                print("Manual blink test triggered")
                if len(last_ear_values) > 0:
                    print(f"Current EAR: {last_ear_values[-1]:.3f}")
                    print(f"Recent values: {[f'{v:.3f}' for v in last_ear_values]}")
                else:
                    print("No EAR values available")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Test completed")

if __name__ == "__main__":
    test_blink_detection()
