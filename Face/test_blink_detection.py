#!/usr/bin/env python3
"""
Blink Detection Test Script
Use this to test blink detection without running the full liveness system.
This will help identify any remaining issues with blink detection.
"""

import cv2
import numpy as np
import sys
import os
import time

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_detector import FaceDetector
from config_manager import ConfigManager

def main():
    print("Blink Detection Test")
    print("=" * 50)
    print("This script will test blink detection in isolation.")
    print("Blink your eyes to see if detection works without crashing.")
    print("Press 'Q' to quit, 'R' to reset.")
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
        
        print("Camera initialized successfully")
        print("Blink your eyes to test detection...")
        
        # Test variables
        test_count = 0
        successful_detections = 0
        errors = 0
        
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("Failed to capture frame")
                    break
                
                # Mirror the frame
                frame = cv2.flip(frame, 1)
                
                # Detect faces
                faces, gray = face_detector.detect_faces(frame)
                
                if len(faces) > 0:
                    try:
                        landmarks = face_detector.get_landmarks(gray, faces[0])
                        
                        # Test blink detection
                        is_blinking, ear = face_detector.is_blinking(landmarks)
                        
                        # Draw landmarks
                        frame = face_detector.draw_landmarks(frame, landmarks)
                        
                        # Display results
                        test_count += 1
                        
                        if ear is not None and ear > 0:
                            # Valid EAR detected
                            if is_blinking:
                                status_color = (0, 0, 255)  # Red for blinking
                                status_text = "BLINKING"
                                successful_detections += 1
                            else:
                                status_color = (0, 255, 0)  # Green for not blinking
                                status_text = "NOT BLINKING"
                            
                            # Display EAR value and status
                            cv2.putText(frame, f"EAR: {ear:.3f}", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                            cv2.putText(frame, f"Status: {status_text}", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                        else:
                            # Invalid EAR
                            cv2.putText(frame, "EAR: Invalid", (10, 30), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            cv2.putText(frame, "Status: Error", (10, 60), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            errors += 1
                        
                        # Display test statistics
                        cv2.putText(frame, f"Tests: {test_count}", (10, 90), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                        cv2.putText(frame, f"Success: {successful_detections}", (10, 120), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                        cv2.putText(frame, f"Errors: {errors}", (10, 150), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        
                        # Calculate success rate
                        if test_count > 0:
                            success_rate = (successful_detections / test_count) * 100
                            cv2.putText(frame, f"Success Rate: {success_rate:.1f}%", (10, 180), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                        
                    except Exception as e:
                        print(f"Error processing frame: {e}")
                        errors += 1
                        cv2.putText(frame, f"Error: {str(e)[:50]}", (10, 30), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:
                    cv2.putText(frame, "No face detected", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Display instructions
                cv2.putText(frame, "Blink to test detection", (10, frame.shape[0] - 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(frame, "Press Q to quit, R to reset", (10, frame.shape[0] - 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                cv2.imshow('Blink Detection Test', frame)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q') or key == ord('Q'):
                    break
                elif key == ord('r') or key == ord('R'):
                    # Reset test statistics
                    test_count = 0
                    successful_detections = 0
                    errors = 0
                    print("Test statistics reset")
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.03)  # ~30 FPS
                
            except Exception as e:
                print(f"Error in main loop: {e}")
                import traceback
                traceback.print_exc()
                errors += 1
                continue
        
        cap.release()
        cv2.destroyAllWindows()
        
        # Final statistics
        print(f"\nTest completed!")
        print(f"Total tests: {test_count}")
        print(f"Successful detections: {successful_detections}")
        print(f"Errors: {errors}")
        if test_count > 0:
            success_rate = (successful_detections / test_count) * 100
            print(f"Success rate: {success_rate:.1f}%")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()





