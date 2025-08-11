#!/usr/bin/env python3

import cv2
import sys
import os
from config_manager import ConfigManager
from face_detector import FaceDetector
from liveness_detector_debug import LivenessDetectorDebug

def test_debug_liveness():
    """Test the debug version of the liveness detector"""
    
    print("=== DEBUG LIVENESS DETECTOR TEST ===")
    print("This will help identify why right head turns aren't being detected")
    print("=" * 50)
    
    try:
        # Initialize components
        print("1. Initializing config manager...")
        config_manager = ConfigManager("config_debug.yaml")
        
        print("2. Initializing face detector...")
        face_detector = FaceDetector(config_manager)
        if not face_detector.initialize():
            print("❌ Face detector initialization failed")
            return False
        
        print("3. Initializing debug liveness detector...")
        liveness_detector = LivenessDetectorDebug(face_detector)
        
        print("4. Opening camera...")
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
        print("✅ All components initialized successfully!")
        print("\n=== TESTING HEAD POSE DETECTION ===")
        print("Instructions:")
        print("- Position your face in the center")
        print("- Follow the on-screen instructions")
        print("- Turn your head LEFT when prompted")
        print("- Turn your head RIGHT when prompted")
        print("- Watch the console for debug information")
        print("- Press 'Q' to quit")
        print("=" * 50)
        
        # Create window
        window_name = "Debug Liveness Test"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 800, 600)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Mirror the frame
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces, gray = face_detector.detect_faces(frame)
            landmarks_list = []
            
            for face in faces:
                landmarks = face_detector.get_landmarks(gray, face)
                if landmarks is not None:
                    landmarks_list.append(landmarks)
            
            # Update liveness detector
            state = liveness_detector.update(frame, faces, landmarks_list)
            
            # Draw UI elements
            frame = draw_debug_ui(frame, faces, landmarks_list, state, liveness_detector)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\nQuitting...")
                break
            elif key == ord('r'):
                print("\nResetting liveness detector...")
                liveness_detector.reset()
            elif key == ord('p'):
                print("\nProceeding from guidelines...")
                liveness_detector.proceed_from_guidelines()
            
            frame_count += 1
            
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if 'cap' in locals():
            cap.release()
        cv2.destroyAllWindows()
        if 'face_detector' in locals():
            face_detector.cleanup()
        print("\nTest completed")

def draw_debug_ui(frame, faces, landmarks_list, state, liveness_detector):
    """Draw debug UI elements on the frame"""
    
    # Get debug info
    debug_info = liveness_detector.get_debug_info()
    
    # Draw current state
    state_text = f"State: {debug_info['state']}"
    cv2.putText(frame, state_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Draw instruction
    instruction = liveness_detector.get_current_instruction()
    cv2.putText(frame, "Instruction:", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
    
    # Split long instructions into multiple lines
    words = instruction.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line + " " + word) < 50:
            current_line += (" " + word) if current_line else word
        else:
            lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)
    
    y_offset = 85
    for line in lines:
        cv2.putText(frame, line, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        y_offset += 20
    
    # Draw progress
    y_offset += 10
    cv2.putText(frame, "Progress:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    y_offset += 25
    
    for step, completed in debug_info['completed_steps'].items():
        status = "✅" if completed else "⏳"
        step_text = f"{status} {step.replace('_', ' ').title()}"
        cv2.putText(frame, step_text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        y_offset += 20
    
    # Draw current detection info
    y_offset += 10
    cv2.putText(frame, "Detection Info:", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)
    y_offset += 25
    
    cv2.putText(frame, f"Direction Frames: {debug_info['direction_frame_count']}/{debug_info['frames_required']}", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    y_offset += 20
    
    cv2.putText(frame, f"Threshold: ±{debug_info['threshold']}°", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    
    # Draw landmarks and face boundaries
    if len(faces) > 0 and len(landmarks_list) > 0:
        face = faces[0]
        landmarks = landmarks_list[0]
        
        # Draw landmarks
        frame = liveness_detector.face_detector.draw_landmarks(frame, landmarks)
        
        # Draw face boundary
        frame = liveness_detector.face_detector.draw_face_boundary(frame, face)
        
        # Draw face center guide
        h, w = frame.shape[:2]
        center_x, center_y = w // 2, h // 2
        cv2.circle(frame, (center_x, center_y), 5, (0, 255, 255), -1)
        cv2.circle(frame, (center_x, center_y), 100, (0, 255, 255), 2)
    
    # Draw controls
    y_offset = frame.shape[0] - 80
    cv2.putText(frame, "Controls: Q=Quit, R=Reset, P=Proceed", 
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    
    return frame

if __name__ == "__main__":
    test_debug_liveness()

