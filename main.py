#!/usr/bin/env python3
"""
Photo Quality Face Recognition with Liveness Detection

This application implements a comprehensive face recognition system with:
- Real-time face detection using dlib
- Liveness detection with head pose estimation and blink detection
- High-quality photo capture with image enhancement
- DeepFace integration for face analysis

Usage:
    python main.py

Controls:
    'q' - Quit application
    'r' - Restart liveness detection

Requirements:
    - Camera connected to the system
    - shape_predictor_68_face_landmarks.dat file in the same directory
      (Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
"""

import cv2
import sys
import os
import argparse
from face_detector import FaceDetector
from liveness_detector import LivenessDetector
from camera_interface import CameraInterface

def check_requirements():
    """Check if all required files and dependencies are available"""
    print("Checking requirements...")
    
    # Check for dlib landmarks file
    landmarks_file = "shape_predictor_68_face_landmarks.dat"
    if not os.path.exists(landmarks_file):
        #print(f"\nERROR: {landmarks_file} not found!")
        #print("Please download it from:")
        #print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        #print("Extract the .dat file to the current directory.")
        return False
    
    # Check camera availability
    test_cap = cv2.VideoCapture(0)
    if not test_cap.isOpened():
        #print("\nERROR: Camera not available!")
        #print("Please ensure a camera is connected and not being used by another application.")
        test_cap.release()
        return False
    test_cap.release()
    
    print("✓ All requirements met!")
    return True

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Face Recognition with Liveness Detection')
    parser.add_argument('--camera', type=int, default=0, 
                       help='Camera index (default: 0)')
    parser.add_argument('--no-check', action='store_true',
                       help='Skip requirements check')
    
    args = parser.parse_args()
    
    # Check requirements unless skipped
    if not args.no_check and not check_requirements():
        sys.exit(1)
    
    
    try:

        face_detector = FaceDetector()
        
        liveness_detector = LivenessDetector(face_detector)
        
        camera_interface = CameraInterface(face_detector, liveness_detector)
        
        print(f"Starting camera (index: {args.camera})...")
        
        # run the application
        camera_interface.run()
        
    except FileNotFoundError as e:
        print(f"\nERROR: Required file not found: {e}")
        sys.exit(1)
        
    except Exception as e:
        print(f"\nERROR: {e}")
        sys.exit(1)
        
    finally:
        print("\nApplication closed.")

if __name__ == "__main__":
    main()