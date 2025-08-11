#!/usr/bin/env python3
"""
Test script for liveness detection fixes
Run this to test the improved head turn detection and manual controls
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from camera_interface import CameraInterface
from face_detector import FaceDetector
from liveness_detector import LivenessDetector
from config_manager import ConfigManager

def main():
    print("=" * 60)
    print("LIVENESS DETECTION TEST - FIXES VERIFICATION")
    print("=" * 60)
    
    try:
        # Initialize components
        print("1. Initializing components...")
        config_manager = ConfigManager()
        face_detector = FaceDetector(config_manager)
        liveness_detector = LivenessDetector(face_detector)
        camera_interface = CameraInterface(face_detector, liveness_detector)
        
        print("2. Starting camera...")
        camera_interface.run()
        
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
    except Exception as e:
        print(f"Error during test: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("3. Cleaning up...")
        if 'camera_interface' in locals():
            camera_interface.stop()

if __name__ == "__main__":
    main()
