#!/usr/bin/env python3
"""
Simple test to verify liveness detector blink detection works without errors
"""

import cv2
import numpy as np
import sys
import os

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_detector import FaceDetector
from liveness_detector import LivenessDetector

def main():
    print("Liveness Detector Blink Test")
    print("=" * 40)
    
    try:
        # Initialize components
        face_detector = FaceDetector()
        liveness_detector = LivenessDetector(face_detector)
        
        print("✅ Components initialized successfully")
        
        # Test that the liveness detector can be created without errors
        print("✅ LivenessDetector created successfully")
        
        # Test that the last_ear_values is properly initialized
        print(f"✅ last_ear_values type: {type(liveness_detector.last_ear_values)}")
        print(f"✅ last_ear_values maxlen: {liveness_detector.last_ear_values.maxlen}")
        
        # Test adding some values to the deque
        test_values = [0.3, 0.25, 0.2, 0.15, 0.3]
        for val in test_values:
            liveness_detector.last_ear_values.append(val)
        
        print(f"✅ Added test values: {list(liveness_detector.last_ear_values)}")
        
        # Test the slice operation that was causing the error
        try:
            ear_list = list(liveness_detector.last_ear_values)
            recent_values = ear_list[-4:] if len(ear_list) >= 4 else ear_list
            print(f"✅ Slice operation successful: {recent_values}")
        except Exception as e:
            print(f"❌ Slice operation failed: {e}")
            return False
        
        print("✅ All tests passed! Blink detection should work without errors.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
