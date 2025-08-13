#!/usr/bin/env python3
"""
Test script to verify main.py can be imported and initialized
"""

import sys
import os

# Add the Face directory to the path
sys.path.insert(0, 'Face')

def test_imports():
    """Test if all required modules can be imported"""
    try:
        print("Testing imports...")
        
        # Test basic imports
        import cv2
        print("✅ OpenCV imported successfully")
        
        import dlib
        print("✅ dlib imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        # Test our modules - import directly since we're in the Face directory
        from face_detector import FaceDetector
        print("✅ FaceDetector imported successfully")
        
        from liveness_detector import LivenessDetector
        print("✅ LivenessDetector imported successfully")
        
        from camera_interface import CameraInterface
        print("✅ CameraInterface imported successfully")
        
        print("\n✅ All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def test_face_detector():
    """Test FaceDetector initialization"""
    try:
        print("\nTesting FaceDetector...")
        
        # Check if landmarks file exists
        landmarks_file = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(landmarks_file):
            print(f"❌ Landmarks file not found: {landmarks_file}")
            print("Please download the shape_predictor_68_face_landmarks.dat file")
            return False
        
        print("✅ Landmarks file found")
        
        # Try to initialize FaceDetector
        from face_detector import FaceDetector
        detector = FaceDetector()
        print("✅ FaceDetector initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ FaceDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_liveness_detector():
    """Test LivenessDetector initialization"""
    try:
        print("\nTesting LivenessDetector...")
        
        from face_detector import FaceDetector
        from liveness_detector import LivenessDetector
        
        detector = FaceDetector()
        liveness = LivenessDetector(detector)
        print("✅ LivenessDetector initialized successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ LivenessDetector test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("="*60)
    print("TESTING MAIN.PY COMPONENTS")
    print("="*60)
    
    # Test imports
    if not test_imports():
        print("\n❌ Import tests failed. Please check your dependencies.")
        return False
    
    # Test FaceDetector
    if not test_face_detector():
        print("\n❌ FaceDetector test failed.")
        return False
    
    # Test LivenessDetector
    if not test_liveness_detector():
        print("\n❌ LivenessDetector test failed.")
        return False
    
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("✅ Your main.py should work correctly now.")
    print("="*60)
    
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
