#!/usr/bin/env python3
"""
Simple test script to verify exit functionality works properly.
This will help test if the camera can be quit correctly.
"""

import cv2
import time
import sys
import os

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from config_manager import ConfigManager
from face_detector import FaceDetector
from liveness_detector import LivenessDetector
from camera_interface import CameraInterface

def test_exit_functionality():
    """Test the exit functionality of the camera interface"""
    
    print("Testing camera exit functionality...")
    print("This will open the camera for 10 seconds or until you press 'Q'")
    print("Press 'Q' to quit, or wait for auto-exit")
    
    try:
        # Initialize components
        config_manager = ConfigManager("config.yaml")
        face_detector = FaceDetector(config_manager)
        liveness_detector = LivenessDetector(face_detector)
        camera_interface = CameraInterface(face_detector, liveness_detector)
        
        # Start camera in a separate thread or process for testing
        import threading
        
        def run_camera():
            try:
                camera_interface.run(camera_index=0)
            except Exception as e:
                print(f"Camera error: {e}")
        
        # Start camera in background
        camera_thread = threading.Thread(target=run_camera)
        camera_thread.daemon = True
        camera_thread.start()
        
        # Wait for 10 seconds or until user quits
        print("Camera started. Press 'Q' in the camera window to quit, or wait 10 seconds...")
        time.sleep(10)
        
        # Force stop if still running
        if camera_interface.is_running:
            print("Auto-stopping camera after 10 seconds...")
            camera_interface.stop()
        
        # Wait for thread to finish
        camera_thread.join(timeout=5)
        
        print("Exit functionality test completed!")
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_exit_functionality()
    if success:
        print("✅ Exit functionality test passed!")
    else:
        print("❌ Exit functionality test failed!")








