#!/usr/bin/env python3

import cv2
import sys
import os
import argparse
import logging
import signal
from typing import Optional
from config_manager import ConfigManager
from face_detector import FaceDetector
from liveness_detector import LivenessDetector
from camera_interface import CameraInterface

class FaceRecognitionSystem:
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config_manager = None
        self.face_detector = None
        self.liveness_detector = None
        self.camera_interface = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self._shutdown_requested = False
        
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}. Shutting down gracefully...")
        self._shutdown_requested = True
        if self.camera_interface:
            self.camera_interface.stop()
    
    def initialize(self) -> bool:
        try:
            # Try to initialize config manager, but don't fail if it doesn't work
            try:
                self.config_manager = ConfigManager(self.config_path)
                if hasattr(self.config_manager, 'validate_config') and not self.config_manager.validate_config():
                    self.logger.warning("Configuration validation failed, continuing with defaults")
            except Exception as e:
                self.logger.warning(f"Config manager initialization failed: {e}, continuing with defaults")
                self.config_manager = None
            
            # Initialize face detector without config_manager parameter
            self.face_detector = FaceDetector()
            
            # Check if landmarks file exists
            landmarks_file = "shape_predictor_68_face_landmarks.dat"
            if not os.path.exists(landmarks_file):
                self.logger.error(f"Landmarks file not found: {landmarks_file}")
                return False
            
            self.liveness_detector = LivenessDetector(self.face_detector)
            
            self.camera_interface = CameraInterface(self.face_detector, self.liveness_detector)
            
            self.logger.info("Face recognition system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def check_requirements(self) -> bool:
        self.logger.info("Checking system requirements...")
        
        # Check for essential landmarks file
        landmarks_file = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(landmarks_file):
            self.logger.error(f"Landmarks file not found: {landmarks_file}")
            self.logger.error("Please download the shape_predictor_68_face_landmarks.dat file")
            return False
        
        # Check for required Python packages
        try:
            import cv2
            import dlib
            import numpy as np
            import scipy
            import imutils
            self.logger.info("All required Python packages are available")
        except ImportError as e:
            self.logger.error(f"Missing required package: {e}")
            return False
        
        # Camera check is optional - will be checked when starting
        self.logger.info("Camera will be checked when starting the system")
        
        return True
    
    def run(self, camera_index: Optional[int] = None, skip_checks: bool = False) -> None:
        try:
            #check requirements unless skipped
            if not skip_checks and not self.check_requirements():
                sys.exit(1)
            
            # Use provided camera_index or default to 0
            if camera_index is None:
                camera_index = 0
            
            self.logger.info(f"Starting camera (index: {camera_index})...")
            print("\n" + "="*60)
            print("FACE RECOGNITION SYSTEM STARTED")
            print("="*60)
            print("Controls:")
            print("  Q or ESC - Quit the application")
            print("  F - Toggle fullscreen mode")
            print("  R - Reset liveness detection")
            print("  A - Approve photo (during review)")
            print("="*60)
            print("Press Ctrl+C to force quit if needed")
            print("="*60 + "\n")
            
            # Start the camera interface
            self.camera_interface.run()
            
        except FileNotFoundError as e:
            self.logger.error(f"Required file not found: {e}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"System error: {e}")
            import traceback
            traceback.print_exc()
            sys.exit(1)
        finally:
            self.cleanup()
    
    def cleanup(self) -> None:
        try:
            if self.face_detector:
                self.face_detector.cleanup()
            if self.camera_interface:
                pass
            self.logger.info("System cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def get_system_status(self) -> dict:
        status = {
            'config_loaded': self.config_manager is not None,
            'face_detector_initialized': self.face_detector is not None and hasattr(self.face_detector, 'is_initialized') and self.face_detector.is_initialized(),
            'liveness_detector_initialized': self.liveness_detector is not None,
            'camera_interface_initialized': self.camera_interface is not None
        }
        
        if self.config_manager:
            status['config_path'] = self.config_path
            # Only check config validation if the method exists
            if hasattr(self.config_manager, 'validate_config'):
                status['config_valid'] = self.config_manager.validate_config()
            else:
                status['config_valid'] = 'Method not available'
        
        return status
    
    def __enter__(self):
        if not self.initialize():
            raise RuntimeError("Failed to initialize face recognition system")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()

def main():
    parser = argparse.ArgumentParser(description='Face Recognition with Liveness Detection (OOP Version)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--camera', type=int, default=None,
                       help='Camera index (overrides config)')
    parser.add_argument('--no-check', action='store_true',
                       help='Skip requirements check')
    parser.add_argument('--status', action='store_true',
                       help='Show system status and exit')
    
    args = parser.parse_args()
    
    with FaceRecognitionSystem(args.config) as system:
        if args.status:
            status = system.get_system_status()
            print("System Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
            return
        
        system.run(camera_index=args.camera, skip_checks=args.no_check)

if __name__ == "__main__":
    main() 