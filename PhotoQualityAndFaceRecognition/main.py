#!/usr/bin/env python3

import cv2
import sys
import os
import argparse
import logging
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
        
    def initialize(self) -> bool:
        try:
            self.config_manager = ConfigManager(self.config_path)
            
            if not self.config_manager.validate_config():
                self.logger.error("Configuration validation failed")
                return False
            
            self.face_detector = FaceDetector(self.config_manager)
            if not self.face_detector.initialize():
                self.logger.error("Face detector initialization failed")
                return False
            
            self.liveness_detector = LivenessDetector(self.face_detector)
            
            self.camera_interface = CameraInterface(self.face_detector, self.liveness_detector)
            
            self.logger.info("Face recognition system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    def check_requirements(self) -> bool:
        self.logger.info("Checking system requirements...")
        
        landmarks_file = self.config_manager.get('face_detection.landmarks_file', 
                                                'shape_predictor_68_face_landmarks.dat')
        if not os.path.exists(landmarks_file):
            self.logger.error(f"Landmarks file not found: {landmarks_file}")
            return False
        
        camera_config = self.config_manager.get_camera_config()
        camera_index = camera_config.get('default_index', 0)
        
        test_cap = cv2.VideoCapture(camera_index)
        if not test_cap.isOpened():
            self.logger.error(f"Camera not available at index: {camera_index}")
            test_cap.release()
            return False
        test_cap.release()
        
        self.logger.info("✓ All requirements met!")
        return True
    
    def run(self, camera_index: Optional[int] = None, skip_checks: bool = False) -> None:
        try:
            # Check requirements unless skipped
            if not skip_checks and not self.check_requirements():
                sys.exit(1)
            
            if camera_index is not None:
                self.config_manager._config['camera']['default_index'] = camera_index
            
            camera_config = self.config_manager.get_camera_config()
            camera_index = camera_config.get('default_index', 0)
            
            self.logger.info(f"Starting camera (index: {camera_index})...")
            
            self.camera_interface.run()
            
        except FileNotFoundError as e:
            self.logger.error(f"Required file not found: {e}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"System error: {e}")
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
            'face_detector_initialized': self.face_detector is not None and self.face_detector.is_initialized(),
            'liveness_detector_initialized': self.liveness_detector is not None,
            'camera_interface_initialized': self.camera_interface is not None
        }
        
        if self.config_manager:
            status['config_path'] = self.config_path
            status['config_valid'] = self.config_manager.validate_config()
        
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