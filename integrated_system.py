#!/usr/bin/env python3
"""
Integrated Face Recognition and Fingerprint System
Combines both biometric modalities in a single application.
"""

import cv2
import sys
import os
import time
import logging
import threading
from typing import Optional, Dict, Any
import argparse

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'Face'))
sys.path.append(os.path.join(os.path.dirname(__file__), 'Fingerprint'))

from Face.config_manager import ConfigManager as FaceConfigManager
from Face.face_detector import FaceDetector
from Face.liveness_detector import LivenessDetector
from Face.camera_interface import CameraInterface
from Fingerprint.hardware_scanner import OpticalFingerprintScanner, FingerprintScannerInterface
from Fingerprint.fingerprint_preprocessor import FingerprintPreprocessor
from Fingerprint.minutiae_extractor import MinutiaeExtractor

class IntegratedBiometricSystem:
    """Main class that integrates face recognition and fingerprint scanning"""
    
    def __init__(self, face_config_path: str = "Face/config.yaml"):
        self.face_config_path = face_config_path
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Face recognition components
        self.face_config_manager = None
        self.face_detector = None
        self.liveness_detector = None
        self.camera_interface = None
        
        # Fingerprint components
        self.fingerprint_scanner = None
        self.fingerprint_interface = None
        self.fingerprint_preprocessor = None
        self.minutiae_extractor = None
        
        # System state
        self.current_mode = "menu"  # menu, face_scan, fingerprint_scan
        self.is_running = False
        
    def initialize(self) -> bool:
        """Initialize all system components"""
        try:
            self.logger.info("Initializing integrated biometric system...")
            
            # Initialize face recognition components
            if not self._initialize_face_system():
                return False
            
            # Initialize fingerprint components
            if not self._initialize_fingerprint_system():
                return False
            
            self.logger.info("Integrated biometric system initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    def _initialize_face_system(self) -> bool:
        """Initialize face recognition components"""
        try:
            self.face_config_manager = FaceConfigManager(self.face_config_path)
            
            if not self.face_config_manager.validate_config():
                self.logger.error("Face system configuration validation failed")
                return False
            
            self.face_detector = FaceDetector(self.face_config_manager)
            if not self.face_detector.initialize():
                self.logger.error("Face detector initialization failed")
                return False
            
            self.liveness_detector = LivenessDetector(self.face_detector)
            self.camera_interface = CameraInterface(self.face_detector, self.liveness_detector)
            
            self.logger.info("Face recognition system initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Face system initialization failed: {e}")
            return False
    
    def _initialize_fingerprint_system(self) -> bool:
        """Initialize fingerprint scanning components"""
        try:
            # Try different device indices for the scanner
            device_indices = [0, 1, 2]  # Common indices for USB devices
            
            for device_index in device_indices:
                try:
                    self.logger.info(f"Trying fingerprint scanner at device index {device_index}...")
                    self.fingerprint_scanner = OpticalFingerprintScanner(device_index=device_index)
                    
                    if self.fingerprint_scanner.initialize():
                        self.logger.info(f"Fingerprint scanner found at device {device_index}")
                        break
                    else:
                        self.fingerprint_scanner.cleanup()
                        self.fingerprint_scanner = None
                        
                except Exception as e:
                    self.logger.warning(f"Failed to initialize scanner at device {device_index}: {e}")
                    continue
            
            if not self.fingerprint_scanner:
                self.logger.warning("No fingerprint scanner found - fingerprint functionality disabled")
                return True  # Continue without fingerprint scanner
            
            # Initialize fingerprint processing components
            self.fingerprint_interface = FingerprintScannerInterface(self.fingerprint_scanner)
            self.fingerprint_preprocessor = FingerprintPreprocessor()
            self.minutiae_extractor = MinutiaeExtractor()
            
            self.logger.info("Fingerprint system initialized")
            return True
            
        except Exception as e:
            self.logger.error(f"Fingerprint system initialization failed: {e}")
            return False
    
    def run(self):
        """Main system loop"""
        self.is_running = True
        
        try:
            while self.is_running:
                if self.current_mode == "menu":
                    self._show_main_menu()
                elif self.current_mode == "face_scan":
                    self._run_face_recognition()
                elif self.current_mode == "fingerprint_scan":
                    self._run_fingerprint_scanning()
                elif self.current_mode == "exit":
                    break
                    
        except KeyboardInterrupt:
            print("\nShutdown requested...")
        finally:
            self.cleanup()
    
    def _show_main_menu(self):
        """Display main menu and handle user selection"""
        print("\n" + "="*60)
        print("INTEGRATED BIOMETRIC SYSTEM")
        print("="*60)
        print("1. Face Recognition & Liveness Detection")
        print("2. Fingerprint Scanning")
        print("3. Combined Biometric Verification")
        print("4. System Status")
        print("5. Exit")
        print("="*60)
        
        try:
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                self.current_mode = "face_scan"
            elif choice == "2":
                if self.fingerprint_scanner:
                    self.current_mode = "fingerprint_scan"
                else:
                    print("❌ Fingerprint scanner not available")
                    input("Press Enter to continue...")
            elif choice == "3":
                self._run_combined_verification()
            elif choice == "4":
                self._show_system_status()
                input("Press Enter to continue...")
            elif choice == "5":
                self.current_mode = "exit"
            else:
                print("Invalid choice. Please select 1-5.")
                input("Press Enter to continue...")
                
        except EOFError:
            self.current_mode = "exit"
    
    def _run_face_recognition(self):
        """Run face recognition and liveness detection"""
        print("\n🔍 Starting Face Recognition System...")
        print("Press 'Q' in the camera window to return to main menu")
        
        try:
            # Run face recognition in a separate thread
            face_thread = threading.Thread(target=self._face_recognition_worker)
            face_thread.daemon = True
            face_thread.start()
            
            # Wait for completion or user exit
            face_thread.join()
            
        except Exception as e:
            self.logger.error(f"Face recognition error: {e}")
        
        self.current_mode = "menu"
    
    def _face_recognition_worker(self):
        """Worker thread for face recognition"""
        try:
            self.camera_interface.run(camera_index=0)
        except Exception as e:
            self.logger.error(f"Face recognition worker error: {e}")
    
    def _run_fingerprint_scanning(self):
        """Run fingerprint scanning"""
        if not self.fingerprint_scanner:
            print("❌ Fingerprint scanner not available")
            self.current_mode = "menu"
            return
        
        print("\n🔍 Fingerprint Scanning Mode")
        print("Available options:")
        print("1. Scan single finger")
        print("2. Scan multiple fingers")
        print("3. Return to main menu")
        
        try:
            choice = input("Select option (1-3): ").strip()
            
            if choice == "1":
                self._scan_single_fingerprint()
            elif choice == "2":
                self._scan_multiple_fingerprints()
            elif choice == "3":
                self.current_mode = "menu"
            else:
                print("Invalid choice")
                input("Press Enter to continue...")
                
        except EOFError:
            self.current_mode = "menu"
    
    def _scan_single_fingerprint(self):
        """Scan a single fingerprint"""
        print("\n🔍 Single Fingerprint Scan")
        
        # Get finger details
        finger_types = ["thumb", "index", "middle", "ring", "little"]
        hand_sides = ["left", "right"]
        
        print("Available finger types:")
        for i, finger in enumerate(finger_types, 1):
            print(f"{i}. {finger}")
        
        try:
            finger_choice = int(input("Select finger type (1-5): ")) - 1
            if 0 <= finger_choice < len(finger_types):
                finger_type = finger_types[finger_choice]
            else:
                print("Invalid finger type")
                return
        except ValueError:
            print("Invalid input")
            return
        
        print("Available hand sides:")
        for i, side in enumerate(hand_sides, 1):
            print(f"{i}. {side}")
        
        try:
            hand_choice = int(input("Select hand side (1-2): ")) - 1
            if 0 <= hand_choice < len(hand_sides):
                hand_side = hand_sides[hand_choice]
            else:
                print("Invalid hand side")
                return
        except ValueError:
            print("Invalid input")
            return
        
        # Get subject ID
        subject_id = input("Enter subject ID: ").strip()
        if not subject_id:
            subject_id = "unknown"
        
        # Scan fingerprint
        print(f"\n🔍 Scanning {hand_side} {finger_type} finger for subject {subject_id}...")
        fingerprint = self.fingerprint_interface.scan_fingerprint(finger_type, hand_side)
        
        if fingerprint is not None:
            # Save fingerprint
            saved_path = self.fingerprint_interface.save_fingerprint(
                fingerprint, subject_id, finger_type, hand_side
            )
            
            if saved_path:
                print(f"✅ Fingerprint saved: {saved_path}")
                
                # Process fingerprint for analysis
                self._process_fingerprint(fingerprint, saved_path)
            else:
                print("❌ Failed to save fingerprint")
        else:
            print("❌ Fingerprint capture failed")
        
        input("Press Enter to continue...")
    
    def _scan_multiple_fingerprints(self):
        """Scan multiple fingerprints for a subject"""
        if not self.fingerprint_scanner:
            print("❌ Fingerprint scanner not available")
            return
        
        print("\n🔍 Multiple Fingerprint Scan")
        
        subject_id = input("Enter subject ID: ").strip()
        if not subject_id:
            subject_id = "unknown"
        
        finger_types = ["thumb", "index", "middle", "ring", "little"]
        hand_sides = ["left", "right"]
        
        captured_fingerprints = []
        
        for hand_side in hand_sides:
            for finger_type in finger_types:
                print(f"\n🔍 Scanning {hand_side} {finger_type} finger...")
                print("Press Enter to continue or 's' to skip...")
                
                choice = input().strip().lower()
                if choice == 's':
                    continue
                
                fingerprint = self.fingerprint_interface.scan_fingerprint(finger_type, hand_side)
                
                if fingerprint is not None:
                    saved_path = self.fingerprint_interface.save_fingerprint(
                        fingerprint, subject_id, finger_type, hand_side
                    )
                    
                    if saved_path:
                        captured_fingerprints.append({
                            'path': saved_path,
                            'finger_type': finger_type,
                            'hand_side': hand_side
                        })
                        print(f"✅ {hand_side} {finger_type} captured")
                    else:
                        print(f"❌ Failed to save {hand_side} {finger_type}")
                else:
                    print(f"❌ Failed to capture {hand_side} {finger_type}")
        
        print(f"\n📊 Scan Summary:")
        print(f"Subject ID: {subject_id}")
        print(f"Total fingerprints captured: {len(captured_fingerprints)}")
        
        for fp in captured_fingerprints:
            print(f"  - {fp['hand_side']} {fp['finger_type']}: {fp['path']}")
        
        input("Press Enter to continue...")
    
    def _process_fingerprint(self, fingerprint_image, image_path: str):
        """Process captured fingerprint for analysis"""
        try:
            print("\n🔍 Processing fingerprint for analysis...")
            
            # Preprocess fingerprint
            if self.fingerprint_preprocessor:
                preprocessed = self.fingerprint_preprocessor.process(fingerprint_image)
                if preprocessed:
                    print("✅ Fingerprint preprocessing completed")
                else:
                    print("⚠️ Fingerprint preprocessing had issues")
            
            # Extract minutiae
            if self.minutiae_extractor:
                minutiae = self.minutiae_extractor.extract_minutiae(fingerprint_image)
                if minutiae:
                    print(f"✅ Extracted {len(minutiae)} minutiae points")
                else:
                    print("⚠️ No minutiae points detected")
            
            print("✅ Fingerprint processing completed")
            
        except Exception as e:
            self.logger.error(f"Fingerprint processing failed: {e}")
            print(f"❌ Fingerprint processing failed: {e}")
    
    def _run_combined_verification(self):
        """Run combined face and fingerprint verification"""
        print("\n🔍 Combined Biometric Verification")
        print("This mode will capture both face and fingerprint for verification")
        
        # This is a placeholder for future implementation
        print("Combined verification mode not yet implemented")
        print("Please use individual modes for now")
        
        input("Press Enter to continue...")
    
    def _show_system_status(self):
        """Display system status"""
        print("\n📊 SYSTEM STATUS")
        print("="*40)
        
        # Face system status
        print("Face Recognition System:")
        if self.face_detector and self.face_detector.is_initialized():
            print("  ✅ Initialized and ready")
        else:
            print("  ❌ Not initialized")
        
        # Fingerprint system status
        print("Fingerprint System:")
        if self.fingerprint_scanner and self.fingerprint_scanner.is_connected():
            print("  ✅ Scanner connected and ready")
        else:
            print("  ❌ Scanner not available")
        
        # Camera status
        print("Camera Interface:")
        if self.camera_interface:
            print("  ✅ Available")
        else:
            print("  ❌ Not available")
        
        print("="*40)
    
    def cleanup(self):
        """Clean up system resources"""
        try:
            self.logger.info("Cleaning up integrated biometric system...")
            
            if self.camera_interface:
                self.camera_interface.stop()
            
            if self.fingerprint_scanner:
                self.fingerprint_scanner.cleanup()
            
            cv2.destroyAllWindows()
            
            self.logger.info("System cleanup completed")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Integrated Face Recognition and Fingerprint System')
    parser.add_argument('--config', type=str, default='Face/config.yaml',
                       help='Path to face recognition configuration file')
    parser.add_argument('--status', action='store_true',
                       help='Show system status and exit')
    
    args = parser.parse_args()
    
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Create and run system
    system = IntegratedBiometricSystem(args.config)
    
    if args.status:
        if system.initialize():
            system._show_system_status()
        return
    
    try:
        if system.initialize():
            system.run()
        else:
            print("❌ System initialization failed")
            sys.exit(1)
    except Exception as e:
        print(f"❌ System error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()







