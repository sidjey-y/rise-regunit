#!/usr/bin/env python3
"""
Hardware Fingerprint Scanner Interface
Integrates with optical fingerprint scanners and connects to the existing fingerprint processing pipeline.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Tuple
from abc import ABC, abstractmethod
from finger_validator import FingerValidator

class FingerprintScanner(ABC):
    """Abstract base class for fingerprint scanners"""
    
    @abstractmethod
    def initialize(self) -> bool:
        """Initialize the scanner hardware"""
        pass
    
    @abstractmethod
    def capture_fingerprint(self) -> Optional[np.ndarray]:
        """Capture a fingerprint image from the scanner"""
        pass
    
    @abstractmethod
    def is_connected(self) -> bool:
        """Check if scanner is connected and ready"""
        pass
    
    @abstractmethod
    def cleanup(self):
        """Clean up scanner resources"""
        pass

class OpticalFingerprintScanner(FingerprintScanner):
    """Optical fingerprint scanner implementation"""
    
    def __init__(self, device_index: int = 0, resolution: Tuple[int, int] = (500, 500)):
        """
        Initialize optical fingerprint scanner
        
        Args:
            device_index: Camera device index (usually 0 for built-in, 1+ for USB devices)
            resolution: Desired capture resolution (width, height)
        """
        self.device_index = device_index
        self.resolution = resolution
        self.cap = None
        self.logger = logging.getLogger(self.__class__.__name__)
        self.is_initialized = False
        
        # Scanner settings
        self.exposure = -6  # Lower exposure for better fingerprint contrast
        self.brightness = 50
        self.contrast = 100
        self.saturation = 0  # Grayscale for fingerprints
        
    def initialize(self) -> bool:
        """Initialize the scanner hardware"""
        try:
            self.logger.info(f"Initializing optical fingerprint scanner (device {self.device_index})...")
            
            # Try to open the camera device
            self.cap = cv2.VideoCapture(self.device_index)
            if not self.cap.isOpened():
                self.logger.error(f"Failed to open camera device {self.device_index}")
                return False
            
            # Set camera properties for fingerprint scanning
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])
            self.cap.set(cv2.CAP_PROP_EXPOSURE, self.exposure)
            self.cap.set(cv2.CAP_PROP_BRIGHTNESS, self.brightness)
            self.cap.set(cv2.CAP_PROP_CONTRAST, self.contrast)
            self.cap.set(cv2.CAP_PROP_SATURATION, self.saturation)
            
            # Test capture
            ret, test_frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to capture test frame from scanner")
                return False
            
            self.logger.info(f"Scanner initialized successfully. Resolution: {test_frame.shape}")
            self.is_initialized = True
            return True
            
        except Exception as e:
            self.logger.error(f"Scanner initialization failed: {e}")
            return False
    
    def capture_fingerprint(self) -> Optional[np.ndarray]:
        """Capture a fingerprint image from the scanner"""
        if not self.is_initialized or not self.cap:
            self.logger.error("Scanner not initialized")
            return None
        
        try:
            # Capture frame
            ret, frame = self.cap.read()
            if not ret:
                self.logger.error("Failed to capture frame from scanner")
                return None
            
            # Convert to grayscale for fingerprint processing
            if len(frame.shape) == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = frame.copy()
            
            # Apply fingerprint-specific preprocessing
            processed = self._preprocess_for_fingerprint(gray)
            
            return processed
            
        except Exception as e:
            self.logger.error(f"Fingerprint capture failed: {e}")
            return None
    
    def _preprocess_for_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """Preprocess captured image for fingerprint analysis"""
        try:
            # Enhance contrast
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced = clahe.apply(image)
            
            # Reduce noise
            denoised = cv2.fastNlMeansDenoising(enhanced)
            
            # Sharpen edges
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            sharpened = cv2.filter2D(denoised, -1, kernel)
            
            return sharpened
            
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            return image
    
    def is_connected(self) -> bool:
        """Check if scanner is connected and ready"""
        return self.is_initialized and self.cap and self.cap.isOpened()
    
    def cleanup(self):
        """Clean up scanner resources"""
        try:
            if self.cap:
                self.cap.release()
                self.cap = None
            self.is_initialized = False
            self.logger.info("Scanner cleanup completed")
        except Exception as e:
            self.logger.error(f"Scanner cleanup failed: {e}")

class FingerprintScannerInterface:
    """Main interface for fingerprint scanning operations with automatic finger detection"""
    
    def __init__(self, scanner: FingerprintScanner):
        self.scanner = scanner
        self.logger = logging.getLogger(self.__class__.__name__)
        self.validator = FingerValidator()
        self.auto_detection_enabled = True
        
    def scan_fingerprint(self, finger_type: str = "index", hand_side: str = "right") -> Optional[np.ndarray]:
        """
        Scan a fingerprint with user guidance
        
        Args:
            finger_type: Type of finger (thumb, index, middle, ring, little)
            hand_side: Hand side (left, right)
            
        Returns:
            Captured fingerprint image or None if failed
        """
        if not self.scanner.is_connected():
            self.logger.error("Scanner not connected")
            return None
        
        print(f"\n🔍 Scanning {hand_side} {finger_type} finger...")
        print("Place your finger on the scanner and press any key to capture")
        print("Press 'q' to cancel")
        
        # Show live preview
        preview_window = f"Fingerprint Scanner - {hand_side.title()} {finger_type.title()}"
        cv2.namedWindow(preview_window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(preview_window, 400, 400)
        
        try:
            while True:
                # Capture live preview
                frame = self.scanner.capture_fingerprint()
                if frame is not None:
                    # Display with instructions
                    display_frame = self._create_display_frame(frame, finger_type, hand_side)
                    cv2.imshow(preview_window, display_frame)
                
                # Wait for key press
                key = cv2.waitKey(100) & 0xFF
                
                if key == ord('q'):
                    print("Scanning cancelled")
                    break
                elif key != 255:  # Any other key pressed
                    print("Capturing fingerprint...")
                    # Capture final image
                    final_image = self.scanner.capture_fingerprint()
                    if final_image is not None:
                        print("✅ Fingerprint captured successfully!")
                        return final_image
                    else:
                        print("❌ Capture failed, try again")
        
        finally:
            cv2.destroyWindow(preview_window)
        
        return None
    
    def scan_fingerprint_with_validation(self, expected_finger_type: str = "index", 
                                       expected_hand_side: str = "right",
                                       max_retries: int = 3) -> Optional[np.ndarray]:
        """
        Scan a fingerprint with automatic validation and retry logic
        
        Args:
            expected_finger_type: Expected finger type (thumb, index, middle, ring, little)
            expected_hand_side: Expected hand side (left, right)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Captured fingerprint image or None if failed
        """
        if not self.scanner.is_connected():
            self.logger.error("Scanner not connected")
            return None
        
        print(f"\n🔍 Scanning {expected_hand_side} {expected_finger_type} finger...")
        print("System will automatically detect which finger you're scanning")
        
        retry_count = 0
        
        while retry_count < max_retries:
            print(f"\n--- Attempt {retry_count + 1}/{max_retries} ---")
            
            # Capture fingerprint
            fingerprint = self.scanner.capture_fingerprint()
            if fingerprint is None:
                print("❌ Capture failed, please try again")
                retry_count += 1
                continue
            
            # Validate the captured finger
            validation_result = self.validator.validate_finger_scan(
                fingerprint, expected_finger_type, expected_hand_side
            )
            
            # Display validation results
            print(f"\n📊 Validation Results:")
            print(f"Expected: {validation_result['expected_finger']}")
            print(f"Detected: {validation_result['detected_finger']}")
            print(f"Confidence: {validation_result['confidence']:.2f}")
            print(f"Validation Score: {validation_result['validation_score']:.2f}")
            
            # Check if validation passed
            if validation_result['is_valid']:
                print("✅ Finger validation successful!")
                print(validation_result['guidance'])
                
                # Analyze quality
                quality = self.validator.analyze_finger_quality(fingerprint)
                if 'overall_score' in quality:
                    print(f"📈 Image Quality: {quality['quality_level']} ({quality['overall_score']:.2f})")
                
                return fingerprint
            else:
                print("❌ Finger validation failed!")
                print(validation_result['guidance'])
                
                # Show retry instructions
                if retry_count < max_retries - 1:
                    print("\n" + self.validator.get_retry_instructions(expected_finger_type, expected_hand_side))
                    
                    # Ask user if they want to retry
                    retry_input = input("\nPress Enter to retry, or 'q' to cancel: ").strip().lower()
                    if retry_input == 'q':
                        print("Scanning cancelled by user")
                        break
                else:
                    print(f"\n❌ Maximum retry attempts ({max_retries}) reached.")
                    print("Please ensure you're scanning the correct finger.")
                
                retry_count += 1
        
        return None
    
    def _create_display_frame(self, image: np.ndarray, finger_type: str, hand_side: str) -> np.ndarray:
        """Create display frame with instructions"""
        # Convert to BGR for display
        if len(image.shape) == 2:
            display = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            display = image.copy()
        
        # Add text overlay
        cv2.putText(display, f"{hand_side.title()} {finger_type.title()}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display, "Press any key to capture", 
                    (10, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(display, "Press 'q' to cancel", 
                    (10, display.shape[0] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return display
    
    def enable_auto_detection(self, enabled: bool = True):
        """Enable or disable automatic finger detection"""
        self.auto_detection_enabled = enabled
        status = "enabled" if enabled else "disabled"
        self.logger.info(f"Automatic finger detection {status}")
        print(f"🔍 Automatic finger detection {status}")
    
    def get_validation_settings(self) -> Dict[str, Any]:
        """Get current validation settings"""
        return {
            'auto_detection_enabled': self.auto_detection_enabled,
            'confidence_threshold': self.validator.confidence_threshold,
            'validator_status': 'initialized'
        }
    
    def save_fingerprint(self, image: np.ndarray, subject_id: str, finger_type: str, 
                        hand_side: str, output_dir: str = "captured_fingerprints") -> Optional[str]:
        """
        Save captured fingerprint image
        
        Args:
            image: Captured fingerprint image
            subject_id: Subject identifier
            finger_type: Type of finger
            hand_side: Hand side
            output_dir: Output directory
            
        Returns:
            Path to saved image or None if failed
        """
        try:
            import os
            from datetime import datetime
            
            # Create output directory
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{subject_id}_{hand_side}_{finger_type}_{timestamp}.png"
            filepath = os.path.join(output_dir, filename)
            
            # Save image
            cv2.imwrite(filepath, image)
            self.logger.info(f"Fingerprint saved: {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Failed to save fingerprint: {e}")
            return None

def test_scanner():
    """Test function for the fingerprint scanner"""
    print("Testing fingerprint scanner...")
    
    # Initialize scanner
    scanner = OpticalFingerprintScanner(device_index=0)
    if not scanner.initialize():
        print("❌ Scanner initialization failed")
        return False
    
    # Create interface
    interface = FingerprintScannerInterface(scanner)
    
    try:
        # Test scan
        print("Scanner initialized successfully!")
        print("Testing automatic finger detection...")
        
        # Show validation settings
        settings = interface.get_validation_settings()
        print(f"🔧 Validation settings: {settings}")
        
        # Test automatic finger detection with validation
        print("\n🔍 Testing automatic finger detection for RIGHT THUMB...")
        print("Try scanning different fingers to see the validation in action!")
        
        fingerprint = interface.scan_fingerprint_with_validation("thumb", "right", max_retries=3)
        
        if fingerprint is not None:
            print("✅ Fingerprint captured and validated successfully!")
            
            # Save the fingerprint
            saved_path = interface.save_fingerprint(fingerprint, "test_user", "thumb", "right")
            if saved_path:
                print(f"✅ Fingerprint saved to: {saved_path}")
            else:
                print("❌ Failed to save fingerprint")
            
            return True
        else:
            print("❌ Fingerprint capture or validation failed")
            return False
            
    finally:
        scanner.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    test_scanner()

