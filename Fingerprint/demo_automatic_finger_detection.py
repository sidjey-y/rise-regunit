#!/usr/bin/env python3
"""
Demo: Automatic Finger Detection and Validation System

This script demonstrates how the system automatically detects which finger is being scanned
and validates it against the expected finger. If the wrong finger is scanned, it will
detect the mismatch and prompt the user to retry with the correct finger.

Usage:
    python demo_automatic_finger_detection.py

Features:
    - Automatic finger classification using computer vision
    - Real-time validation of scanned fingers
    - User guidance when wrong finger is detected
    - Quality analysis of captured fingerprints
    - Retry logic with clear instructions
"""

import cv2
import numpy as np
import time
import logging
from typing import Dict, Any, Optional

from hardware_scanner import OpticalFingerprintScanner, FingerprintScannerInterface
from finger_classifier import ComputerVisionFingerClassifier
from finger_validator import FingerValidator

class AutomaticFingerDetectionDemo:
    """Demo class for automatic finger detection system"""
    
    def __init__(self):
        self.logger = self._setup_logging()
        self.scanner = None
        self.interface = None
        self.classifier = ComputerVisionFingerClassifier()
        self.validator = FingerValidator()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for the demo"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        return logging.getLogger(self.__class__.__name__)
    
    def initialize_system(self) -> bool:
        """Initialize the fingerprint scanning system"""
        try:
            print("🚀 Initializing Automatic Finger Detection System...")
            
            # Initialize scanner
            self.scanner = OpticalFingerprintScanner(device_index=0)
            if not self.scanner.initialize():
                print("❌ Scanner initialization failed")
                return False
            
            # Create interface
            self.interface = FingerprintScannerInterface(self.scanner)
            
            # Show system status
            settings = self.interface.get_validation_settings()
            print(f"✅ System initialized successfully!")
            print(f"🔧 Validation settings: {settings}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            print(f"❌ Initialization error: {e}")
            return False
    
    def run_interactive_demo(self):
        """Run interactive demo with user input"""
        print("\n🎯 AUTOMATIC FINGER DETECTION DEMO")
        print("=" * 50)
        
        while True:
            print("\n📋 Available Options:")
            print("1. Test automatic detection for specific finger")
            print("2. Run validation test with sample images")
            print("3. Show system information")
            print("4. Exit")
            
            try:
                choice = input("\nSelect option (1-4): ").strip()
                
                if choice == '1':
                    self.test_specific_finger()
                elif choice == '2':
                    self.run_validation_test()
                elif choice == '3':
                    self.show_system_info()
                elif choice == '4':
                    print("👋 Goodbye!")
                    break
                else:
                    print("❌ Invalid option. Please select 1-4.")
                    
            except KeyboardInterrupt:
                print("\n👋 Demo interrupted by user")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
    
    def test_specific_finger(self):
        """Test automatic detection for a specific finger"""
        print("\n🔍 Testing Automatic Finger Detection")
        print("-" * 40)
        
        # Get finger details from user
        finger_types = ["thumb", "index", "middle", "ring", "little"]
        hand_sides = ["left", "right"]
        
        print("\nAvailable finger types:")
        for i, finger in enumerate(finger_types, 1):
            print(f"  {i}. {finger}")
        
        try:
            finger_choice = int(input("\nSelect finger type (1-5): ")) - 1
            if not (0 <= finger_choice < len(finger_types)):
                print("❌ Invalid finger type")
                return
            finger_type = finger_types[finger_choice]
        except ValueError:
            print("❌ Invalid input")
            return
        
        print("\nAvailable hand sides:")
        for i, side in enumerate(hand_sides, 1):
            print(f"  {i}. {side}")
        
        try:
            hand_choice = int(input("\nSelect hand side (1-2): ")) - 1
            if not (0 <= hand_choice < len(hand_sides)):
                print("❌ Invalid hand side")
                return
            hand_side = hand_sides[hand_choice]
        except ValueError:
            print("❌ Invalid input")
            return
        
        # Get retry count
        try:
            max_retries = int(input("\nMaximum retry attempts (default 3): ") or "3")
            max_retries = max(1, min(5, max_retries))  # Limit between 1-5
        except ValueError:
            max_retries = 3
        
        print(f"\n🎯 Testing detection for: {hand_side.title()} {finger_type.title()}")
        print(f"🔄 Maximum retries: {max_retries}")
        print("\n💡 Try scanning different fingers to see the validation in action!")
        print("   The system will automatically detect which finger you're scanning")
        print("   and tell you if it matches what's expected.")
        
        # Run the validation scan
        fingerprint = self.interface.scan_fingerprint_with_validation(
            finger_type, hand_side, max_retries
        )
        
        if fingerprint is not None:
            print("\n✅ SUCCESS: Correct finger detected and validated!")
            
            # Ask if user wants to save
            save_choice = input("\nSave fingerprint? (y/n): ").strip().lower()
            if save_choice in ['y', 'yes']:
                subject_id = input("Enter subject ID: ").strip() or "demo_user"
                saved_path = self.interface.save_fingerprint(
                    fingerprint, subject_id, finger_type, hand_side
                )
                if saved_path:
                    print(f"💾 Fingerprint saved to: {saved_path}")
                else:
                    print("❌ Failed to save fingerprint")
        else:
            print("\n❌ FAILED: Could not validate correct finger within retry limit")
    
    def run_validation_test(self):
        """Run validation test with sample images"""
        print("\n🧪 Running Validation Test")
        print("-" * 30)
        
        # Create test images (simulating different fingers)
        test_cases = [
            ("thumb", "right", "Large round finger"),
            ("index", "left", "Medium elongated finger"),
            ("little", "right", "Small thin finger")
        ]
        
        print("Testing finger classification with sample images...")
        
        for finger_type, hand_side, description in test_cases:
            print(f"\n🔍 Testing: {hand_side.title()} {finger_type.title()} ({description})")
            
            # Create a simulated fingerprint image
            # In real usage, this would be a captured image
            test_image = self._create_test_fingerprint(finger_type)
            
            # Classify the finger
            classification = self.classifier.classify_finger(test_image)
            
            print(f"  Detected: {classification['finger_type']} ({classification['hand_side']})")
            print(f"  Confidence: {classification['confidence']:.2f}")
            print(f"  Method: {classification['classification_method']}")
            
            # Validate against expected
            validation = self.validator.validate_finger_scan(
                test_image, finger_type, hand_side
            )
            
            print(f"  Valid: {'✅' if validation['is_valid'] else '❌'}")
            print(f"  Score: {validation['validation_score']:.2f}")
        
        print("\n✅ Validation test completed!")
    
    def _create_test_fingerprint(self, finger_type: str) -> np.ndarray:
        """Create a test fingerprint image for demonstration"""
        # Create a base image
        size = 200
        image = np.zeros((size, size), dtype=np.uint8)
        
        # Add different patterns based on finger type
        if finger_type == "thumb":
            # Large, round pattern
            cv2.circle(image, (size//2, size//2), size//3, 255, -1)
        elif finger_type in ["index", "middle"]:
            # Medium, elongated pattern
            cv2.ellipse(image, (size//2, size//2), (size//4, size//3), 0, 0, 360, 255, -1)
        else:
            # Small, thin pattern
            cv2.ellipse(image, (size//2, size//2), (size//5, size//4), 0, 0, 360, 255, -1)
        
        # Add some noise to simulate real fingerprint
        noise = np.random.randint(0, 50, (size, size), dtype=np.uint8)
        image = cv2.add(image, noise)
        
        return image
    
    def show_system_info(self):
        """Show system information and capabilities"""
        print("\nℹ️  SYSTEM INFORMATION")
        print("-" * 30)
        
        if self.interface:
            settings = self.interface.get_validation_settings()
            print(f"🔧 Validation Settings:")
            for key, value in settings.items():
                print(f"   {key}: {value}")
        
        print(f"\n🤖 Finger Classifier:")
        print(f"   Model loaded: {self.classifier.model is not None}")
        print(f"   Feature count: {len(self.classifier.feature_names)}")
        print(f"   Supported fingers: {', '.join(self.classifier.finger_types)}")
        
        print(f"\n✅ Finger Validator:")
        print(f"   Confidence threshold: {self.validator.confidence_threshold}")
        print(f"   Supported characteristics: {len(self.validator.finger_characteristics)}")
        
        print(f"\n📊 Quality Analysis:")
        print(f"   Brightness scoring: Available")
        print(f"   Contrast scoring: Available")
        print(f"   Sharpness scoring: Available")
        
        print(f"\n🔄 Retry System:")
        print(f"   Automatic validation: Enabled")
        print(f"   User guidance: Available")
        print(f"   Quality feedback: Available")
    
    def cleanup(self):
        """Clean up system resources"""
        try:
            if self.scanner:
                self.scanner.cleanup()
            cv2.destroyAllWindows()
            print("🧹 System cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup failed: {e}")

def main():
    """Main demo function"""
    print("🎯 Automatic Finger Detection System Demo")
    print("=" * 50)
    
    demo = AutomaticFingerDetectionDemo()
    
    try:
        # Initialize system
        if not demo.initialize_system():
            print("❌ Failed to initialize system. Exiting.")
            return
        
        # Run interactive demo
        demo.run_interactive_demo()
        
    except Exception as e:
        print(f"❌ Demo error: {e}")
        demo.logger.error(f"Demo error: {e}")
    
    finally:
        demo.cleanup()

if __name__ == "__main__":
    main()
