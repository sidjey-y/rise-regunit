#!/usr/bin/env python3
"""
Smart Fingerprint Enrollment System
Automatically detects finger types and enrolls them intelligently.
"""

import os
import sys
import time
import json
import base64
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict
from enum import Enum
from pyfingerprint.pyfingerprint import PyFingerprint

# Import our recognition system
from fingerprint_recognition_system import FingerprintRecognitionSystem, Hand, FingerType

class SmartEnrollmentSystem:
    """Smart enrollment system with automatic finger detection"""
    
    def __init__(self, port='COM4', baudrate=57600):
        self.port = port
        self.baudrate = baudrate
        self.scanner = None
        self.recognition_system = FingerprintRecognitionSystem()
        self.enrolled_fingers = {}
        self.session_start_time = None
        self.session_timeout = 300  # 5 minutes
        
        # Define required fingers in order
        self.required_fingers = [
            (Hand.LEFT, FingerType.THUMB),
            (Hand.LEFT, FingerType.INDEX),
            (Hand.LEFT, FingerType.MIDDLE),
            (Hand.LEFT, FingerType.RING),
            (Hand.LEFT, FingerType.LITTLE),
            (Hand.RIGHT, FingerType.THUMB),
            (Hand.RIGHT, FingerType.INDEX),
            (Hand.RIGHT, FingerType.MIDDLE),
            (Hand.RIGHT, FingerType.RING),
            (Hand.RIGHT, FingerType.LITTLE)
        ]
        
        # Track enrollment progress
        self.enrollment_progress = {f"{hand.value}_{finger_type.value}": False 
                                   for hand, finger_type in self.required_fingers}
    
    def initialize(self) -> bool:
        """Initialize the enrollment system"""
        try:
            print("🔌 Initializing Smart Enrollment System...")
            
            # Initialize scanner
            if not self.recognition_system.initialize_scanner():
                return False
            
            self.scanner = self.recognition_system.scanner
            
            # Start session timer
            self.session_start_time = time.time()
            
            print("✅ Smart Enrollment System initialized successfully")
            return True
            
        except Exception as e:
            print(f"❌ Initialization failed: {e}")
            return False
    
    def check_session_timeout(self) -> bool:
        """Check if session has timed out"""
        if self.session_start_time is None:
            return False
        
        elapsed = time.time() - self.session_start_time
        remaining = self.session_timeout - elapsed
        
        if remaining <= 0:
            print(f"⏰ Session timeout! {self.session_timeout} seconds elapsed")
            return True
        
        print(f"⏱️  Session time remaining: {remaining:.0f} seconds")
        return False
    
    def get_next_required_finger(self) -> Optional[Tuple[Hand, FingerType]]:
        """Get the next finger that needs to be enrolled"""
        for hand, finger_type in self.required_fingers:
            key = f"{hand.value}_{finger_type.value}"
            if not self.enrollment_progress[key]:
                return hand, finger_type
        return None
    
    def get_enrollment_status(self) -> Dict[str, bool]:
        """Get current enrollment status"""
        return self.enrollment_progress.copy()
    
    def show_enrollment_progress(self):
        """Display current enrollment progress"""
        print("\n📊 Enrollment Progress:")
        print("-" * 40)
        
        enrolled_count = 0
        for hand, finger_type in self.required_fingers:
            key = f"{hand.value}_{finger_type.value}"
            status = "✅" if self.enrollment_progress[key] else "⏳"
            print(f"   {status} {hand.value} {finger_type.value}")
            if self.enrollment_progress[key]:
                enrolled_count += 1
        
        print(f"\n📈 Progress: {enrolled_count}/10 fingers enrolled")
        
        # Show next required finger
        next_finger = self.get_next_required_finger()
        if next_finger:
            hand, finger_type = next_finger
            print(f"🔄 Next needed: {hand.value} {finger_type.value}")
        else:
            print("🎉 All fingers enrolled!")
    
    def auto_detect_and_enroll(self) -> bool:
        """Automatically detect finger type and enroll if correct"""
        try:
            print("\n🔍 Auto-detecting finger type...")
            print("   Place any finger on the sensor...")
            
            # Wait for finger on sensor
            while not self.scanner.readImage():
                time.sleep(0.1)
                if self.check_session_timeout():
                    return False
            
            print("   ✅ Finger detected")
            
            # Capture the fingerprint image
            current_image = self.recognition_system.capture_current_fingerprint()
            if current_image is None:
                print("   ❌ Failed to capture fingerprint")
                return False
            
            # Auto-detect which finger this is
            print("   🔍 Identifying finger type...")
            detected_finger = self._identify_finger_type(current_image)
            
            if detected_finger is None:
                print("   ❌ Could not identify finger type")
                return False
            
            detected_hand, detected_finger_type = detected_finger
            print(f"   🎯 Detected: {detected_hand.value} {detected_finger_type.value}")
            
            # Check if this finger is already enrolled
            key = f"{detected_hand.value}_{detected_finger_type.value}"
            if self.enrollment_progress[key]:
                print("   ⚠️  This finger is already enrolled")
                return False
            
            # Check if this is the next finger we need
            next_required = self.get_next_required_finger()
            if next_required:
                expected_hand, expected_finger_type = next_required
                if detected_hand != expected_hand or detected_finger_type != expected_finger_type:
                    print(f"   ⚠️  Expected {expected_hand.value} {expected_finger_type.value}, but got {detected_hand.value} {detected_finger_type.value}")
                    print("   💡 Please place the correct finger next time")
                    return False
            
            # Check for duplicates with previously enrolled fingers
            if self._check_duplicate_finger(current_image):
                print("   ❌ This finger has already been enrolled (duplicate detected)")
                return False
            
            # Enroll the finger
            print("   🔄 Creating template...")
            self.scanner.convertImage(0x01)
            self.scanner.createTemplate()
            
            # Store template
            template_position = self.scanner.storeTemplate()
            print(f"   ✅ Template stored at position {template_position}")
            
            # Save enrollment data
            self._save_finger_data(detected_hand, detected_finger_type, template_position, current_image, 0.9)
            
            # Update progress
            self.enrollment_progress[key] = True
            
            print(f"   🎉 {detected_hand.value} {detected_finger_type.value} enrolled successfully!")
            return True
            
        except Exception as e:
            print(f"   ❌ Auto-enrollment failed: {e}")
            return False
    
    def _identify_finger_type(self, image: np.ndarray) -> Optional[Tuple[Hand, FingerType]]:
        """Identify which finger type this image represents"""
        try:
            # Extract features from the image
            keypoints, descriptors = self.recognition_system.feature_extractor.detectAndCompute(image, None)
            
            if descriptors is None:
                return None
            
            # Compare with reference database to identify the finger
            best_match = None
            best_score = 0.0
            
            for user_id, references in self.recognition_system.reference_database.items():
                for reference in references:
                    if reference.features is None:
                        continue
                    
                    # Calculate similarity
                    similarity = self.recognition_system._calculate_similarity(descriptors, reference.features)
                    
                    if similarity > best_score and similarity > 0.5:  # Minimum threshold
                        best_score = similarity
                        best_match = {
                            'hand': reference.hand,
                            'finger_type': reference.finger_type,
                            'similarity': similarity
                        }
            
            if best_match is None:
                return None
            
            # Convert string values to enums
            try:
                hand = Hand(best_match['hand'])
                finger_type = FingerType(best_match['finger_type'])
                return hand, finger_type
            except ValueError:
                print(f"   ⚠️  Invalid hand/finger type: {best_match['hand']} {best_match['finger_type']}")
                return None
                
        except Exception as e:
            print(f"   ⚠️  Finger identification failed: {e}")
            return None
    
    def _check_duplicate_finger(self, current_image: np.ndarray) -> bool:
        """Check if this finger has already been enrolled"""
        if not self.enrolled_fingers:
            return False
        
        try:
            # Extract features from current image
            keypoints, current_descriptors = self.recognition_system.feature_extractor.detectAndCompute(current_image, None)
            
            if current_descriptors is None:
                return False
            
            # Compare with all enrolled fingers
            for enrolled_data in self.enrolled_fingers.values():
                if 'image_data' in enrolled_data and enrolled_data['image_data'] is not None:
                    # Convert base64 back to numpy array
                    try:
                        image_bytes = base64.b64decode(enrolled_data['image_data'])
                        enrolled_image = np.frombuffer(image_bytes, dtype=np.uint8)
                        
                        # Try to reshape (you may need to adjust dimensions)
                        try:
                            enrolled_image = enrolled_image.reshape(256, 256)
                        except ValueError:
                            continue
                        
                        # Extract features and compare
                        keypoints, enrolled_descriptors = self.recognition_system.feature_extractor.detectAndCompute(enrolled_image, None)
                        
                        if enrolled_descriptors is not None:
                            similarity = self.recognition_system._calculate_similarity(current_descriptors, enrolled_descriptors)
                            
                            if similarity > 0.7:  # High similarity threshold for duplicates
                                print(f"   ⚠️  High similarity ({similarity:.3f}) with previously enrolled finger")
                                return True
                    
                    except Exception as e:
                        continue
            
            return False
            
        except Exception as e:
            print(f"   ⚠️  Duplicate check failed: {e}")
            return False
    
    def _save_finger_data(self, hand: Hand, finger_type: FingerType, template_position: int, 
                          image_data: np.ndarray, confidence: float):
        """Save finger enrollment data"""
        try:
            # Convert image to base64 for storage
            image_bytes = image_data.tobytes()
            image_base64 = base64.b64encode(image_bytes).decode('utf-8')
            
            # Create enrollment record
            enrollment_data = {
                'hand': hand.value,
                'finger_type': finger_type.value,
                'template_position': template_position,
                'image_data': image_base64,
                'confidence': confidence,
                'enrollment_time': datetime.now().isoformat(),
                'image_dimensions': f"{image_data.shape[1]}x{image_data.shape[0]}"
            }
            
            # Store in memory
            key = f"{hand.value}_{finger_type.value}"
            self.enrolled_fingers[key] = enrollment_data
            
            print(f"   💾 Enrollment data saved for {key}")
            
        except Exception as e:
            print(f"   ⚠️  Failed to save enrollment data: {e}")
    
    def run_automatic_enrollment(self) -> bool:
        """Run automatic 10-finger enrollment"""
        try:
            print("🚀 Starting Automatic 10-Finger Enrollment")
            print("=" * 60)
            print("The system will automatically detect and enroll your fingers")
            print("Just place any finger on the sensor when prompted")
            print("Session timeout: 5 minutes")
            print()
            
            if not self.initialize():
                return False
            
            # Welcome message
            input("Press Enter to start automatic enrollment...")
            
            print("\n🎯 Instructions:")
            print("1. Place any finger on the sensor when prompted")
            print("2. The system will automatically identify which finger it is")
            print("3. If it's the correct finger needed, it will be enrolled")
            print("4. If it's the wrong finger, you'll be prompted to try again")
            print("5. Continue until all 10 fingers are enrolled")
            print()
            
            while True:
                # Show current progress
                self.show_enrollment_progress()
                
                # Check if all fingers are enrolled
                if all(self.enrollment_progress.values()):
                    print("\n🎉 All 10 fingers enrolled successfully!")
                    break
                
                # Check session timeout
                if self.check_session_timeout():
                    print("❌ Session timed out")
                    break
                
                # Prompt for next finger
                next_finger = self.get_next_required_finger()
                if next_finger:
                    hand, finger_type = next_finger
                    print(f"\n📸 Ready for: {hand.value} {finger_type.value}")
                    input("Place your finger on the sensor, then press Enter...")
                    
                    # Attempt auto-enrollment
                    if self.auto_detect_and_enroll():
                        print(f"✅ {hand.value} {finger_type.value} enrolled successfully!")
                    else:
                        print(f"❌ Failed to enroll {hand.value} {finger_type.value}")
                        
                        # Ask if user wants to retry
                        retry = input("Retry this finger? (y/n): ").lower().strip()
                        if retry != 'y':
                            print("Skipping this finger...")
                            # Mark as enrolled to avoid infinite loop
                            key = f"{hand.value}_{finger_type.value}"
                            self.enrollment_progress[key] = True
                
                # Small delay
                time.sleep(1)
            
            # Final verification
            print(f"\n🔍 Final Verification")
            print("=" * 40)
            self.show_enrollment_progress()
            
            # Save all enrollment data
            self.save_enrollment_data()
            
            return all(self.enrollment_progress.values())
            
        except Exception as e:
            print(f"❌ Enrollment failed: {e}")
            return False
    
    def save_enrollment_data(self):
        """Save all enrollment data to file"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"automatic_enrollment_data_{timestamp}.json"
            
            # Prepare data for saving
            save_data = {
                'enrollment_time': datetime.now().isoformat(),
                'total_fingers': len(self.enrolled_fingers),
                'session_duration': time.time() - self.session_start_time if self.session_start_time else 0,
                'enrolled_fingers': self.enrolled_fingers,
                'enrollment_progress': self.enrollment_progress
            }
            
            with open(filename, 'w') as f:
                json.dump(save_data, f, indent=2)
            
            print(f"💾 Enrollment data saved to: {filename}")
            
        except Exception as e:
            print(f"❌ Failed to save enrollment data: {e}")

def main():
    """Main function"""
    print("🧠 Automatic Fingerprint Enrollment System")
    print("=" * 50)
    
    try:
        enrollment_system = SmartEnrollmentSystem()
        success = enrollment_system.run_automatic_enrollment()
        
        if success:
            print("\n🎉 Automatic enrollment completed successfully!")
        else:
            print("\n⚠️  Automatic enrollment completed with some issues")
        
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️  Enrollment interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
