#!/usr/bin/env python3
"""
Simple Finger Enrollment System
Guides users through scanning all 10 fingers with validation
"""

import cv2
import numpy as np
import json
import os
from datetime import datetime
from simple_finger_detector import SimpleFingerDetector

class SimpleEnrollmentSystem:
    def __init__(self):
        """Initialize the enrollment system"""
        self.detector = SimpleFingerDetector()
        self.enrollment_data = {}
        self.scanned_fingers = set()
        
        # Define the 10 fingers to scan
        self.fingers_to_scan = [
            ('left', 'thumb'),
            ('left', 'index'),
            ('left', 'middle'),
            ('left', 'ring'),
            ('left', 'pinky'),
            ('right', 'thumb'),
            ('right', 'index'),
            ('right', 'middle'),
            ('right', 'ring'),
            ('right', 'pinky')
        ]
        
        self.current_finger_index = 0
        
    def start_enrollment(self):
        """Start the enrollment process"""
        print("🎯 Finger Enrollment System")
        print("=" * 40)
        print("You will be guided to scan all 10 fingers in sequence")
        print("The system will detect wrong fingers and duplicates")
        print()
        
        # Initialize camera
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            print("❌ Error: Could not open camera")
            return False
        
        print("📱 Camera opened successfully!")
        print("👆 Follow the on-screen instructions")
        print("🔴 Press 'q' to quit, 's' to scan current frame")
        print()
        
        self._run_enrollment_loop()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        
        # Save enrollment data
        self._save_enrollment_data()
        
        return True
    
    def _run_enrollment_loop(self):
        """Main enrollment loop"""
        while self.current_finger_index < len(self.fingers_to_scan):
            expected_hand, expected_finger = self.fingers_to_scan[self.current_finger_index]
            
            print(f"\n🔍 Scanning {expected_hand} {expected_finger}...")
            print(f"Progress: {self.current_finger_index + 1}/{len(self.fingers_to_scan)}")
            
            if self._scan_single_finger(expected_hand, expected_finger):
                self.current_finger_index += 1
            else:
                # User cancelled
                break
        
        if self.current_finger_index == len(self.fingers_to_scan):
            print("\n🎉 Enrollment completed successfully!")
            print(f"✅ All {len(self.fingers_to_scan)} fingers enrolled")
        else:
            print(f"\n⚠️ Enrollment incomplete: {self.current_finger_index}/{len(self.fingers_to_scan)} fingers")
    
    def _scan_single_finger(self, expected_hand: str, expected_finger: str) -> bool:
        """Scan a single finger with validation"""
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts:
            attempt += 1
            print(f"\n--- Attempt {attempt}/{max_attempts} ---")
            
            # Show instructions
            self._show_instructions(expected_hand, expected_finger)
            
            # Wait for user to position finger
            print("👆 Position your finger and press 's' to scan")
            print("   Press 'q' to quit, 'r' to retry")
            
            # Capture and validate
            if self._capture_and_validate(expected_hand, expected_finger):
                return True
            
            if attempt < max_attempts:
                print("🔄 Retrying...")
        
        print(f"❌ Failed to scan {expected_hand} {expected_finger} after {max_attempts} attempts")
        return False
    
    def _show_instructions(self, hand: str, finger: str):
        """Show finger-specific instructions"""
        instructions = {
            'thumb': "Place your thumb on the scanner - it's the largest and roundest finger",
            'index': "Place your index finger - it's the first finger next to your thumb",
            'middle': "Place your middle finger - it's the longest finger",
            'ring': "Place your ring finger - it's between middle and little finger",
            'pinky': "Place your little finger (pinky) - it's the smallest finger"
        }
        
        print(f"📋 Instructions for {hand} {finger}:")
        print(f"   • Use your {hand} hand")
        print(f"   • {instructions.get(finger, 'Place the finger on the scanner')}")
        print(f"   • Ensure the finger is centered and fully visible")
        print(f"   • Keep your finger steady")
    
    def _capture_and_validate(self, expected_hand: str, expected_finger: str) -> bool:
        """Capture frame and validate finger"""
        while True:
            ret, frame = self.cap.read()
            if not ret:
                print("❌ Error: Could not read frame")
                return False
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect finger
            result = self.detector.detect_finger_type(frame)
            
            # Draw landmarks and instructions
            if result['landmarks']:
                frame = self._draw_enrollment_ui(frame, result, expected_hand, expected_finger)
            
            # Display frame
            cv2.imshow('Finger Enrollment', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                return False
            elif key == ord('s') and result['finger_type']:
                return self._process_scan(result, expected_hand, expected_finger)
            elif key == ord('r'):
                return False  # Retry
    
    def _draw_enrollment_ui(self, frame: np.ndarray, result: dict, 
                           expected_hand: str, expected_finger: str) -> np.ndarray:
        """Draw enrollment UI on frame"""
        # Draw landmarks
        frame = self.detector.draw_landmarks(frame, result['landmarks'])
        
        # Draw current instruction
        current_hand, current_finger = self.fingers_to_scan[self.current_finger_index]
        cv2.putText(frame, f"Scan: {current_hand.title()} {current_finger.title()}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # Draw progress
        progress = f"Progress: {self.current_finger_index + 1}/{len(self.fingers_to_scan)}"
        cv2.putText(frame, progress, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        
        # Draw detected finger info
        if result['finger_type'] and result['hand_side']:
            detected_desc = self.detector.get_finger_description(
                result['finger_type'], result['hand_side']
            )
            cv2.putText(frame, f"Detected: {detected_desc}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show validation status
            validation = self.detector.validate_expected_finger(
                expected_finger, expected_hand,
                result['finger_type'], result['hand_side']
            )
            
            if validation['valid']:
                cv2.putText(frame, "✅ CORRECT FINGER!", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(frame, "❌ WRONG FINGER!", (10, 120),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Show what's wrong
                if not validation['finger_match']:
                    cv2.putText(frame, f"Expected: {expected_finger}, Got: {result['finger_type']}", 
                               (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                if not validation['hand_match']:
                    cv2.putText(frame, f"Expected: {expected_hand} hand, Got: {result['hand_side']} hand", 
                               (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Draw instructions
        cv2.putText(frame, "Press 's' to scan, 'r' to retry, 'q' to quit", 
                   (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    def _process_scan(self, result: dict, expected_hand: str, expected_finger: str) -> bool:
        """Process the scanned finger and validate"""
        if not result['finger_type'] or not result['hand_side']:
            print("❌ No finger detected, please try again")
            return False
        
        detected_hand = result['hand_side']
        detected_finger = result['finger_type']
        confidence = result['confidence']
        
        # Check for duplicates
        finger_key = f"{detected_hand}_{detected_finger}"
        if finger_key in self.scanned_fingers:
            print(f"❌ Duplicate detected! {detected_hand.title()} {detected_finger} already scanned")
            print("   Please scan the correct finger")
            return False
        
        # Validate against expected finger
        validation = self.detector.validate_expected_finger(
            expected_finger, expected_hand,
            detected_finger, detected_hand
        )
        
        if validation['valid']:
            # Success! Record the scan
            self.scanned_fingers.add(finger_key)
            self.enrollment_data[finger_key] = {
                'hand': detected_hand,
                'finger': detected_finger,
                'confidence': confidence,
                'timestamp': datetime.now().isoformat(),
                'validation_score': validation['score']
            }
            
            print(f"✅ Successfully scanned {detected_hand.title()} {detected_finger}")
            print(f"   Confidence: {confidence:.2f}")
            print(f"   Validation Score: {validation['score']:.2f}")
            return True
        
        else:
            # Wrong finger detected
            print(f"❌ Wrong finger detected!")
            print(f"   Expected: {expected_hand.title()} {expected_finger}")
            print(f"   Got: {detected_hand.title()} {detected_finger}")
            
            if not validation['finger_match']:
                print(f"   ❌ Wrong finger type! Expected: {expected_finger}, Got: {detected_finger}")
            if not validation['hand_match']:
                print(f"   ❌ Wrong hand! Expected: {expected_hand} hand, Got: {detected_hand} hand")
            
            print("   Please scan the correct finger")
            return False
    
    def _save_enrollment_data(self):
        """Save enrollment data to file"""
        if not self.enrollment_data:
            print("⚠️ No enrollment data to save")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"enrollment_data_{timestamp}.json"
        
        try:
            with open(filename, 'w') as f:
                json.dump(self.enrollment_data, f, indent=2)
            print(f"💾 Enrollment data saved to {filename}")
        except Exception as e:
            print(f"❌ Error saving enrollment data: {e}")
    
    def get_enrollment_summary(self):
        """Get summary of enrollment data"""
        if not self.enrollment_data:
            return "No fingers enrolled"
        
        summary = f"Enrollment Summary ({len(self.enrollment_data)}/10 fingers):\n"
        for finger_key, data in self.enrollment_data.items():
            summary += f"  ✅ {data['hand'].title()} {data['finger']} (Confidence: {data['confidence']:.2f})\n"
        
        return summary

def main():
    """Main function"""
    enrollment_system = SimpleEnrollmentSystem()
    
    try:
        success = enrollment_system.start_enrollment()
        if success:
            print("\n" + "=" * 40)
            print(enrollment_system.get_enrollment_summary())
    except KeyboardInterrupt:
        print("\n\n⚠️ Enrollment interrupted by user")
    except Exception as e:
        print(f"\n❌ Error during enrollment: {e}")

if __name__ == "__main__":
    main()
