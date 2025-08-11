#!/usr/bin/env python3
"""
Fingerprint Recognition System
Uses existing fingerprint_data directory as reference to identify finger types.
"""

import os
import sys
import cv2
import numpy as np
import time
import json
import base64
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
from pyfingerprint.pyfingerprint import PyFingerprint

class Hand(Enum):
    LEFT = "Left"
    RIGHT = "Right"

class FingerType(Enum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    LITTLE = "little"

@dataclass
class FingerprintReference:
    """Reference fingerprint data"""
    user_id: str
    hand: str
    finger_type: str
    filepath: str
    image_data: np.ndarray
    features: np.ndarray

class FingerprintRecognitionSystem:
    """Fingerprint recognition using reference database"""
    
    def __init__(self, reference_dir: str = "fingerprint_data", port='COM4', baudrate=57600):
        self.reference_dir = reference_dir
        self.port = port
        self.baudrate = baudrate
        self.scanner = None
        self.reference_database: Dict[str, List[FingerprintReference]] = {}
        self.feature_extractor = cv2.SIFT_create()
        
        # Load reference database
        self.load_reference_database()
    
    def load_reference_database(self):
        """Load all reference fingerprints from the database"""
        print("📚 Loading reference fingerprint database...")
        
        try:
            # Scan through all user directories
            for user_dir in os.listdir(self.reference_dir):
                user_path = os.path.join(self.reference_dir, user_dir)
                if not os.path.isdir(user_path):
                    continue
                
                fingerprint_path = os.path.join(user_path, "Fingerprint")
                if not os.path.exists(fingerprint_path):
                    continue
                
                print(f"   📁 Loading user {user_dir}...")
                user_references = []
                
                # Load all fingerprint files for this user
                for filename in os.listdir(fingerprint_path):
                    if filename.endswith('.BMP') or filename.endswith('.bmp'):
                        filepath = os.path.join(fingerprint_path, filename)
                        
                        # Parse filename to extract metadata
                        # Format: 1__M_Left_thumb_finger.BMP
                        parts = filename.replace('.BMP', '').replace('.bmp', '').split('_')
                        if len(parts) >= 4:
                            user_id = parts[0]
                            gender = parts[1]
                            hand = parts[2]
                            finger_type = parts[3]
                            
                            # Load and process the image
                            try:
                                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                                if image is not None:
                                    # Extract features
                                    keypoints, descriptors = self.feature_extractor.detectAndCompute(image, None)
                                    
                                    if descriptors is not None:
                                        # Create reference object
                                        reference = FingerprintReference(
                                            user_id=user_id,
                                            hand=hand,
                                            finger_type=finger_type,
                                            filepath=filepath,
                                            image_data=image,
                                            features=descriptors
                                        )
                                        user_references.append(reference)
                                        print(f"      ✅ {hand} {finger_type}: {len(descriptors)} features")
                                    else:
                                        print(f"      ⚠️  {hand} {finger_type}: No features extracted")
                                else:
                                    print(f"      ❌ {hand} {finger_type}: Failed to load image")
                                    
                            except Exception as e:
                                print(f"      ❌ {hand} {finger_type}: Error processing - {e}")
                
                if user_references:
                    self.reference_database[user_dir] = user_references
                    print(f"   ✅ User {user_dir}: {len(user_references)} fingerprints loaded")
                else:
                    print(f"   ⚠️  User {user_dir}: No valid fingerprints found")
            
            total_references = sum(len(refs) for refs in self.reference_database.values())
            print(f"\n📊 Database loaded: {len(self.reference_database)} users, {total_references} fingerprints")
            
        except Exception as e:
            print(f"❌ Failed to load reference database: {e}")
    
    def initialize_scanner(self) -> bool:
        """Initialize the fingerprint scanner"""
        try:
            print(f"🔌 Initializing scanner on {self.port}...")
            self.scanner = PyFingerprint(self.port, self.baudrate, 0xFFFFFFFF, 0x00000000)
            
            if self.scanner.verifyPassword():
                print("✅ Scanner initialized successfully")
                return True
            else:
                print("❌ Scanner password verification failed")
                return False
                
        except Exception as e:
            print(f"❌ Scanner initialization failed: {e}")
            return False
    
    def capture_current_fingerprint(self) -> Optional[np.ndarray]:
        """Capture current fingerprint from scanner"""
        if not self.scanner:
            print("❌ Scanner not initialized")
            return None
        
        try:
            print("📸 Place your finger on the sensor...")
            
            # Wait for finger
            while not self.scanner.readImage():
                time.sleep(0.1)
            
            print("   ✅ Finger detected")
            
            # Try to get image data
            try:
                # Method 1: Try to get raw image
                if hasattr(self.scanner, 'downloadImage'):
                    try:
                        # Try with parameter first (newer API)
                        image_data = self.scanner.downloadImage(0x01)
                    except TypeError:
                        try:
                            # Try without parameter (older API)
                            image_data = self.scanner.downloadImage()
                        except Exception as e:
                            print(f"   ⚠️  downloadImage failed: {e}")
                            image_data = None
                    
                    if image_data:
                        # Convert bytes to numpy array
                        # This is a simplified conversion - you may need to adjust based on your scanner
                        image_array = np.frombuffer(image_data, dtype=np.uint8)
                        # Reshape based on your scanner's image dimensions
                        # Common dimensions: 256x288, 256x256, etc.
                        try:
                            # Try common dimensions
                            for width, height in [(256, 288), (256, 256), (288, 256)]:
                                try:
                                    reshaped = image_array.reshape(height, width)
                                    print(f"   📸 Raw image captured: {width}x{height}")
                                    return reshaped
                                except ValueError:
                                    continue
                            
                            # If no standard dimensions work, try to guess
                            size = len(image_array)
                            # Find factors close to square
                            for i in range(int(np.sqrt(size)), 0, -1):
                                if size % i == 0:
                                    height, width = i, size // i
                                    if 200 <= width <= 400 and 200 <= height <= 400:
                                        reshaped = image_array.reshape(height, width)
                                        print(f"   📸 Raw image captured: {width}x{height} (estimated)")
                                        return reshaped
                            
                        except Exception as e:
                            print(f"   ⚠️  Image reshaping failed: {e}")
                
                # Method 2: Create template and try to extract image
                print("   🔄 Converting to template...")
                self.scanner.convertImage(0x01)
                self.scanner.createTemplate()
                
                # For now, return a placeholder
                print("   ⚠️  Raw image not available, using placeholder")
                return self._create_placeholder_image()
                
            except Exception as e:
                print(f"   ⚠️  Image capture failed: {e}")
                return self._create_placeholder_image()
                
        except Exception as e:
            print(f"❌ Fingerprint capture failed: {e}")
            return None
    
    def _create_placeholder_image(self) -> np.ndarray:
        """Create a placeholder image for testing"""
        # Create a 256x256 grayscale image with a simple pattern
        image = np.zeros((256, 256), dtype=np.uint8)
        
        # Add some patterns to simulate a fingerprint
        for i in range(0, 256, 20):
            image[i:i+10, :] = 128  # Horizontal lines
            image[:, i:i+10] = 128   # Vertical lines
        
        # Add some random noise
        noise = np.random.randint(0, 50, (256, 256), dtype=np.uint8)
        image = np.clip(image + noise, 0, 255)
        
        return image
    
    def identify_finger(self, current_image: np.ndarray, expected_hand: Hand, expected_finger: FingerType) -> Tuple[bool, str, float]:
        """
        Identify if the current fingerprint matches the expected finger type
        
        Returns:
            Tuple[bool, str, float]: (is_correct_finger, reason, confidence)
        """
        try:
            print(f"🔍 Identifying finger: Expected {expected_hand.value} {expected_finger.value}")
            
            # Extract features from current image
            keypoints, current_descriptors = self.feature_extractor.detectAndCompute(current_image, None)
            
            if current_descriptors is None:
                return False, "No features extracted from current image", 0.0
            
            print(f"   📊 Current image: {len(current_descriptors)} features")
            
            best_match = None
            best_score = 0.0
            best_reference = None
            
            # Compare with all reference fingerprints
            for user_id, references in self.reference_database.items():
                for reference in references:
                    if reference.features is None:
                        continue
                    
                    # Calculate similarity score
                    similarity = self._calculate_similarity(current_descriptors, reference.features)
                    
                    if similarity > best_score:
                        best_score = similarity
                        best_match = {
                            'user_id': user_id,
                            'hand': reference.hand,
                            'finger_type': reference.finger_type,
                            'similarity': similarity,
                            'filepath': reference.filepath
                        }
                        best_reference = reference
            
            if best_match is None:
                return False, "No matching fingerprints found in database", 0.0
            
            print(f"   🎯 Best match: User {best_match['user_id']} {best_match['hand']} {best_match['finger_type']}")
            print(f"   📈 Similarity score: {best_match['similarity']:.3f}")
            
            # Check if it's the correct finger type
            is_correct_hand = best_match['hand'].lower() == expected_hand.value.lower()
            is_correct_finger = best_match['finger_type'].lower() == expected_finger.value.lower()
            
            # Determine confidence thresholds
            if best_match['similarity'] < 0.3:
                return False, f"Fingerprint quality too low (score: {best_match['similarity']:.3f})", best_match['similarity']
            
            if is_correct_hand and is_correct_finger:
                if best_match['similarity'] >= 0.6:
                    return True, f"Correct finger confirmed (score: {best_match['similarity']:.3f})", best_match['similarity']
                else:
                    return False, f"Correct finger but low confidence (score: {best_match['similarity']:.3f})", best_match['similarity']
            
            elif is_correct_hand and not is_correct_finger:
                return False, f"Wrong finger type: Expected {expected_finger.value}, got {best_match['finger_type']}", best_match['similarity']
            
            elif not is_correct_hand and is_correct_finger:
                return False, f"Wrong hand: Expected {expected_hand.value}, got {best_match['hand']}", best_match['similarity']
            
            else:
                return False, f"Wrong finger: Expected {expected_hand.value} {expected_finger.value}, got {best_match['hand']} {best_match['finger_type']}", best_match['similarity']
                
        except Exception as e:
            print(f"❌ Finger identification failed: {e}")
            return False, f"Identification error: {e}", 0.0
    
    def _calculate_similarity(self, desc1: np.ndarray, desc2: np.ndarray) -> float:
        """Calculate similarity between two feature descriptors"""
        try:
            # Use FLANN matcher for fast matching
            FLANN_INDEX_KDTREE = 1
            index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
            search_params = dict(checks=50)
            
            flann = cv2.FlannBasedMatcher(index_params, search_params)
            matches = flann.knnMatch(desc1, desc2, k=2)
            
            # Apply ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.7 * n.distance:
                        good_matches.append(m)
            
            # Calculate similarity score
            if len(matches) == 0:
                return 0.0
            
            similarity = len(good_matches) / len(matches)
            return similarity
            
        except Exception as e:
            print(f"   ⚠️  Similarity calculation failed: {e}")
            return 0.0
    
    def test_recognition(self):
        """Test the fingerprint recognition system"""
        try:
            print("🧠 Testing Fingerprint Recognition System")
            print("=" * 60)
            
            if not self.initialize_scanner():
                return False
            
            print("\n📸 Testing finger recognition...")
            print("   Place your LEFT THUMB on the sensor...")
            
            # Capture current fingerprint
            current_image = self.capture_current_fingerprint()
            if current_image is None:
                print("❌ Failed to capture fingerprint")
                return False
            
            # Try to identify the finger
            is_correct, reason, confidence = self.identify_finger(
                current_image, Hand.LEFT, FingerType.THUMB
            )
            
            print(f"\n📊 Recognition Result:")
            print(f"   Correct finger: {'✅ YES' if is_correct else '❌ NO'}")
            print(f"   Reason: {reason}")
            print(f"   Confidence: {confidence:.3f}")
            
            if is_correct:
                print("\n🎉 Recognition successful! The system correctly identified your left thumb.")
            else:
                print("\n⚠️  Recognition failed. The system detected a different finger.")
            
            return True
            
        except Exception as e:
            print(f"❌ Test failed: {e}")
            return False

def main():
    """Main function"""
    print("🧠 Fingerprint Recognition System")
    print("=" * 50)
    
    try:
        recognition_system = FingerprintRecognitionSystem()
        success = recognition_system.test_recognition()
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

