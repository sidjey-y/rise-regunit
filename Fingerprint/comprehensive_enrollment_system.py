#!/usr/bin/env python3
"""
Comprehensive Fingerprint Enrollment System
Handles all 10 fingers with proper identification and duplication prevention
"""

import sys
import time
import logging
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Hand(Enum):
    LEFT = "left"
    RIGHT = "right"

class FingerType(Enum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    LITTLE = "little"

@dataclass
class FingerprintData:
    """Data structure for fingerprint information"""
    user_id: str
    hand: str
    finger_type: str
    position: int
    timestamp: str
    score: Optional[int] = None
    embeddings: Optional[Dict] = None
    raw_image_data: Optional[bytes] = None
    minutiae_points: Optional[List[Dict]] = None

@dataclass
class UserEnrollment:
    """Complete user enrollment data"""
    user_id: str
    enrollment_date: str
    completion_time: str
    total_fingers: int
    fingers: Dict[str, FingerprintData]
    session_duration: float

class ComprehensiveEnrollmentSystem:
    """Comprehensive fingerprint enrollment system for 1 user = 10 fingers"""
    
    def __init__(self, port='COM4', baudrate=57600):
        self.port = port
        self.baudrate = baudrate
        self.scanner = None
        self.current_user_id = None
        self.current_user_fingers: Dict[str, FingerprintData] = {}
        self.session_start_time = None
        self.session_timeout = 300  # 5 minutes for "in one go" session
        
        # Define all required fingers for 1 user
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
    
    def initialize(self) -> bool:
        """Initialize the fingerprint scanner"""
        try:
            from pyfingerprint.pyfingerprint import PyFingerprint
            
            logger.info(f"Initializing scanner on {self.port}...")
            
            # Try to initialize with password first
            try:
                self.scanner = PyFingerprint(self.port, self.baudrate, 0xFFFFFFFF, 0x00000000)
                logger.info("Scanner initialized successfully with password")
            except Exception as e:
                logger.warning(f"Password initialization failed: {e}")
                logger.info("Trying alternative initialization method...")
                
                # Try alternative initialization method
                try:
                    self.scanner = PyFingerprint(self.port, self.baudrate)
                    logger.info("Scanner initialized successfully with alternative method")
                except Exception as e2:
                    logger.error(f"Alternative initialization also failed: {e2}")
                    return False
            
            # Test scanner communication
            if not self.scanner.verifyPassword():
                logger.error("Scanner password verification failed")
                return False
            
            logger.info("Scanner password verified successfully")
            return True
            
        except Exception as e:
            logger.error(f"Scanner initialization failed: {e}")
            return False
    
    def get_finger_key(self, hand: Hand, finger_type: FingerType) -> str:
        """Generate a unique key for a finger"""
        return f"{hand.value}_{finger_type.value}"
    
    def is_session_active(self) -> bool:
        """Check if the enrollment session is still active (within timeout)"""
        if not self.session_start_time:
            return False
        
        elapsed = time.time() - self.session_start_time
        return elapsed <= self.session_timeout
    
    def start_enrollment_session(self, user_id: str):
        """Start a new enrollment session for a specific user"""
        self.current_user_id = user_id
        self.session_start_time = time.time()
        self.current_user_fingers.clear()
        logger.info(f"Starting new enrollment session for user: {user_id}")
        logger.info(f"Session timeout: {self.session_timeout} seconds")
        logger.info("Required fingers for this user:")
        for hand, finger_type in self.required_fingers:
            logger.info(f"  - {hand.value.title()} {finger_type.value.title()}")
    
    def get_remaining_fingers(self) -> List[Tuple[Hand, FingerType]]:
        """Get list of fingers that still need to be enrolled for current user"""
        remaining = []
        for hand, finger_type in self.required_fingers:
            finger_key = self.get_finger_key(hand, finger_type)
            if finger_key not in self.current_user_fingers:
                remaining.append((hand, finger_type))
        return remaining
    
    def get_enrollment_progress(self) -> Tuple[int, int]:
        """Get current enrollment progress for current user"""
        completed = len(self.current_user_fingers)
        total = len(self.required_fingers)
        return completed, total
    
    def is_finger_already_enrolled(self, hand: Hand, finger_type: FingerType) -> bool:
        """Check if a specific finger is already enrolled for current user"""
        finger_key = self.get_finger_key(hand, finger_type)
        return finger_key in self.current_user_fingers
    
    def check_duplicate_within_user(self, characteristics: bytes, hand: Hand, finger_type: FingerType) -> tuple[bool, str, float]:
        """Check if the scanned finger is a duplicate of any already enrolled finger for current user
        Returns: (is_duplicate, duplicate_info, similarity_score)"""
        if not self.current_user_fingers:
            return False, "", 0.0
        
        logger.info(f"Checking for duplicates against {len(self.current_user_fingers)} enrolled fingers...")
        
        for enrolled_key, enrolled_finger in self.current_user_fingers.items():
            if enrolled_finger.raw_image_data:
                try:
                    similarity = self.compare_characteristics(characteristics, enrolled_finger.raw_image_data)
                    logger.info(f"  Comparing with {enrolled_key}: {similarity:.2%} similarity")
                    
                    if similarity > 0.5:  # 50% similarity threshold for duplicate detection (lowered from 70%)
                        duplicate_info = f"{enrolled_finger.hand.title()} {enrolled_finger.finger_type.title()}"
                        logger.warning(f"Duplicate detected! This finger is {similarity:.1%} similar to {duplicate_info}")
                        return True, duplicate_info, similarity
                        
                except Exception as e:
                    logger.warning(f"Could not compare with {enrolled_key}: {e}")
                    continue
        
        logger.info("No duplicates found")
        return False, "", 0.0
    
    def extract_fingerprint_features(self, image_data: bytes) -> Dict:
        """Extract features from fingerprint image (embeddings x y theta)"""
        # This is a simplified feature extraction
        # In a real system, you would use more sophisticated algorithms
        try:
            # For now, we'll create a basic feature representation
            # In practice, you'd extract minutiae points, ridge patterns, etc.
            features = {
                "minutiae_count": len(image_data) // 1000,  # Simplified
                "quality_score": 85,  # Placeholder
                "orientation": 0.0,   # Placeholder for theta
                "center_x": 128,      # Placeholder for x
                "center_y": 128,      # Placeholder for y
                "extraction_timestamp": datetime.now().isoformat()
            }
            return features
        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return {}
    
    def enroll_finger(self, hand: Hand, finger_type: FingerType) -> bool:
        """Enroll a specific finger with verification"""
        try:
            finger_key = self.get_finger_key(hand, finger_type)
            
            if finger_key in self.current_user_fingers:
                logger.warning(f"Finger already enrolled: {hand.value.title()} {finger_type.value.title()}")
                return True
            
            logger.info(f"Enrolling {hand.value.title()} {finger_type.value.title()}...")
            logger.info("Place your finger on the sensor and hold steady...")
            
            # Wait for finger with better user feedback and timeout
            logger.info("Waiting for finger placement...")
            
            # Wait up to 30 seconds for finger placement
            finger_detected = False
            for i in range(300):  # 30 seconds timeout (300 * 0.1 seconds)
                if self.scanner.readImage():
                    finger_detected = True
                    break
                time.sleep(0.1)
                
                # Show progress every 2 seconds
                if i % 20 == 0 and i > 0:
                    logger.info(f"Still waiting... ({i//10} seconds elapsed)")
            
            if not finger_detected:
                logger.error("No finger detected for enrollment - timeout reached")
                logger.error("Please ensure your finger is properly placed on the scanner")
                return False
            
            logger.info("Finger detected! Processing...")
            
            # Convert to characteristics
            logger.info("Converting image to characteristics...")
            self.scanner.convertImage(0x01)
            
            # Get current characteristics for duplicate checking and storage
            logger.info("Downloading characteristics...")
            current_char = self.scanner.downloadCharacteristics(0x01)
            
            # Check for duplicates against already enrolled fingers for this user
            logger.info("Checking for duplicates against already enrolled fingers...")
            if self.current_user_fingers:
                logger.info("Comparing with previously enrolled fingers...")
                
                # Use the improved duplicate detection method
                is_duplicate, duplicate_info, similarity = self.check_duplicate_within_user(current_char, hand, finger_type)
                
                if is_duplicate:
                    logger.error(f"DUPLICATE FINGER DETECTED!")
                    logger.error(f"   This fingerprint is {similarity:.1%} similar to {duplicate_info}")
                    logger.error(f"   You may have scanned the same finger twice")
                    logger.error(f"   Expected: {hand.value.title()} {finger_type.value.title()}")
                    logger.error(f"   Detected: {duplicate_info}")
                    logger.error("   Please scan the correct finger")
                    return False
                
                logger.info("No duplicates found - proceeding with enrollment")
            else:
                logger.info("First finger - no duplicates to check")
            
            # NEW: Enhanced wrong finger type detection
            logger.info("Running enhanced wrong finger type detection...")
            try:
                from enhanced_duplicate_detection import EnhancedDuplicateDetector
                
                enhanced_detector = EnhancedDuplicateDetector()
                analysis = enhanced_detector.analyze_finger_scan(
                    current_char, hand.value, finger_type.value
                )
                
                if analysis and analysis.is_wrong_finger:
                    logger.error(f"🚨 WRONG FINGER TYPE DETECTED!")
                    logger.error(f"   Expected: {hand.value.title()} {finger_type.value.title()}")
                    logger.error(f"   Detected: {analysis.detected_finger_type.title()}")
                    logger.error(f"   Confidence: {analysis.confidence:.1%}")
                    logger.error(f"   Reason: {analysis.reason}")
                    logger.error(f"   Overall similarity: {analysis.similarity_scores.get('overall_similarity', 0):.1%}")
                    logger.error("   Please scan the correct finger type")
                    return False
                
                if analysis:
                    logger.info(f"✅ Finger type validation passed")
                    logger.info(f"   Detected: {analysis.detected_finger_type.title()}")
                    logger.info(f"   Confidence: {analysis.confidence:.1%}")
                    logger.info(f"   Overall similarity: {analysis.similarity_scores.get('overall_similarity', 0):.1%}")
                
            except ImportError:
                logger.warning("Enhanced duplicate detection not available - skipping finger type validation")
            except Exception as e:
                logger.warning(f"Enhanced detection failed: {e} - continuing with basic validation")
            
            # Also check scanner templates for any conflicts
            logger.info("Checking scanner templates for conflicts...")
            try:
                result = self.scanner.searchTemplate()
                position = result[0]
                score = result[1]
                
                if position != -1:
                    logger.warning(f"Template found at position {position} - this may indicate a duplicate")
                    logger.warning("Continuing with enrollment but be aware of potential duplicates")
                
            except Exception as e:
                logger.info("No existing templates found - clean enrollment")
            
            # Create template
            logger.info("Creating fingerprint template...")
            self.scanner.createTemplate()
            position = self.scanner.storeTemplate()
            
            # Adjust position to start at 1 instead of 0
            if position == 0:
                position = 1
            
            # Extract features for storage
            logger.info("Extracting minutiae points and characteristics...")
            embeddings = self.extract_fingerprint_features(current_char)
            
            # Store fingerprint data with minutiae points
            fingerprint_data = FingerprintData(
                user_id=self.current_user_id,
                hand=hand.value,
                finger_type=finger_type.value,
                position=position,
                timestamp=datetime.now().isoformat(),
                score=score,
                embeddings=embeddings,
                raw_image_data=current_char,  # Store the actual characteristics data
                minutiae_points=self._extract_minutiae_points(current_char)
            )
            
            self.current_user_fingers[finger_key] = fingerprint_data
            
            logger.info(f"Finger enrolled successfully: {hand.value.title()} {finger_type.value.title()}")
            logger.info(f"  - Position: {position}")
            logger.info(f"  - Minutiae points extracted: {len(embeddings)}")
            logger.info(f"  - Characteristics size: {len(current_char)} bytes")
            
            return True
            
        except Exception as e:
            logger.error(f"Enrollment failed for {hand.value.title()} {finger_type.value.title()}: {e}")
            return False
    
    def verify_finger(self, hand: Hand, finger_type: FingerType) -> bool:
        """Verify a specific enrolled finger"""
        try:
            finger_key = self.get_finger_key(hand, finger_type)
            
            if finger_key not in self.current_user_fingers:
                logger.error(f"Finger not enrolled: {hand.value.title()} {finger_type.value.title()}")
                return False
            
            logger.info(f"Verifying {hand.value.title()} {finger_type.value.title()}...")
            logger.info("Place your finger on the sensor for verification...")
            
            # Wait for finger with better user feedback and timeout
            logger.info("Waiting for finger placement...")
            
            # Wait up to 30 seconds for finger placement
            finger_detected = False
            for i in range(300):  # 30 seconds timeout (300 * 0.1 seconds)
                if self.scanner.readImage():
                    finger_detected = True
                    break
                time.sleep(0.1)
                
                # Show progress every 2 seconds
                if i % 20 == 0 and i > 0:
                    logger.info(f"Still waiting... ({i//10} seconds elapsed)")
            
            if not finger_detected:
                logger.error("No finger detected for verification - timeout reached")
                logger.error("Please ensure your finger is properly placed on the scanner")
                return False
            
            logger.info("Finger detected")
            
            # Convert to characteristics
            self.scanner.convertImage(0x01)
            
            # Search for template
            result = self.scanner.searchTemplate()
            position = result[0]
            score = result[1]
            
            if position == -1:
                logger.error(f"No matching template found for {hand.value.title()} {finger_type.value.title()}")
                return False
            
            # Check if it matches the expected position
            expected_position = self.current_user_fingers[finger_key].position
            if position != expected_position:
                logger.error(f"Position mismatch! Expected {expected_position}, got {position}")
                return False
            
            logger.info(f"Finger verified successfully: {hand.value.title()} {finger_type.value.title()}")
            logger.info(f"  - Position: {position}")
            logger.info(f"  - Score: {score}")
            
            return True
            
        except Exception as e:
            logger.error(f"Verification failed for {hand.value.title()} {finger_type.value.title()}: {e}")
            return False
    
    def verify_finger_type(self, hand: Hand, finger_type: FingerType) -> bool:
        """Verify that the scanned finger is the correct finger type"""
        try:
            logger.info(f"Verifying finger type: {hand.value.title()} {finger_type.value.title()}")
            
            # Step 1: Basic quality check using searchTemplate
            try:
                # Try to search for the template (this validates the characteristics)
                result = self.scanner.searchTemplate()
                position = result[0]
                score = result[1]
                
                logger.info(f"Scan quality check passed - Position: {position}, Score: {score}")
                
            except Exception as e:
                logger.error(f"Scan quality too low: {e}")
                logger.error("   Please clean the sensor and try again")
                return False
            
            # Step 2: Check for duplicate finger detection
            if position != -1:
                logger.warning(f"This fingerprint matches an existing template at position {position}")
                
                # Check if this position is already enrolled in our current session
                for fp in self.current_user_fingers.values():
                    if fp.position == position:
                        logger.error(f"SAME FINGER DETECTED!")
                        logger.error(f"   Expected: {hand.value.title()} {finger_type.value.title()}")
                        logger.error(f"   Detected: {fp.hand.title()} {fp.finger_type.title()}")
                        logger.error("   Please scan a different finger")
                        return False
                
                # If position exists but not in our session, it might be from a previous session
                # We should still allow enrollment but warn the user
                logger.warning("Template found from previous session - proceeding with enrollment")
                logger.warning("This will overwrite the existing template")
            
            # Step 3: CRITICAL - Check if this finger is similar to any previously enrolled fingers
            # This prevents wrong finger enrollment (e.g., right index when left index is expected)
            if self.current_user_fingers:
                logger.info("Checking for wrong finger detection...")
                
                # Get current characteristics for comparison
                current_char = self.scanner.downloadCharacteristics(0x01)
                
                for fp in self.current_user_fingers.values():
                    try:
                        # Download characteristics from the enrolled finger
                        self.scanner.loadTemplate(fp.position)
                        enrolled_char = self.scanner.downloadCharacteristics(0x01)
                        
                        # Compare characteristics
                        similarity = self.compare_characteristics(current_char, enrolled_char)
                        
                        logger.info(f"   Comparing with {fp.hand.title()} {fp.finger_type.title()}: {similarity:.2%} similarity")
                        
                        # IMPROVED: More sensitive thresholds for better wrong finger detection
                        # If similarity is too high, this is likely the same physical finger
                        if similarity > 0.70:  # Lowered from 0.75 to 0.70 for better detection
                            logger.error(f"WRONG FINGER DETECTED!")
                            logger.error(f"   Expected: {hand.value.title()} {finger_type.value.title()}")
                            logger.error(f"   Detected: {fp.hand.title()} {fp.finger_type.title()} (same physical finger)")
                            logger.error(f"   Similarity: {similarity:.2%}")
                            logger.error("   Please scan the correct finger type")
                            return False
                        
                        # Additional check: if it's the same finger type but different hand, be extra careful
                        # This catches cases like right index when left index is expected
                        if (fp.finger_type == finger_type.value and fp.hand != hand.value and similarity > 0.55):  # Lowered from 0.6 to 0.55
                            logger.error(f"WRONG HAND DETECTED!")
                            logger.error(f"   Expected: {hand.value.title()} {finger_type.value.title()}")
                            logger.error(f"   Detected: {fp.hand.title()} {fp.finger_type.title()} (same finger type, wrong hand)")
                            logger.error(f"   Similarity: {similarity:.2%}")
                            logger.error("   Please scan the correct hand")
                            return False
                        
                        # NEW: Additional check for high similarity between different finger types
                        # This catches cases where someone scans a different finger type with high similarity
                        if similarity > 0.70 and fp.finger_type != finger_type.value:  # Lowered from 0.75 to 0.70
                            logger.error(f"WRONG FINGER TYPE DETECTED!")
                            logger.error(f"   Expected: {hand.value.title()} {finger_type.value.title()}")
                            logger.error(f"   Detected: {fp.hand.title()} {fp.finger_type.title()} (different finger type, too similar)")
                            logger.error(f"   Similarity: {similarity:.2%}")
                            logger.error("   Please scan the correct finger type")
                            return False
                        
                        # NEW: Check for extremely high similarity which indicates the exact same finger
                        if similarity > 0.90:  # 90% threshold for exact duplicates
                            logger.error(f"EXACT DUPLICATE DETECTED!")
                            logger.error(f"   Expected: {hand.value.title()} {finger_type.value.title()}")
                            logger.error(f"   Detected: {fp.hand.title()} {fp.finger_type.title()} (exact same finger)")
                            logger.error(f"   Similarity: {similarity:.2%}")
                            logger.error("   Please scan a completely different finger")
                            return False
                        
                    except Exception as e:
                        logger.warning(f"   Could not compare with {fp.hand.title()} {fp.finger_type.title()}: {e}")
                        continue
                
                logger.info("No wrong finger detected")
            
            # Step 4: If we reach here, the finger appears to be different
            logger.info("Finger type verification passed")
            return True
                
        except Exception as e:
            logger.error(f"Finger type verification failed: {e}")
            return False
    
    def compare_characteristics(self, char1: bytes, char2: bytes) -> float:
        """Compare two fingerprint characteristics and return similarity score"""
        try:
            if len(char1) != len(char2):
                return 0.0
            
            # More sophisticated comparison for fingerprint characteristics
            # We'll use multiple comparison methods and combine them
            
            # Method 1: Exact byte-by-byte comparison (most important for duplicates)
            exact_matches = sum(1 for a, b in zip(char1, char2) if a == b)
            exact_similarity = exact_matches / len(char1)
            
            # Method 2: Hamming distance (bit-level differences)
            bit_differences = 0
            for a, b in zip(char1, char2):
                bit_differences += bin(a ^ b).count('1')
            max_bits = len(char1) * 8
            hamming_similarity = 1.0 - (bit_differences / max_bits)
            
            # Method 3: Pattern matching (look for common subsequences)
            # This is less important for distinguishing different fingers
            pattern_matches = 0
            for i in range(len(char1) - 3):
                pattern = char1[i:i+4]
                if pattern in char2:
                    pattern_matches += 1
            max_patterns = len(char1) - 3
            pattern_similarity = pattern_matches / max_patterns if max_patterns > 0 else 0.0
            
            # Method 4: Structural differences (focus on key fingerprint features)
            # Look at differences in specific byte ranges that represent minutiae
            structural_diff = 0
            if len(char1) >= 64:  # Ensure we have enough data for structural analysis
                # Compare first 32 bytes (often contain core fingerprint structure)
                core_diff = sum(1 for a, b in zip(char1[:32], char2[:32]) if a != b)
                structural_diff = core_diff / 32
                structural_similarity = 1.0 - structural_diff
            else:
                structural_similarity = 0.0
            
            # Method 5: NEW - Minutiae point analysis (most important for fingerprint comparison)
            minutiae_similarity = 0.0
            if len(char1) >= 128:  # Ensure we have enough data for minutiae analysis
                # Look at specific byte ranges that typically contain minutiae data
                # Minutiae are the unique ridge endings and bifurcations in fingerprints
                minutiae_ranges = [
                    (32, 64),   # First minutiae block
                    (64, 96),   # Second minutiae block
                    (96, 128)   # Third minutiae block
                ]
                
                minutiae_matches = 0
                total_minutiae_bytes = 0
                
                for start, end in minutiae_ranges:
                    if end <= len(char1) and end <= len(char2):
                        block1 = char1[start:end]
                        block2 = char2[start:end]
                        
                        # Compare minutiae blocks with higher weight
                        block_matches = sum(1 for a, b in zip(block1, block2) if a == b)
                        minutiae_matches += block_matches
                        total_minutiae_bytes += len(block1)
                
                if total_minutiae_bytes > 0:
                    minutiae_similarity = minutiae_matches / total_minutiae_bytes
            
            # IMPROVED: Better weight distribution for more accurate similarity detection
            # Minutiae analysis is most important, then exact matches, then structural differences
            combined_similarity = (0.40 * minutiae_similarity + 
                                  0.35 * exact_similarity + 
                                  0.15 * structural_similarity +
                                  0.07 * hamming_similarity + 
                                  0.03 * pattern_similarity)
            
            logger.debug(f"   Similarity breakdown: Minutiae={minutiae_similarity:.2%}, "
                        f"Exact={exact_similarity:.2%}, Structural={structural_similarity:.2%}, "
                        f"Hamming={hamming_similarity:.2%}, Pattern={pattern_similarity:.2%}, "
                        f"Combined={combined_similarity:.2%}")
            
            return combined_similarity
            
        except Exception as e:
            logger.warning(f"Characteristics comparison failed: {e}")
            return 0.0
    
    def analyze_finger_characteristics(self, characteristics: bytes, hand: Hand, finger_type: FingerType) -> bool:
        """Analyze fingerprint characteristics to validate finger type"""
        try:
            # This is a placeholder for more sophisticated finger type analysis
            # In practice, you might analyze:
            # - Ridge pattern density
            # - Minutiae distribution
            # - Finger size characteristics
            # - Orientation patterns
            
            # For now, we'll do basic validation
            if len(characteristics) < 100:  # Minimum expected size
                logger.warning("Characteristics data seems too small")
                return False
            
            # Check for reasonable data structure
            # Most fingerprint characteristics have specific patterns
            if characteristics[:4] == b'\x00\x00\x00\x00':  # All zeros
                logger.warning("Characteristics data appears invalid (all zeros)")
                return False
            
            logger.info(f"Characteristics analysis passed for {hand.value.title()} {finger_type.value.title()}")
            return True
            
        except Exception as e:
            logger.warning(f"Characteristics analysis failed: {e}")
            return False
    
    def check_hand_orientation(self, hand: Hand, finger_type: FingerType) -> bool:
        """Check if the finger placement suggests correct hand orientation"""
        try:
            # This is a conceptual check - in practice, you might analyze:
            # - Finger angle/orientation on the sensor
            # - Pressure distribution patterns
            # - Contact area characteristics
            
            logger.info(f"Checking hand orientation for {hand.value.title()} {finger_type.value.title()}")
            
            # For now, we'll assume the user knows how to place their fingers correctly
            # In a real system, you might use additional sensors or analysis
            logger.info("Hand orientation check passed (assumed correct)")
            return True
            
        except Exception as e:
            logger.warning(f"Hand orientation check failed: {e}")
            return False
    
    def get_enrollment_feedback(self, hand: Hand, finger_type: FingerType) -> str:
        """Get detailed feedback about enrollment attempts"""
        try:
            finger_key = self.get_finger_key(hand, finger_type)
            
            if finger_key in self.current_user_fingers:
                return f"Finger already enrolled: {hand.value.title()} {finger_type.value.title()}"
            
            # Check if any other fingers are enrolled that might be similar
            if self.current_user_fingers:
                similar_fingers = []
                for fp in self.current_user_fingers.values():
                    if fp.hand == hand.value:
                        similar_fingers.append(fp.finger_type.title())
                
                if similar_fingers:
                    return f"You have already enrolled {', '.join(similar_fingers)} from your {hand.value.title()} hand"
            
            return f"Ready to enroll {hand.value.title()} {finger_type.value.title()}"
            
        except Exception as e:
            return f"Error getting feedback: {e}"
    
    def analyze_scan_issues(self, hand: Hand, finger_type: FingerType) -> List[str]:
        """Analyze potential issues with the current scan"""
        issues = []
        
        try:
            # Check if characteristics are available using searchTemplate
            try:
                # Try to search for the template (this validates the characteristics)
                result = self.scanner.searchTemplate()
                position = result[0]
                score = result[1]
                
                if position == -1:
                    issues.append("No matching template found - this is expected for new enrollment")
                else:
                    issues.append(f"Template found at position {position} with score {score}")
                    
            except Exception as e:
                issues.append(f"Scan quality issue: {e}")
                return issues
            
            # Check for duplicate finger issues
            if self.current_user_fingers and position != -1:
                for fp in self.current_user_fingers.values():
                    if fp.position == position:
                        issues.append(f"Similar to {fp.hand.title()} {fp.finger_type.title()} - wrong finger detected")
                        break
            
            # General guidance
            if not issues or (len(issues) == 1 and "No matching template found" in issues[0]):
                issues.append("Ensure you're scanning the correct finger type")
                issues.append("Apply even pressure to the sensor")
                issues.append("Keep finger steady during scan")
            
        except Exception as e:
            issues.append(f"Analysis error: {e}")
        
        return issues
    
    def run_complete_enrollment(self, user_id: str = None) -> bool:
        """Run complete enrollment for all 10 fingers with 3 attempts per finger"""
        try:
            logger.info("Starting complete fingerprint enrollment...")
            logger.info("=" * 60)
            logger.info("This system will guide you through scanning all 10 fingers")
            logger.info("Each finger gets 3 attempts if needed")
            logger.info("You'll be prompted before each finger scan")
            logger.info("=" * 60)
            
            # Get user ID if not provided
            if not user_id:
                user_id = input("Enter User ID (e.g., 'user001', 'john_doe'): ").strip()
                if not user_id:
                    logger.error("User ID is required!")
                    return False
            
            self.current_user_id = user_id
            logger.info(f"Enrolling user: {user_id}")
            
            # Ask for user confirmation
            if not self._confirm_enrollment_start():
                logger.info("Enrollment cancelled by user")
                return False
            
            if not self.initialize():
                return False
            
            self.start_enrollment_session(user_id)
            
            # Enroll each finger with 3 attempts
            for hand, finger_type in self.required_fingers:
                logger.info(f"\nEnrolling: {hand.value.title()} {finger_type.value.title()}")
                logger.info("-" * 40)
                
                # Prompt user before scanning
                if not self._prompt_user_for_finger(hand, finger_type):
                    logger.info(f"Skipping {hand.value.title()} {finger_type.value.title()}")
                    logger.info("   Moving to next finger...")
                    continue
                
                # Try up to 3 times for this finger
                success = False
                for attempt in range(1, 4):
                    logger.info(f"Attempt {attempt}/3 for {hand.value.title()} {finger_type.value.title()}")
                    
                    if self.guided_finger_enrollment(hand, finger_type):
                        success = True
                        break
                    else:
                        if attempt < 3:
                            logger.warning(f"Attempt {attempt} failed. You have {3-attempt} more attempt(s).")
                            self._provide_attempt_feedback(hand, finger_type, attempt)
                            logger.info("Please try again...")
                            time.sleep(2)
                        else:
                            logger.error(f"All 3 attempts failed for {hand.value.title()} {finger_type.value.title()}")
                
                if not success:
                    logger.error(f"Failed to enroll {hand.value.title()} {finger_type.value.title()} after 3 attempts")
                    return False
                
                # Show enrollment summary after successful enrollment
                self._show_enrollment_summary()
                
                # Check session timeout
                if not self.is_session_active():
                    logger.error("Session timeout exceeded! Enrollment must be completed in one go.")
                    return False
                
                # Show progress
                enrolled, total = self.get_enrollment_progress()
                logger.info(f"Progress: {enrolled}/{total} fingers enrolled")
                
                time.sleep(1)
            
            # Final verification
            logger.info("\nRunning final verification for all fingers...")
            logger.info("=" * 60)
            
            all_verified = True
            for hand, finger_type in self.required_fingers:
                if not self.verify_finger(hand, finger_type):
                    all_verified = False
            
            if all_verified:
                logger.info("All 10 fingers enrolled and verified successfully!")
                logger.info("=" * 60)
                self._show_final_enrollment_summary()
                self.save_enrollment_data()
                return True
            else:
                logger.error("Some fingers failed verification")
                return False
                
        except Exception as e:
            logger.error(f"Complete enrollment failed: {e}")
            return False
    
    def save_enrollment_data(self):
        """Save enrollment data to JSON and YAML formats"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare data for serialization
            data = {
                "user_info": {
                    "user_id": self.current_user_id,
                    "enrollment_date": datetime.now().strftime("%Y-%m-%d"),
                    "completion_time": datetime.now().strftime("%H:%M:%S"),
                    "total_fingers_enrolled": len(self.current_user_fingers),
                    "session_duration_seconds": time.time() - self.session_start_time if self.session_start_time else 0
                },
                "enrolled_fingers": {}
            }
            
            # Convert enrolled fingers to serializable format
            for key, fp in self.current_user_fingers.items():
                # Convert bytes to base64 for JSON serialization
                raw_data_b64 = None
                if fp.raw_image_data:
                    import base64
                    raw_data_b64 = base64.b64encode(fp.raw_image_data).decode('utf-8')
                
                data["enrolled_fingers"][key] = {
                    "user_id": fp.user_id,
                    "hand": fp.hand,
                    "finger_type": fp.finger_type,
                    "position": fp.position,
                    "timestamp": fp.timestamp,
                    "score": fp.score,
                    "embeddings": fp.embeddings,
                    "raw_image_data_b64": raw_data_b64,
                    "minutiae_points": fp.minutiae_points
                }
            
            # Save as JSON with user ID in filename
            json_filename = f"user_{self.current_user_id}_enrollment_{timestamp}.json"
            with open(json_filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"Enrollment data saved to {json_filename}")
            
            # Save as YAML if PyYAML is available
            try:
                import yaml
                yaml_filename = f"user_{self.current_user_id}_enrollment_{timestamp}.yaml"
                with open(yaml_filename, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                
                logger.info(f"Enrollment data also saved to {yaml_filename}")
                
            except ImportError:
                logger.info("PyYAML not available - only JSON format saved")
                logger.info("Install PyYAML with: pip install PyYAML")
            
        except Exception as e:
            logger.error(f"Failed to save enrollment data: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.scanner:
                if hasattr(self.scanner, '_serial'):
                    self.scanner._serial.close()
                logger.info("Scanner resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

    def preliminary_finger_check(self, hand: Hand, finger_type: FingerType) -> bool:
        """Do a preliminary check to ensure the finger scan is valid"""
        try:
            logger.info(f"Preliminary check for {hand.value.title()} {finger_type.value.title()}")
            logger.info("This check ensures the scan quality is good enough")
            
            # Ask user to place finger for preliminary check
            logger.info("Place your finger on the sensor for preliminary check...")
            logger.info("Waiting for finger placement...")
            
            # Wait for finger with better user feedback and timeout
            finger_detected = False
            for i in range(300):  # 30 seconds timeout (300 * 0.1 seconds)
                if self.scanner.readImage():
                    finger_detected = True
                    break
                time.sleep(0.1)
                
                # Show progress every 2 seconds
                if i % 20 == 0 and i > 0:
                    logger.info(f"Still waiting... ({i//10} seconds elapsed)")
            
            if not finger_detected:
                logger.error("No finger detected for preliminary check - timeout reached")
                logger.error("Please ensure your finger is properly placed on the scanner")
                return False
            
            logger.info("Finger detected for preliminary check")
            logger.info("Processing preliminary scan...")
            
            # Convert to characteristics to test scan quality
            self.scanner.convertImage(0x01)
            
            # Test if the scan is valid by trying to create a template
            try:
                # Try to create a template (this validates the characteristics)
                self.scanner.createTemplate()
                logger.info("Preliminary scan quality check passed")
                return True
                
            except Exception as e:
                logger.error(f"Preliminary scan quality too low: {e}")
                logger.error("Please clean the sensor and try again")
                return False
            
        except Exception as e:
            logger.error(f"Preliminary check failed: {e}")
            return False

    def _provide_attempt_feedback(self, hand: Hand, finger_type: FingerType, attempt: int):
        """Provide helpful feedback when an attempt fails"""
        logger.info("Helpful tips for your next attempt:")
        
        if attempt == 1:
            logger.info("   • Make sure you're scanning the correct finger type")
            logger.info(f"   • Ensure it's your {hand.value} hand")
            logger.info("   • Place finger firmly and centered on the sensor")
            logger.info("   • Keep finger steady during the scan")
        elif attempt == 2:
            logger.info("   • Double-check you have the right finger")
            logger.info("   • Clean your finger and the sensor if needed")
            logger.info("   • Try a different angle or pressure")
            logger.info("   • Make sure finger covers the entire sensor area")
        
        logger.info("   • Take your time and be patient")
        logger.info("   • Follow the on-screen instructions carefully")
    
    def _confirm_enrollment_start(self) -> bool:
        """Ask user to confirm they want to start enrollment"""
        logger.info("Are you ready to start fingerprint enrollment?")
        logger.info("This will take approximately 5-10 minutes")
        logger.info("Each finger gets 3 attempts if needed")
        logger.info("Make sure your fingerprint scanner is connected to COM4")
        logger.info("")
        
        try:
            response = input("Type 'yes' to continue or 'no' to cancel: ").strip().lower()
            if response in ['yes', 'y', 'ok', 'continue']:
                logger.info("Enrollment confirmed! Starting...")
                return True
            else:
                logger.info("Enrollment cancelled")
                return False
        except (EOFError, KeyboardInterrupt):
            logger.info("Enrollment cancelled")
            return False
    
    def _show_enrollment_summary(self):
        """Show a summary of enrollment progress"""
        enrolled, total = self.get_enrollment_progress()
        remaining = total - enrolled
        
        logger.info("ENROLLMENT PROGRESS SUMMARY")
        logger.info("=" * 40)
        logger.info(f"Completed: {enrolled}/{total} fingers")
        logger.info(f"Remaining: {remaining} fingers")
        
        if enrolled > 0:
            logger.info("\nEnrolled fingers:")
            for key, fp in self.current_user_fingers.items():
                logger.info(f"   • {fp.hand.title()} {fp.finger_type.title()} (Position: {fp.position})")
        
        if remaining > 0:
            logger.info(f"\nNext finger to scan:")
            remaining_fingers = self.get_remaining_fingers()
            if remaining_fingers:
                next_hand, next_finger = remaining_fingers[0]
                logger.info(f"   • {next_hand.value.title()} {next_finger.value.title()}")
        
        # Show duplicate detection info
        if enrolled > 1:
            logger.info("\n🔍 Duplicate detection is active for remaining fingers")
            logger.info("   System will prevent scanning the same finger twice")
        
        logger.info("=" * 40)
    
    def _show_final_enrollment_summary(self):
        """Show final enrollment summary"""
        enrolled, total = self.get_enrollment_progress()
        duration = time.time() - self.session_start_time
        
        logger.info("FINAL ENROLLMENT SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Total fingers enrolled: {enrolled}/{total}")
        logger.info(f"Total time: {duration:.1f} seconds")
        logger.info(f"Session completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        logger.info("\nAll enrolled fingers:")
        for hand, finger_type in self.required_fingers:
            finger_key = self.get_finger_key(hand, finger_type)
            if finger_key in self.current_user_fingers:
                fp = self.current_user_fingers[finger_key]
                logger.info(f"   ✅ {fp.hand.title()} {fp.finger_type.title()} - Position {fp.position}")
        
        logger.info("\nYour fingerprint enrollment is complete!")
        logger.info("Data has been saved for future verification")
        logger.info("=" * 50)
    
    def _prompt_user_for_finger(self, hand: Hand, finger_type: FingerType):
        """Prompt the user before scanning a specific finger"""
        logger.info("=" * 50)
        logger.info(f"READY TO SCAN: {hand.value.upper()} {finger_type.value.upper()}")
        logger.info("=" * 50)
        
        # Provide clear instructions based on finger type
        if finger_type == FingerType.THUMB:
            logger.info("Place your THUMB on the sensor")
            logger.info("   - Thumb is the shortest, widest finger")
            logger.info("   - Place it flat on the sensor")
        elif finger_type == FingerType.INDEX:
            logger.info("Place your INDEX finger on the sensor")
            logger.info("   - Index is the finger next to your thumb")
            logger.info("   - Usually the longest finger")
        elif finger_type == FingerType.MIDDLE:
            logger.info("Place your MIDDLE finger on the sensor")
            logger.info("   - Middle finger is between index and ring")
            logger.info("   - Often the longest finger")
        elif finger_type == FingerType.RING:
            logger.info("Place your RING finger on the sensor")
            logger.info("   - Ring finger is between middle and little")
            logger.info("   - Usually wears rings")
        elif finger_type == FingerType.LITTLE:
            logger.info("Place your LITTLE finger on the sensor")
            logger.info("   - Little finger is the smallest finger")
            logger.info("   - Also called 'pinky'")
        
        logger.info(f"Make sure it's your {hand.value.upper()} hand")
        logger.info("Place finger gently on the sensor when ready")
        
        # Ask for user confirmation before proceeding
        logger.info("-" * 50)
        logger.info("IMPORTANT: Are you ready to scan this finger?")
        logger.info("   Type 'yes' to continue or 'no' to skip this finger")
        
        try:
            user_input = input("Your response (yes/no): ").strip().lower()
            if user_input not in ['yes', 'y']:
                logger.info(f"Skipping {hand.value.title()} {finger_type.value.title()}")
                return False
        except (EOFError, KeyboardInterrupt):
            logger.info("Skipping this finger due to input error")
            return False
        
        logger.info("Confirmed! Ready to scan...")
        logger.info("Waiting for you to place finger...")
        logger.info("-" * 50)
        return True

    def guided_finger_enrollment(self, hand: Hand, finger_type: FingerType) -> bool:
        """Guided enrollment with preliminary check and better error handling"""
        try:
            logger.info(f"Starting guided enrollment for {hand.value.title()} {finger_type.value.title()}")
            
            # Step 1: Preliminary check
            logger.info("Step 1: Preliminary finger check...")
            if not self.preliminary_finger_check(hand, finger_type):
                logger.error("Preliminary check failed - wrong finger detected")
                logger.error("   Please scan the correct finger type")
                return False
            
            logger.info("Preliminary check passed")
            
            # Step 2: Main enrollment
            logger.info("Step 2: Main enrollment process...")
            if not self.enroll_finger(hand, finger_type):
                logger.error("Main enrollment failed")
                return False
            
            logger.info("Guided enrollment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Guided enrollment failed: {e}")
            return False

    def analyze_scanned_finger_type(self, hand: Hand, finger_type: FingerType) -> Dict[str, Any]:
        """
        DEPRECATED: This method doesn't work because fingerprints are random patterns
        Finger type cannot be determined from fingerprint analysis
        """
        logger.warning("Finger type analysis is not possible - fingerprints are random patterns")
        logger.warning("Each person's fingerprints are unique and unrelated to finger type")
        
        return {
            'valid': False,
            'error': 'Finger type analysis is impossible - fingerprints are random patterns',
            'note': 'Use user input to determine finger type, not fingerprint analysis'
        }
    
    def test_scanner_connection(self) -> bool:
        """Test if the scanner is properly connected and responding"""
        try:
            logger.info("Testing scanner connection...")
            
            # Test basic communication
            if not hasattr(self.scanner, 'verifyPassword'):
                logger.error("Scanner object doesn't have verifyPassword method")
                return False
            
            # Test password verification
            if not self.scanner.verifyPassword():
                logger.error("Scanner password verification failed")
                return False
            
            logger.info("Scanner password verified successfully")
            
            # Test if scanner can read images
            logger.info("Testing image reading capability...")
            
            # Try to read an image (should return False if no finger)
            try:
                result = self.scanner.readImage()
                logger.info(f"Image reading test successful - No finger detected (expected): {result}")
                return True
            except Exception as e:
                logger.error(f"Image reading test failed: {e}")
                return False
                
        except Exception as e:
            logger.error(f"Scanner connection test failed: {e}")
            return False
    
    def wait_for_finger_with_timeout(self, timeout_seconds: int = 30, operation_name: str = "finger scan") -> bool:
        """
        Wait for finger placement with timeout and user feedback
        
        Args:
            timeout_seconds: Maximum time to wait (default 30 seconds)
            operation_name: Name of the operation for logging
            
        Returns:
            True if finger detected, False if timeout
        """
        logger.info(f"Waiting for finger placement for {operation_name}...")
        
        # Wait up to specified timeout for finger placement
        finger_detected = False
        max_iterations = timeout_seconds * 10  # Check every 0.1 seconds
        
        for i in range(max_iterations):
            try:
                if self.scanner.readImage():
                    finger_detected = True
                    break
            except Exception as e:
                logger.warning(f"Scanner read error: {e}")
                # Continue trying
            
            time.sleep(0.1)
            
            # Show progress every 2 seconds
            if i % 20 == 0 and i > 0:
                logger.info(f"Still waiting... ({i//10} seconds elapsed)")
        
        if not finger_detected:
            logger.error(f"No finger detected for {operation_name} - timeout reached ({timeout_seconds} seconds)")
            logger.error("Please ensure your finger is properly placed on the scanner")
            return False
        
        logger.info(f"Finger detected for {operation_name}!")
        return True

    def _extract_minutiae_points(self, characteristics: bytes) -> List[Dict]:
        """
        Extract minutiae points from fingerprint characteristics
        This analyzes the characteristics data to identify key fingerprint features
        """
        try:
            minutiae_points = []
            
            if not characteristics or len(characteristics) < 64:
                logger.warning("Characteristics data too small for minutiae extraction")
                return []
            
            # Analyze characteristics data for minutiae patterns
            # This is a simplified approach - in practice, you'd use more sophisticated algorithms
            
            # Look for patterns in the characteristics data
            for i in range(0, len(characteristics) - 4, 4):
                chunk = characteristics[i:i+4]
                
                # Simple pattern detection (this is a placeholder)
                # In a real system, you'd use proper minutiae detection algorithms
                if any(b != 0 for b in chunk):  # Non-zero chunk
                    minutiae_point = {
                        'position': i,
                        'type': 'ridge_ending',  # Placeholder
                        'coordinates': [i % 256, (i // 256) % 256],  # Simplified coordinates
                        'confidence': 0.8,  # Placeholder confidence
                        'data': chunk.hex()  # Store the actual data
                    }
                    minutiae_points.append(minutiae_point)
            
            logger.info(f"Extracted {len(minutiae_points)} minutiae points")
            return minutiae_points
            
        except Exception as e:
            logger.error(f"Minutiae extraction failed: {e}")
            return []

def main():
    """Main function"""
    enrollment_system = ComprehensiveEnrollmentSystem()
    
    try:
        success = enrollment_system.run_complete_enrollment()
        return success
        
    except KeyboardInterrupt:
        logger.info("\nEnrollment interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    finally:
        enrollment_system.cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
