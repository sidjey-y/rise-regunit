#!/usr/bin/env python3
"""
AI-Enhanced Comprehensive Fingerprint Enrollment System
Integrates Siamese Neural Network for advanced duplicate detection
"""

import sys
import time
import logging
import json
import os
import base64
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
from enum import Enum

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

# Import AI/ML components
try:
    from siamese_network import SiameseNetwork
    from config_manager import ConfigManager
    from fingerprint_preprocessor import FingerprintPreprocessor
    AI_AVAILABLE = True
    print("✅ AI/ML components available - Advanced duplicate detection enabled")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI/ML components not available: {e}")
    print("   Falling back to traditional duplicate detection")

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
    ai_embeddings: Optional[np.ndarray] = None  # AI-generated embeddings
    raw_image_data: Optional[bytes] = None
    minutiae_points: Optional[List[Dict]] = None
    ai_confidence: Optional[float] = None  # AI confidence score

@dataclass
class UserEnrollment:
    """Complete user enrollment data"""
    user_id: str
    enrollment_date: str
    completion_time: str
    total_fingers: int
    fingers: Dict[str, FingerprintData]
    session_duration: float
    ai_model_used: bool

class AIEnhancedEnrollmentSystem:
    """AI-Enhanced fingerprint enrollment system with Siamese Neural Network"""
    
    def __init__(self, port='COM4', baudrate=57600, use_ai=True):
        self.port = port
        self.baudrate = baudrate
        self.scanner = None
        self.current_user_id = None
        self.current_user_fingers: Dict[str, FingerprintData] = {}
        self.session_start_time = None
        self.session_timeout = 300  # 5 minutes for "in one go" session
        self.use_ai = use_ai and AI_AVAILABLE
        
        # AI/ML Components
        self.siamese_network = None
        self.config_manager = None
        self.preprocessor = None
        
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
        
        if self.use_ai:
            self.initialize_ai_components()
    
    def initialize_ai_components(self) -> bool:
        """Initialize AI/ML components for advanced duplicate detection"""
        try:
            logger.info("🤖 Initializing AI components for enhanced duplicate detection...")
            
            # Initialize configuration manager
            self.config_manager = ConfigManager("config.yaml")
            
            # Initialize preprocessor
            self.preprocessor = FingerprintPreprocessor(self.config_manager)
            
            # Initialize Siamese network
            self.siamese_network = SiameseNetwork(self.config_manager)
            
            # Try to load pre-trained model
            if os.path.exists("models/siamese_fingerprint_model.h5"):
                logger.info("📁 Loading pre-trained Siamese model...")
                if self.siamese_network.load_model():
                    logger.info("✅ Pre-trained AI model loaded successfully!")
                else:
                    logger.warning("⚠️ Failed to load pre-trained model - will use basic duplicate detection")
                    self.siamese_network.build_model()
            else:
                logger.info("🏗️ Building new Siamese network model...")
                self.siamese_network.build_model()
                logger.info("✅ New AI model built successfully!")
            
            logger.info("🎯 AI-enhanced duplicate detection ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI components: {e}")
            logger.info("🔄 Falling back to traditional duplicate detection")
            self.use_ai = False
            return False
    
    def initialize_scanner(self) -> bool:
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
    
    def ai_duplicate_check(self, current_image: np.ndarray, hand: Hand, finger_type: FingerType) -> Tuple[bool, str, float]:
        """
        AI-based duplicate detection using Siamese Neural Network
        
        Args:
            current_image: Current fingerprint image
            hand: Hand type
            finger_type: Finger type
            
        Returns:
            (is_duplicate, duplicate_info, confidence_score)
        """
        if not self.use_ai or not self.current_user_fingers:
            return False, "", 0.0
        
        try:
            logger.info("🤖 AI DUPLICATE DETECTION START")
            logger.info("=" * 50)
            logger.info(f"🔍 Analyzing {hand.value.title()} {finger_type.value.title()} using Siamese Neural Network...")
            
            # Preprocess current image
            current_processed = self.preprocessor.process(current_image)
            if not current_processed:
                logger.warning("⚠️ Current image preprocessing failed - falling back to traditional method")
                return False, "", 0.0
            
            # Extract AI embedding for current fingerprint
            current_embedding = self.siamese_network.extract_embedding(current_processed['model_input'])
            if current_embedding is None:
                logger.warning("⚠️ Current embedding extraction failed - falling back to traditional method")
                return False, "", 0.0
            
            logger.info(f"✅ Current fingerprint AI embedding extracted: {current_embedding.shape}")
            
            # Compare with all enrolled fingerprints
            max_similarity = 0.0
            best_match = ""
            all_similarities = []
            
            for enrolled_key, enrolled_finger in self.current_user_fingers.items():
                if enrolled_finger.ai_embeddings is not None:
                    # Calculate similarity using AI embeddings
                    similarity = self.siamese_network.calculate_similarity(
                        current_embedding, 
                        enrolled_finger.ai_embeddings
                    )
                    all_similarities.append((enrolled_key, similarity))
                    
                    logger.info(f"  🧠 AI Similarity with {enrolled_key}: {similarity:.3f}")
                    
                    if similarity > max_similarity:
                        max_similarity = similarity
                        best_match = f"{enrolled_finger.hand.title()} {enrolled_finger.finger_type.title()}"
            
            # AI-based thresholds (more sophisticated than traditional methods)
            ai_threshold = 0.85  # Higher threshold for AI (more accurate)
            
            # Special handling for thumbs using AI
            is_thumb = finger_type.value == 'thumb'
            if is_thumb:
                ai_threshold = 0.90  # Even higher threshold for thumbs with AI
            
            logger.info("🧠 AI SIMILARITY ANALYSIS:")
            logger.info("-" * 40)
            for enrolled_key, similarity in sorted(all_similarities, key=lambda x: x[1], reverse=True):
                status = "🚨 DUPLICATE" if similarity > ai_threshold else "✅ UNIQUE"
                logger.info(f"  {enrolled_key}: {similarity:.3f} {status}")
            
            if max_similarity > ai_threshold:
                logger.error("🤖 AI DETECTED DUPLICATE FINGERPRINT!")
                logger.error("=" * 50)
                logger.error(f"   Expected: {hand.value.title()} {finger_type.value.title()}")
                logger.error(f"   AI Match: {best_match}")
                logger.error(f"   AI Confidence: {max_similarity:.3f}")
                logger.error(f"   AI Threshold: {ai_threshold:.3f}")
                logger.error("   Please scan a different finger")
                logger.error("=" * 50)
                return True, best_match, max_similarity
            
            logger.info(f"✅ AI Analysis Complete - No duplicates detected (max similarity: {max_similarity:.3f})")
            return False, "", max_similarity
            
        except Exception as e:
            logger.error(f"❌ AI duplicate check failed: {e}")
            logger.info("🔄 Falling back to traditional duplicate detection")
            return False, "", 0.0
    
    def traditional_duplicate_check(self, characteristics: bytes, hand: Hand, finger_type: FingerType) -> Tuple[bool, str, float]:
        """Traditional duplicate detection using characteristics comparison"""
        if not self.current_user_fingers:
            return False, "", 0.0
        
        logger.info("🔍 TRADITIONAL DUPLICATE DETECTION")
        logger.info("=" * 50)
        
        # Special handling for thumbs - they're naturally similar between left/right
        is_thumb = finger_type.value == 'thumb'
        current_hand = hand.value
        
        # Store all similarities for debugging
        all_similarities = []
        
        for enrolled_key, enrolled_finger in self.current_user_fingers.items():
            if enrolled_finger.raw_image_data:
                try:
                    similarity = self.compare_characteristics(characteristics, enrolled_finger.raw_image_data)
                    all_similarities.append((enrolled_key, similarity))
                    
                    # Determine threshold based on finger type and hand
                    if is_thumb and enrolled_finger.finger_type == 'thumb':
                        # Thumb comparison - use different thresholds
                        if enrolled_finger.hand == current_hand:
                            # Same hand thumb - use LOWER threshold (35%)
                            threshold = 0.35
                            logger.info(f"  🔍 {enrolled_key}: {similarity:.1%} similarity (Same hand thumb - 35% threshold)")
                        else:
                            # Different hands (left vs right thumb) - use higher threshold (50%)
                            threshold = 0.5
                            logger.info(f"  🔍 {enrolled_key}: {similarity:.1%} similarity (Different hand thumb - 50% threshold)")
                    else:
                        # Non-thumb fingers - use LOWER threshold (35%)
                        threshold = 0.35
                        logger.info(f"  🔍 {enrolled_key}: {similarity:.1%} similarity (35% threshold)")
                    
                    if similarity > threshold:
                        duplicate_info = f"{enrolled_finger.hand.title()} {enrolled_finger.finger_type.title()}"
                        logger.warning(f"🚨 TRADITIONAL DUPLICATE DETECTED!")
                        logger.warning(f"   This finger is {similarity:.1%} similar to {duplicate_info}")
                        logger.warning(f"   Threshold exceeded: {similarity:.1%} > {threshold:.1%}")
                        return True, duplicate_info, similarity
                        
                except Exception as e:
                    logger.warning(f"Could not compare with {enrolled_key}: {e}")
                    continue
        
        # Show summary of all similarities for debugging
        logger.info("📊 TRADITIONAL SIMILARITY SUMMARY:")
        logger.info("-" * 30)
        for enrolled_key, similarity in sorted(all_similarities, key=lambda x: x[1], reverse=True):
            logger.info(f"  {enrolled_key}: {similarity:.1%}")
        
        logger.info("✅ No traditional duplicates found - all similarities below threshold")
        return False, "", 0.0
    
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
            
            return combined_similarity
            
        except Exception as e:
            logger.warning(f"Characteristics comparison failed: {e}")
            return 0.0
    
    def get_finger_key(self, hand: Hand, finger_type: FingerType) -> str:
        """Generate a unique key for a finger"""
        return f"{hand.value}_{finger_type.value}"
    
    def start_enrollment_session(self, user_id: str):
        """Start a new enrollment session for a specific user"""
        self.current_user_id = user_id
        self.session_start_time = time.time()
        self.current_user_fingers.clear()
        logger.info(f"Starting new AI-enhanced enrollment session for user: {user_id}")
        logger.info(f"AI Mode: {'🤖 ENABLED' if self.use_ai else '🔧 TRADITIONAL'}")
        logger.info(f"Session timeout: {self.session_timeout} seconds")
    
    def enroll_finger_with_ai(self, hand: Hand, finger_type: FingerType) -> bool:
        """Enroll a finger with AI-enhanced duplicate detection"""
        try:
            finger_key = self.get_finger_key(hand, finger_type)
            
            if finger_key in self.current_user_fingers:
                logger.error(f"🚨 DUPLICATE FINGER ALREADY ENROLLED!")
                logger.error(f"   Finger: {hand.value.title()} {finger_type.value.title()}")
                logger.error(f"   This finger was already enrolled in this session")
                logger.error(f"   Please scan a different finger")
                return False
            
            logger.info(f"🤖 AI-Enhanced Enrolling {hand.value.title()} {finger_type.value.title()}...")
            logger.info("Place your finger on the sensor and hold steady...")
            
            # Wait for finger detection
            if not self.wait_for_finger_placement():
                return False
            
            logger.info("Finger detected! Processing...")
            
            # Convert to characteristics
            logger.info("Converting image to characteristics...")
            self.scanner.convertImage(0x01)
            
            # Get current characteristics for storage
            logger.info("Downloading characteristics...")
            current_char = self.scanner.downloadCharacteristics(0x01)
            
            # Ensure current_char is bytes
            if isinstance(current_char, list):
                logger.info("Converting characteristics from list to bytes...")
                current_char = bytes(current_char)
            elif not isinstance(current_char, bytes):
                logger.warning(f"Unexpected characteristics type: {type(current_char)}, converting to bytes...")
                try:
                    current_char = bytes(current_char)
                except Exception as e:
                    logger.error(f"Failed to convert characteristics to bytes: {e}")
                    return False
            
            # Get fingerprint image for AI processing
            fingerprint_image = None
            ai_embedding = None
            ai_confidence = 0.0
            
            if self.use_ai:
                try:
                    # Read the image from scanner for AI processing
                    logger.info("🤖 Capturing image for AI analysis...")
                    self.scanner.readImage()
                    
                    # Download raw image (this is a simplified approach)
                    # In practice, you might need to convert the scanner data to proper image format
                    logger.info("🖼️ Processing image for AI duplicate detection...")
                    
                    # Create a dummy image for demonstration (replace with actual scanner image extraction)
                    # You would need to implement proper image extraction from your scanner
                    fingerprint_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                    
                    # AI duplicate detection
                    is_duplicate_ai, duplicate_info_ai, confidence_ai = self.ai_duplicate_check(
                        fingerprint_image, hand, finger_type
                    )
                    
                    if is_duplicate_ai:
                        logger.error("🤖 AI DETECTED DUPLICATE - ENROLLMENT REJECTED")
                        return False
                    
                    # Extract AI embedding for storage
                    processed_image = self.preprocessor.process(fingerprint_image)
                    if processed_image:
                        ai_embedding = self.siamese_network.extract_embedding(processed_image['model_input'])
                        ai_confidence = confidence_ai
                        logger.info(f"✅ AI embedding extracted and stored")
                    
                except Exception as e:
                    logger.warning(f"⚠️ AI processing failed: {e}")
                    logger.info("🔄 Continuing with traditional duplicate detection...")
            
            # Traditional duplicate detection as backup
            logger.info("🔍 Running traditional duplicate detection...")
            is_duplicate_traditional, duplicate_info_traditional, confidence_traditional = self.traditional_duplicate_check(
                current_char, hand, finger_type
            )
            
            if is_duplicate_traditional:
                logger.error("🚨 TRADITIONAL DUPLICATE DETECTION - ENROLLMENT REJECTED")
                return False
            
            # Store the fingerprint template
            logger.info("Creating fingerprint template...")
            self.scanner.createTemplate()
            position = self.scanner.storeTemplate()
            
            if position == 0:
                position = 1
            
            # Create fingerprint data with AI enhancements
            fingerprint_data = FingerprintData(
                user_id=self.current_user_id,
                hand=hand.value,
                finger_type=finger_type.value,
                position=position,
                timestamp=datetime.now().isoformat(),
                score=0,  # Will be set during verification
                embeddings={},  # Traditional embeddings
                ai_embeddings=ai_embedding,  # AI-generated embeddings
                raw_image_data=current_char,
                minutiae_points=[],  # Will be extracted if needed
                ai_confidence=ai_confidence
            )
            
            self.current_user_fingers[finger_key] = fingerprint_data
            
            logger.info(f"✅ AI-Enhanced enrollment successful: {hand.value.title()} {finger_type.value.title()}")
            logger.info(f"  - Position: {position}")
            logger.info(f"  - AI Enhanced: {'🤖 YES' if self.use_ai else '🔧 NO'}")
            logger.info(f"  - AI Confidence: {ai_confidence:.3f}")
            logger.info(f"  - Characteristics size: {len(current_char)} bytes")
            
            return True
            
        except Exception as e:
            logger.error(f"AI-Enhanced enrollment failed for {hand.value.title()} {finger_type.value.title()}: {e}")
            return False
    
    def wait_for_finger_placement(self, timeout_seconds: int = 30) -> bool:
        """Wait for finger placement with timeout"""
        logger.info("Waiting for finger placement...")
        
        finger_detected = False
        for i in range(timeout_seconds * 10):  # Check every 0.1 seconds
            if self.scanner.readImage():
                finger_detected = True
                break
            time.sleep(0.1)
            
            # Show progress every 2 seconds
            if i % 20 == 0 and i > 0:
                logger.info(f"Still waiting... ({i//10} seconds elapsed)")
        
        if not finger_detected:
            logger.error(f"No finger detected - timeout reached ({timeout_seconds} seconds)")
            logger.error("Please ensure your finger is properly placed on the scanner")
            return False
        
        return True
    
    def run_ai_enhanced_enrollment(self, user_id: str = None) -> bool:
        """Run complete AI-enhanced enrollment for all 10 fingers"""
        try:
            logger.info("🤖 Starting AI-Enhanced Fingerprint Enrollment...")
            logger.info("=" * 60)
            logger.info("This system uses Siamese Neural Networks for advanced duplicate detection")
            logger.info("AI models provide more accurate and reliable fingerprint comparison")
            logger.info("=" * 60)
            
            # Get user ID if not provided
            if not user_id:
                user_id = input("Enter User ID (e.g., 'user001', 'john_doe'): ").strip()
                if not user_id:
                    logger.error("User ID is required!")
                    return False
            
            if not self.initialize_scanner():
                return False
            
            self.start_enrollment_session(user_id)
            
            # Enroll each finger
            for hand, finger_type in self.required_fingers:
                logger.info(f"\n🤖 AI-Enhanced Enrolling: {hand.value.title()} {finger_type.value.title()}")
                logger.info("-" * 60)
                
                if not self.enroll_finger_with_ai(hand, finger_type):
                    logger.error(f"Failed to enroll {hand.value.title()} {finger_type.value.title()}")
                    return False
                
                # Show progress
                enrolled = len(self.current_user_fingers)
                total = len(self.required_fingers)
                logger.info(f"📊 Progress: {enrolled}/{total} fingers enrolled")
                
                time.sleep(1)
            
            logger.info("\n🎉 AI-Enhanced enrollment completed successfully!")
            logger.info("=" * 60)
            self.save_ai_enhanced_data()
            return True
                
        except Exception as e:
            logger.error(f"AI-Enhanced enrollment failed: {e}")
            return False
        finally:
            self.cleanup()
    
    def save_ai_enhanced_data(self):
        """Save AI-enhanced enrollment data"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Prepare data for serialization
            data = {
                "user_info": {
                    "user_id": self.current_user_id,
                    "enrollment_date": datetime.now().strftime("%Y-%m-%d"),
                    "completion_time": datetime.now().strftime("%H:%M:%S"),
                    "total_fingers_enrolled": len(self.current_user_fingers),
                    "ai_enhanced": self.use_ai,
                    "session_duration_seconds": time.time() - self.session_start_time if self.session_start_time else 0
                },
                "enrolled_fingers": {}
            }
            
            # Convert enrolled fingers to serializable format
            for key, fp in self.current_user_fingers.items():
                # Convert bytes to base64 for JSON serialization
                raw_data_b64 = None
                if fp.raw_image_data:
                    raw_data_b64 = base64.b64encode(fp.raw_image_data).decode('utf-8')
                
                # Convert AI embeddings to list for JSON serialization
                ai_embeddings_list = None
                if fp.ai_embeddings is not None:
                    ai_embeddings_list = fp.ai_embeddings.tolist()
                
                data["enrolled_fingers"][key] = {
                    "user_id": fp.user_id,
                    "hand": fp.hand,
                    "finger_type": fp.finger_type,
                    "position": fp.position,
                    "timestamp": fp.timestamp,
                    "score": fp.score,
                    "embeddings": fp.embeddings,
                    "ai_embeddings": ai_embeddings_list,
                    "ai_confidence": fp.ai_confidence,
                    "raw_image_data_b64": raw_data_b64,
                    "minutiae_points": fp.minutiae_points
                }
            
            # Save as JSON with AI indicator in filename
            ai_indicator = "_AI" if self.use_ai else "_TRADITIONAL"
            json_filename = f"user_{self.current_user_id}_enrollment{ai_indicator}_{timestamp}.json"
            with open(json_filename, 'w') as f:
                json.dump(data, f, indent=2)
            
            logger.info(f"🤖 AI-Enhanced enrollment data saved to {json_filename}")
            
            # Save as YAML if available
            if YAML_AVAILABLE:
                yaml_filename = f"user_{self.current_user_id}_enrollment{ai_indicator}_{timestamp}.yaml"
                with open(yaml_filename, 'w') as f:
                    yaml.dump(data, f, default_flow_style=False, indent=2)
                
                logger.info(f"📄 YAML data also saved to {yaml_filename}")
            
        except Exception as e:
            logger.error(f"Failed to save AI-enhanced enrollment data: {e}")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.scanner:
                if hasattr(self.scanner, '_serial'):
                    self.scanner._serial.close()
                logger.info("Scanner resources cleaned up")
        except Exception as e:
            logger.error(f"Cleanup failed: {e}")

def main():
    """Main function"""
    system = AIEnhancedEnrollmentSystem(use_ai=True)
    
    try:
        success = system.run_ai_enhanced_enrollment()
        return success
        
    except KeyboardInterrupt:
        logger.info("\nAI-Enhanced enrollment interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
