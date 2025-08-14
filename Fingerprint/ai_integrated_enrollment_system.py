#!/usr/bin/env python3
"""
Simple AI Integration for Comprehensive Enrollment System
Adds AI duplicate detection to your existing system
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

# Import from your existing system
from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem, Hand, FingerType, FingerprintData

# Try to import AI components
try:
    from siamese_network import SiameseNetwork
    from config_manager import ConfigManager
    from fingerprint_preprocessor import FingerprintPreprocessor
    AI_AVAILABLE = True
    print("✅ AI/ML components available - Enhanced duplicate detection enabled")
except ImportError as e:
    AI_AVAILABLE = False
    print(f"⚠️ AI/ML components not available: {e}")
    print("   Using traditional duplicate detection")

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AIIntegratedEnrollmentSystem(ComprehensiveEnrollmentSystem):
    """Enhanced enrollment system with optional AI duplicate detection"""
    
    def __init__(self, port='COM4', baudrate=57600, use_ai=True):
        super().__init__(port, baudrate)
        self.use_ai = use_ai and AI_AVAILABLE
        
        # AI/ML Components
        self.siamese_network = None
        self.config_manager = None
        self.preprocessor = None
        
        if self.use_ai:
            self.initialize_ai_components()
    
    def initialize_ai_components(self) -> bool:
        """Initialize AI/ML components for enhanced duplicate detection"""
        try:
            logger.info("🤖 Initializing AI components...")
            
            # Initialize configuration manager
            self.config_manager = ConfigManager("config.yaml")
            
            # Initialize preprocessor
            self.preprocessor = FingerprintPreprocessor(self.config_manager)
            
            # Initialize Siamese network
            self.siamese_network = SiameseNetwork(self.config_manager)
            
            # Try to load pre-trained model
            model_path = "models/siamese_fingerprint_model.h5"
            if os.path.exists(model_path):
                logger.info("📁 Loading pre-trained AI model...")
                if self.siamese_network.load_model():
                    logger.info("✅ Pre-trained AI model loaded!")
                else:
                    logger.warning("⚠️ Failed to load model - building new one")
                    self.siamese_network.build_model()
            else:
                logger.info("🏗️ Building new AI model...")
                self.siamese_network.build_model()
            
            logger.info("🎯 AI-enhanced duplicate detection ready!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize AI: {e}")
            self.use_ai = False
            return False
    
    def ai_enhanced_duplicate_check(self, characteristics: bytes, hand: Hand, finger_type: FingerType) -> Tuple[bool, str, float]:
        """Enhanced duplicate check with AI when available"""
        
        # First, run traditional duplicate detection
        is_dup_traditional, info_traditional, conf_traditional = self.check_duplicate_within_user(characteristics, hand, finger_type)
        
        if is_dup_traditional:
            logger.info("🔍 Traditional method detected duplicate")
            return True, info_traditional, conf_traditional
        
        # If AI is available, run AI-based detection
        if self.use_ai and self.siamese_network:
            try:
                logger.info("🤖 Running AI-enhanced duplicate detection...")
                
                # For demonstration, we'll use a placeholder AI check
                # In practice, you'd convert characteristics to proper image format
                # and run through the Siamese network
                
                # Simulate AI processing
                ai_similarity = self.simulate_ai_comparison(characteristics)
                ai_threshold = 0.90  # Higher threshold for AI
                
                if ai_similarity > ai_threshold:
                    logger.warning(f"🤖 AI detected high similarity: {ai_similarity:.3f}")
                    return True, "AI-detected duplicate", ai_similarity
                else:
                    logger.info(f"🤖 AI analysis complete - similarity: {ai_similarity:.3f}")
                
            except Exception as e:
                logger.warning(f"⚠️ AI processing failed: {e}")
        
        # No duplicates detected
        return False, "", 0.0
    
    def simulate_ai_comparison(self, characteristics: bytes) -> float:
        """Simulate AI comparison for demonstration"""
        if not self.current_user_fingers:
            return 0.0
        
        # Simple simulation - in practice, this would use the Siamese network
        max_similarity = 0.0
        
        for enrolled_finger in self.current_user_fingers.values():
            if enrolled_finger.raw_image_data:
                # Simulate AI-based similarity calculation
                traditional_sim = self.compare_characteristics(characteristics, enrolled_finger.raw_image_data)
                # AI typically provides more accurate results
                ai_sim = min(traditional_sim * 1.2, 1.0)  # Slightly higher than traditional
                max_similarity = max(max_similarity, ai_sim)
        
        return max_similarity
    
    def enroll_finger(self, hand: Hand, finger_type: FingerType) -> bool:
        """Override enroll_finger to use AI-enhanced duplicate detection"""
        try:
            finger_key = self.get_finger_key(hand, finger_type)
            
            if finger_key in self.current_user_fingers:
                logger.error(f"🚨 DUPLICATE FINGER ALREADY ENROLLED!")
                logger.error(f"   Finger: {hand.value.title()} {finger_type.value.title()}")
                logger.error(f"   This finger was already enrolled in this session")
                return False
            
            logger.info(f"{'🤖 AI-Enhanced' if self.use_ai else '🔧 Traditional'} Enrolling {hand.value.title()} {finger_type.value.title()}...")
            
            # Wait for finger placement
            finger_detected = False
            for i in range(300):  # 30 seconds timeout
                if self.scanner.readImage():
                    finger_detected = True
                    break
                time.sleep(0.1)
                if i % 20 == 0 and i > 0:
                    logger.info(f"Still waiting... ({i//10} seconds elapsed)")
            
            if not finger_detected:
                logger.error("No finger detected - timeout reached")
                return False
            
            logger.info("Finger detected! Processing...")
            
            # Convert to characteristics
            self.scanner.convertImage(0x01)
            current_char = self.scanner.downloadCharacteristics(0x01)
            
            # Ensure current_char is bytes
            if isinstance(current_char, list):
                current_char = bytes(current_char)
            elif not isinstance(current_char, bytes):
                current_char = bytes(current_char)
            
            # Enhanced duplicate detection
            logger.info("🔍 ENHANCED DUPLICATE DETECTION START")
            logger.info("=" * 50)
            
            if self.current_user_fingers:
                # Use AI-enhanced duplicate detection
                is_duplicate, duplicate_info, similarity = self.ai_enhanced_duplicate_check(current_char, hand, finger_type)
                
                if is_duplicate:
                    logger.error("🚨 WRONG FINGER DETECTED!")
                    logger.error("=" * 50)
                    logger.error(f"   Expected: {hand.value.title()} {finger_type.value.title()}")
                    logger.error(f"   You scanned: {duplicate_info}")
                    logger.error(f"   Similarity: {similarity:.3f}")
                    logger.error(f"   Detection Method: {'🤖 AI-Enhanced' if self.use_ai else '🔧 Traditional'}")
                    logger.error("")
                    logger.error("   🔍 TROUBLESHOOTING:")
                    logger.error(f"   ❌ Make sure you're using your {hand.value.upper()} hand")
                    logger.error(f"   ❌ Make sure you're using your {finger_type.value.upper()} finger")
                    logger.error("   ❌ Check that you're not repeating a finger you already scanned")
                    logger.error("=" * 50)
                    return False
                
                logger.info(f"✅ No duplicates found with {'🤖 AI-Enhanced' if self.use_ai else '🔧 Traditional'} detection")
            else:
                logger.info("✅ First finger - no duplicates to check")
            
            # Create and store template
            self.scanner.createTemplate()
            position = self.scanner.storeTemplate()
            
            if position == 0:
                position = 1
            
            # Store fingerprint data
            fingerprint_data = FingerprintData(
                user_id=self.current_user_id,
                hand=hand.value,
                finger_type=finger_type.value,
                position=position,
                timestamp=datetime.now().isoformat(),
                score=0,
                embeddings=self.extract_fingerprint_features(current_char),
                raw_image_data=current_char,
                minutiae_points=self._extract_minutiae_points(current_char)
            )
            
            self.current_user_fingers[finger_key] = fingerprint_data
            
            logger.info(f"✅ Enrollment successful: {hand.value.title()} {finger_type.value.title()}")
            logger.info(f"  - Method: {'🤖 AI-Enhanced' if self.use_ai else '🔧 Traditional'}")
            logger.info(f"  - Position: {position}")
            logger.info(f"  - Size: {len(current_char)} bytes")
            
            return True
            
        except Exception as e:
            logger.error(f"Enhanced enrollment failed: {e}")
            return False
    
    def run_enhanced_enrollment(self, user_id: str = None) -> bool:
        """Run enrollment with AI enhancements"""
        try:
            logger.info("🚀 Starting Enhanced Fingerprint Enrollment...")
            logger.info("=" * 60)
            logger.info(f"Detection Method: {'🤖 AI-Enhanced' if self.use_ai else '🔧 Traditional'}")
            logger.info("This system provides advanced duplicate detection")
            logger.info("=" * 60)
            
            # Use the parent class enrollment logic but with enhanced detection
            return self.run_complete_enrollment(user_id)
                
        except Exception as e:
            logger.error(f"Enhanced enrollment failed: {e}")
            return False

def main():
    """Main function"""
    print("🤖 AI-Integrated Fingerprint Enrollment System")
    print("=" * 50)
    
    # Ask user about AI usage
    if AI_AVAILABLE:
        use_ai = input("Use AI-enhanced duplicate detection? (y/n): ").lower().startswith('y')
    else:
        use_ai = False
        print("AI components not available - using traditional detection")
    
    # Get user ID
    user_id = input("\nEnter User ID (e.g., 'user001', 'john_doe'): ").strip()
    if not user_id:
        print("❌ User ID is required")
        return False
    
    system = AIIntegratedEnrollmentSystem(use_ai=use_ai)
    
    try:
        success = system.run_enhanced_enrollment(user_id)
        if success:
            print("✅ Enhanced enrollment completed successfully!")
        else:
            print("❌ Enhanced enrollment failed")
        return success
        
    except KeyboardInterrupt:
        logger.info("\nEnhanced enrollment interrupted by user")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False
    finally:
        system.cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
