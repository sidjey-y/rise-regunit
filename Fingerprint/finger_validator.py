#!/usr/bin/env python3
"""
Finger Validation System
Validates that the scanned finger matches the expected finger and provides user guidance.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from finger_classifier import ComputerVisionFingerClassifier

class FingerValidator:
    """
    Validates scanned fingerprints against expected finger types and provides user guidance
    """
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.classifier = ComputerVisionFingerClassifier()
        self.confidence_threshold = confidence_threshold
        
        # Finger validation rules
        self.finger_characteristics = {
            'thumb': {
                'size': 'large',
                'shape': 'round',
                'typical_area': (40000, 80000),
                'description': 'Thumb is typically the largest and roundest finger'
            },
            'index': {
                'size': 'medium',
                'shape': 'elongated',
                'typical_area': (25000, 45000),
                'description': 'Index finger is long and often used for pointing'
            },
            'middle': {
                'size': 'medium',
                'shape': 'elongated',
                'typical_area': (25000, 45000),
                'description': 'Middle finger is similar to index but slightly longer'
            },
            'ring': {
                'size': 'medium',
                'shape': 'elongated',
                'description': 'Ring finger is similar to middle finger'
            },
            'little': {
                'size': 'small',
                'shape': 'elongated',
                'description': 'Little finger is the smallest and thinnest'
            }
        }
    
    def validate_finger_scan(self, scanned_image: np.ndarray, 
                           expected_finger_type: str, 
                           expected_hand_side: str) -> Dict[str, Any]:
        """
        Validate that the scanned finger matches the expected finger
        
        Args:
            scanned_image: The captured fingerprint image
            expected_finger_type: Expected finger type (thumb, index, middle, ring, little)
            expected_hand_side: Expected hand side (left, right)
            
        Returns:
            Dictionary with validation results and guidance
        """
        try:
            # Classify the scanned finger
            classification = self.classifier.classify_finger(scanned_image)
            
            if classification.get('error'):
                return {
                    'is_valid': False,
                    'error': classification['error'],
                    'guidance': 'Unable to classify finger. Please try scanning again.',
                    'classification': classification
                }
            
            detected_finger_type = classification['finger_type']
            detected_hand_side = classification['hand_side']
            confidence = classification['confidence']
            
            # Check if classification confidence is sufficient
            if confidence < self.confidence_threshold:
                return {
                    'is_valid': False,
                    'confidence_too_low': True,
                    'confidence': confidence,
                    'guidance': f'Finger classification confidence is low ({confidence:.2f}). Please ensure finger is properly placed on scanner.',
                    'classification': classification
                }
            
            # Check if finger type matches
            finger_type_match = detected_finger_type.lower() == expected_finger_type.lower()
            
            # Check if hand side matches (if we can determine it)
            hand_side_match = True  # Default to True if we can't determine
            if detected_hand_side != 'unknown':
                hand_side_match = detected_hand_side.lower() == expected_hand_side.lower()
            
            # Determine overall validity
            is_valid = finger_type_match and hand_side_match
            
            # Generate guidance messages
            guidance = self._generate_guidance(
                expected_finger_type, expected_hand_side,
                detected_finger_type, detected_hand_side,
                finger_type_match, hand_side_match,
                confidence
            )
            
            # Calculate validation score
            validation_score = self._calculate_validation_score(
                finger_type_match, hand_side_match, confidence
            )
            
            return {
                'is_valid': is_valid,
                'finger_type_match': finger_type_match,
                'hand_side_match': hand_side_match,
                'confidence': confidence,
                'validation_score': validation_score,
                'expected_finger': f"{expected_hand_side} {expected_finger_type}",
                'detected_finger': f"{detected_hand_side} {detected_finger_type}",
                'guidance': guidance,
                'classification': classification,
                'retry_needed': not is_valid
            }
            
        except Exception as e:
            self.logger.error(f"Finger validation failed: {e}")
            return {
                'is_valid': False,
                'error': str(e),
                'guidance': 'Validation failed. Please try scanning again.',
                'retry_needed': True
            }
    
    def _generate_guidance(self, expected_finger_type: str, expected_hand_side: str,
                          detected_finger_type: str, detected_hand_side: str,
                          finger_type_match: bool, hand_side_match: bool,
                          confidence: float) -> str:
        """Generate user guidance based on validation results"""
        
        guidance_parts = []
        
        if not finger_type_match:
            expected_char = self.finger_characteristics.get(expected_finger_type.lower(), {})
            guidance_parts.append(
                f"❌ Wrong finger detected! Expected: {expected_finger_type.title()}, "
                f"Detected: {detected_finger_type.title()}. "
                f"{expected_char.get('description', 'Please scan the correct finger.')}"
            )
        
        if not hand_side_match and detected_hand_side != 'unknown':
            guidance_parts.append(
                f"❌ Wrong hand detected! Expected: {expected_hand_side.title()} hand, "
                f"Detected: {detected_hand_side.title()} hand. "
                f"Please use your {expected_hand_side} hand."
            )
        
        if confidence < self.confidence_threshold:
            guidance_parts.append(
                f"⚠️ Low confidence in finger detection ({confidence:.2f}). "
                "Please ensure your finger is properly placed on the scanner."
            )
        
        if finger_type_match and hand_side_match:
            guidance_parts.append(
                f"✅ Correct finger detected! {expected_hand_side.title()} {expected_finger_type.title()} "
                f"with confidence {confidence:.2f}"
            )
        
        # Add specific guidance for the expected finger
        if expected_finger_type.lower() in self.finger_characteristics:
            char = self.finger_characteristics[expected_finger_type.lower()]
            guidance_parts.append(
                f"💡 Tip: {expected_finger_type.title()} finger is {char['size']} and {char['shape']}. "
                f"{char['description']}"
            )
        
        return " ".join(guidance_parts)
    
    def _calculate_validation_score(self, finger_type_match: bool, 
                                  hand_side_match: bool, confidence: float) -> float:
        """Calculate a validation score from 0.0 to 1.0"""
        score = 0.0
        
        # Base score from confidence
        score += confidence * 0.6
        
        # Bonus for correct finger type
        if finger_type_match:
            score += 0.3
        
        # Bonus for correct hand side
        if hand_side_match:
            score += 0.1
        
        return min(1.0, score)
    
    def get_retry_instructions(self, expected_finger_type: str, 
                              expected_hand_side: str) -> str:
        """Get specific instructions for retrying with the correct finger"""
        
        char = self.finger_characteristics.get(expected_finger_type.lower(), {})
        
        instructions = [
            f"🔄 Please retry with your {expected_hand_side} {expected_finger_type} finger:",
            f"   • Use your {expected_hand_side} hand",
            f"   • Place your {expected_finger_type} finger on the scanner",
            f"   • Ensure the finger is centered and fully visible",
            f"   • Keep your finger steady and apply gentle pressure"
        ]
        
        if char:
            instructions.append(f"   • {expected_finger_type.title()} finger: {char['description']}")
        
        instructions.extend([
            "   • Press any key when ready to capture",
            "   • Press 'q' to cancel"
        ])
        
        return "\n".join(instructions)
    
    def analyze_finger_quality(self, image: np.ndarray) -> Dict[str, Any]:
        """Analyze the quality of the scanned fingerprint"""
        try:
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Calculate various quality metrics
            quality_metrics = {}
            
            # Brightness analysis
            mean_brightness = np.mean(gray)
            brightness_std = np.std(gray)
            quality_metrics['brightness'] = {
                'mean': mean_brightness,
                'std': brightness_std,
                'score': self._score_brightness(mean_brightness, brightness_std)
            }
            
            # Contrast analysis
            contrast = np.std(gray)
            quality_metrics['contrast'] = {
                'value': contrast,
                'score': self._score_contrast(contrast)
            }
            
            # Sharpness analysis
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality_metrics['sharpness'] = {
                'value': laplacian_var,
                'score': self._score_sharpness(laplacian_var)
            }
            
            # Overall quality score
            overall_score = np.mean([
                quality_metrics['brightness']['score'],
                quality_metrics['contrast']['score'],
                quality_metrics['sharpness']['score']
            ])
            
            quality_metrics['overall_score'] = overall_score
            quality_metrics['quality_level'] = self._get_quality_level(overall_score)
            
            return quality_metrics
            
        except Exception as e:
            self.logger.error(f"Quality analysis failed: {e}")
            return {'error': str(e)}
    
    def _score_brightness(self, mean: float, std: float) -> float:
        """Score brightness quality (0.0 to 1.0)"""
        # Ideal brightness range: 80-180
        if 80 <= mean <= 180:
            brightness_score = 1.0
        elif 60 <= mean <= 200:
            brightness_score = 0.8
        elif 40 <= mean <= 220:
            brightness_score = 0.6
        else:
            brightness_score = 0.3
        
        # Penalize very low standard deviation (flat image)
        if std < 10:
            brightness_score *= 0.5
        
        return brightness_score
    
    def _score_contrast(self, contrast: float) -> float:
        """Score contrast quality (0.0 to 1.0)"""
        if contrast > 50:
            return 1.0
        elif contrast > 30:
            return 0.8
        elif contrast > 20:
            return 0.6
        elif contrast > 10:
            return 0.4
        else:
            return 0.2
    
    def _score_sharpness(self, laplacian_var: float) -> float:
        """Score sharpness quality (0.0 to 1.0)"""
        if laplacian_var > 500:
            return 1.0
        elif laplacian_var > 300:
            return 0.8
        elif laplacian_var > 100:
            return 0.6
        elif laplacian_var > 50:
            return 0.4
        else:
            return 0.2
    
    def _get_quality_level(self, score: float) -> str:
        """Get quality level description"""
        if score >= 0.8:
            return "Excellent"
        elif score >= 0.6:
            return "Good"
        elif score >= 0.4:
            return "Fair"
        elif score >= 0.2:
            return "Poor"
        else:
            return "Very Poor"

def test_finger_validator():
    """Test function for the finger validator"""
    print("Testing finger validator...")
    
    # Create validator
    validator = FingerValidator()
    
    # Create a test image
    test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    # Test validation
    result = validator.validate_finger_scan(test_image, "thumb", "right")
    print(f"Validation result: {result}")
    
    # Test quality analysis
    quality = validator.analyze_finger_quality(test_image)
    print(f"Quality analysis: {quality}")
    
    # Test retry instructions
    instructions = validator.get_retry_instructions("index", "left")
    print(f"Retry instructions:\n{instructions}")

if __name__ == "__main__":
    test_finger_validator()
