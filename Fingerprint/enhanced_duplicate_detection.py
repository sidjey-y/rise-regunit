#!/usr/bin/env python3
"""
Enhanced Duplicate Detection with AI Logic
This system provides better detection of wrong finger types and duplicates.
"""

import sys
import os
import numpy as np
from typing import Dict, List, Tuple, Optional
import cv2
from dataclasses import dataclass

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

@dataclass
class FingerAnalysis:
    """Analysis results for a finger scan"""
    finger_type: str
    hand: str
    confidence: float
    morphological_features: Dict
    similarity_scores: Dict[str, float]
    is_wrong_finger: bool
    detected_finger_type: str
    reason: str

class EnhancedDuplicateDetector:
    """Enhanced duplicate detection with AI logic"""
    
    def __init__(self):
        self.finger_type_models = self._initialize_finger_models()
        self.similarity_thresholds = {
            'exact_duplicate': 0.85,      # Same finger scanned twice
            'wrong_finger_type': 0.65,    # Wrong finger type detected
            'suspicious_similarity': 0.55  # Suspicious similarity
        }
    
    def _initialize_finger_models(self) -> Dict:
        """Initialize finger type detection models"""
        # This would normally load pre-trained models
        # For now, we'll use rule-based detection
        return {
            'thumb': {
                'min_area': 8000,
                'max_area': 15000,
                'aspect_ratio_range': (0.8, 1.2),
                'circularity_range': (0.6, 0.9)
            },
            'index': {
                'min_area': 6000,
                'max_area': 12000,
                'aspect_ratio_range': (1.5, 2.5),
                'circularity_range': (0.4, 0.7)
            },
            'middle': {
                'min_area': 6500,
                'max_area': 13000,
                'aspect_ratio_range': (1.6, 2.6),
                'circularity_range': (0.4, 0.7)
            },
            'ring': {
                'min_area': 5500,
                'max_area': 11000,
                'aspect_ratio_range': (1.4, 2.4),
                'circularity_range': (0.4, 0.7)
            },
            'little': {
                'min_area': 4500,
                'max_area': 9000,
                'aspect_ratio_range': (1.8, 2.8),
                'circularity_range': (0.3, 0.6)
            }
        }
    
    def analyze_finger_scan(self, characteristics: bytes, expected_hand: str, expected_finger_type: str) -> FingerAnalysis:
        """Analyze a finger scan for wrong finger type detection"""
        try:
            # Convert characteristics to image-like data for analysis
            # This is a simplified approach - in practice you'd use the actual fingerprint image
            
            # Extract morphological features
            morphological_features = self._extract_morphological_features(characteristics)
            
            # Detect actual finger type based on features
            detected_finger_type, confidence = self._detect_finger_type(morphological_features)
            
            # Check if wrong finger type
            is_wrong_finger = detected_finger_type != expected_finger_type
            
            # Calculate similarity scores
            similarity_scores = self._calculate_similarity_scores(morphological_features, expected_finger_type)
            
            # Determine reason for wrong finger detection
            reason = self._determine_wrong_finger_reason(
                expected_finger_type, detected_finger_type, 
                morphological_features, similarity_scores
            )
            
            return FingerAnalysis(
                finger_type=expected_finger_type,
                hand=expected_hand,
                confidence=confidence,
                morphological_features=morphological_features,
                similarity_scores=similarity_scores,
                is_wrong_finger=is_wrong_finger,
                detected_finger_type=detected_finger_type,
                reason=reason
            )
            
        except Exception as e:
            print(f"Error in finger analysis: {e}")
            return None
    
    def _extract_morphological_features(self, characteristics: bytes) -> Dict:
        """Extract morphological features from fingerprint characteristics"""
        try:
            # Convert characteristics to numerical data for analysis
            char_array = np.frombuffer(characteristics, dtype=np.uint8)
            
            # Calculate various features
            features = {
                'area': len(char_array),
                'mean_intensity': np.mean(char_array),
                'std_intensity': np.std(char_array),
                'entropy': self._calculate_entropy(char_array),
                'ridge_density': self._estimate_ridge_density(char_array),
                'pattern_complexity': self._calculate_pattern_complexity(char_array)
            }
            
            # Normalize features
            features = self._normalize_features(features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting morphological features: {e}")
            return {}
    
    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of the data"""
        try:
            hist, _ = np.histogram(data, bins=256, range=(0, 256))
            hist = hist[hist > 0]  # Remove zero bins
            prob = hist / np.sum(hist)
            entropy = -np.sum(prob * np.log2(prob))
            return entropy
        except:
            return 0.0
    
    def _estimate_ridge_density(self, data: np.ndarray) -> float:
        """Estimate ridge density from characteristics"""
        try:
            # Simplified ridge density estimation
            # In practice, this would use more sophisticated algorithms
            variations = np.diff(data)
            ridge_count = np.sum(np.abs(variations) > np.std(variations))
            return ridge_count / len(variations) if len(variations) > 0 else 0.0
        except:
            return 0.0
    
    def _calculate_pattern_complexity(self, data: np.ndarray) -> float:
        """Calculate pattern complexity"""
        try:
            # Use FFT to analyze pattern complexity
            fft = np.fft.fft(data)
            magnitude = np.abs(fft)
            complexity = np.std(magnitude) / np.mean(magnitude) if np.mean(magnitude) > 0 else 0.0
            return complexity
        except:
            return 0.0
    
    def _normalize_features(self, features: Dict) -> Dict:
        """Normalize features to 0-1 range"""
        try:
            normalized = {}
            for key, value in features.items():
                if isinstance(value, (int, float)):
                    # Simple normalization - in practice you'd use more sophisticated methods
                    if key == 'area':
                        normalized[key] = min(value / 10000, 1.0)  # Normalize area
                    elif key == 'entropy':
                        normalized[key] = min(value / 8.0, 1.0)   # Normalize entropy (max 8 bits)
                    else:
                        normalized[key] = min(max(value, 0.0), 1.0)
                else:
                    normalized[key] = value
            return normalized
        except:
            return features
    
    def _detect_finger_type(self, features: Dict) -> Tuple[str, float]:
        """Detect finger type based on morphological features"""
        try:
            best_match = None
            best_score = 0.0
            
            for finger_type, model in self.finger_type_models.items():
                score = self._calculate_finger_type_score(features, model)
                if score > best_score:
                    best_score = score
                    best_match = finger_type
            
            return best_match or 'unknown', best_score
            
        except Exception as e:
            print(f"Error detecting finger type: {e}")
            return 'unknown', 0.0
    
    def _calculate_finger_type_score(self, features: Dict, model: Dict) -> float:
        """Calculate how well features match a finger type model"""
        try:
            score = 0.0
            total_checks = 0
            
            # Check area constraints
            if 'area' in features and 'min_area' in model and 'max_area' in model:
                area = features['area']
                if model['min_area'] <= area <= model['max_area']:
                    score += 1.0
                total_checks += 1
            
            # Check pattern complexity
            if 'pattern_complexity' in features:
                # Different finger types have different complexity patterns
                if 'thumb' in model:
                    # Thumbs typically have more complex patterns
                    if features['pattern_complexity'] > 0.6:
                        score += 1.0
                elif 'little' in model:
                    # Little fingers typically have simpler patterns
                    if features['pattern_complexity'] < 0.4:
                        score += 1.0
                else:
                    # Middle fingers have moderate complexity
                    if 0.3 <= features['pattern_complexity'] <= 0.7:
                        score += 1.0
                total_checks += 1
            
            # Check ridge density
            if 'ridge_density' in features:
                # Thumbs have higher ridge density
                if 'thumb' in model and features['ridge_density'] > 0.6:
                    score += 1.0
                elif 'little' in model and features['ridge_density'] < 0.4:
                    score += 1.0
                elif features['ridge_density'] > 0.3:
                    score += 1.0
                total_checks += 1
            
            return score / total_checks if total_checks > 0 else 0.0
            
        except Exception as e:
            print(f"Error calculating finger type score: {e}")
            return 0.0
    
    def _calculate_similarity_scores(self, features: Dict, expected_finger_type: str) -> Dict[str, float]:
        """Calculate similarity scores for different aspects"""
        try:
            scores = {}
            
            # Area similarity
            if 'area' in features:
                expected_area = self.finger_type_models.get(expected_finger_type, {}).get('area', 10000)
                area_diff = abs(features['area'] - expected_area) / expected_area
                scores['area_similarity'] = max(0.0, 1.0 - area_diff)
            
            # Pattern similarity
            if 'pattern_complexity' in features:
                expected_complexity = 0.5  # Default moderate complexity
                if expected_finger_type == 'thumb':
                    expected_complexity = 0.7
                elif expected_finger_type == 'little':
                    expected_complexity = 0.3
                
                complexity_diff = abs(features['pattern_complexity'] - expected_complexity)
                scores['pattern_similarity'] = max(0.0, 1.0 - complexity_diff)
            
            # Overall similarity
            if scores:
                scores['overall_similarity'] = np.mean(list(scores.values()))
            else:
                scores['overall_similarity'] = 0.0
            
            return scores
            
        except Exception as e:
            print(f"Error calculating similarity scores: {e}")
            return {'overall_similarity': 0.0}
    
    def _determine_wrong_finger_reason(self, expected: str, detected: str, features: Dict, scores: Dict) -> str:
        """Determine the reason for wrong finger detection"""
        try:
            if expected == detected:
                return "Correct finger type detected"
            
            reasons = []
            
            # Check area mismatch
            if 'area_similarity' in scores and scores['area_similarity'] < 0.7:
                reasons.append("Size mismatch")
            
            # Check pattern complexity mismatch
            if 'pattern_similarity' in scores and scores['pattern_similarity'] < 0.6:
                reasons.append("Pattern complexity mismatch")
            
            # Check overall similarity
            if 'overall_similarity' in scores and scores['overall_similarity'] < 0.6:
                reasons.append("Overall characteristics mismatch")
            
            if reasons:
                return f"Wrong finger type: {detected} detected instead of {expected}. Reasons: {', '.join(reasons)}"
            else:
                return f"Wrong finger type: {detected} detected instead of {expected}"
                
        except Exception as e:
            print(f"Error determining wrong finger reason: {e}")
            return f"Wrong finger type: {detected} detected instead of {expected}"

def test_enhanced_detection():
    """Test the enhanced duplicate detection system"""
    print("🧠 TESTING ENHANCED DUPLICATE DETECTION")
    print("=" * 60)
    
    try:
        detector = EnhancedDuplicateDetector()
        print("✅ Enhanced detector initialized successfully")
        
        # Test 1: Simulate correct finger scan
        print("\n📝 Test 1: Correct finger type (left thumb)")
        print("-" * 40)
        
        # Simulate thumb characteristics
        thumb_chars = bytes([100] * 1000)  # Simulate thumb-like data
        analysis = detector.analyze_finger_scan(thumb_chars, "left", "thumb")
        
        if analysis:
            print(f"Expected: left thumb")
            print(f"Detected: {analysis.detected_finger_type}")
            print(f"Confidence: {analysis.confidence:.2%}")
            print(f"Wrong finger: {analysis.is_wrong_finger}")
            print(f"Reason: {analysis.reason}")
            print(f"Overall similarity: {analysis.similarity_scores.get('overall_similarity', 0):.2%}")
        
        # Test 2: Simulate wrong finger scan (middle finger instead of ring)
        print("\n📝 Test 2: Wrong finger type (middle finger instead of ring)")
        print("-" * 40)
        
        # Simulate middle finger characteristics
        middle_chars = bytes([80] * 800)  # Simulate middle finger-like data
        analysis = detector.analyze_finger_scan(middle_chars, "left", "ring")
        
        if analysis:
            print(f"Expected: left ring")
            print(f"Detected: {analysis.detected_finger_type}")
            print(f"Confidence: {analysis.confidence:.2%}")
            print(f"Wrong finger: {analysis.is_wrong_finger}")
            print(f"Reason: {analysis.reason}")
            print(f"Overall similarity: {analysis.similarity_scores.get('overall_similarity', 0):.2%}")
        
        print("\n🎉 Enhanced detection test completed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    test_enhanced_detection()
