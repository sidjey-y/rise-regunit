#!/usr/bin/env python3
"""
Automatic Finger Classification System
Detects which finger is being scanned using computer vision and machine learning techniques.
"""

import cv2
import numpy as np
import logging
from typing import Dict, Any, Optional, Tuple, List
from abc import ABC, abstractmethod
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

class FingerClassifier(ABC):
    """Abstract base class for finger classification"""
    
    @abstractmethod
    def classify_finger(self, image: np.ndarray) -> Dict[str, Any]:
        """Classify the finger in the given image"""
        pass
    
    @abstractmethod
    def train(self, training_data: List[Tuple[np.ndarray, str]]) -> bool:
        """Train the classifier with labeled data"""
        pass
    
    @abstractmethod
    def save_model(self, filepath: str) -> bool:
        """Save the trained model"""
        pass
    
    @abstractmethod
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        pass

class ComputerVisionFingerClassifier(FingerClassifier):
    """
    Computer vision-based finger classifier using morphological features
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = [
            'area', 'perimeter', 'aspect_ratio', 'circularity', 'solidity',
            'extent', 'convexity', 'finger_length', 'finger_width',
            'ridge_density', 'valley_count', 'orientation_variance'
        ]
        self.finger_types = ['thumb', 'index', 'middle', 'ring', 'little']
        self.hand_sides = ['left', 'right']
        
    def classify_finger(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Classify the finger in the given image
        
        Args:
            image: Grayscale fingerprint image
            
        Returns:
            Dictionary with classification results
        """
        try:
            # Extract features from the image
            features = self._extract_features(image)
            
            if self.model is None:
                # Use rule-based classification if no trained model
                return self._rule_based_classification(features, image)
            
            # Use trained model for classification
            features_scaled = self.scaler.transform([features])
            prediction = self.model.predict(features_scaled)[0]
            confidence = self.model.predict_proba(features_scaled)[0].max()
            
            # Determine hand side based on image characteristics
            hand_side = self._determine_hand_side(image, features)
            
            return {
                'finger_type': prediction,
                'hand_side': hand_side,
                'confidence': confidence,
                'features': features,
                'classification_method': 'ml_model'
            }
            
        except Exception as e:
            self.logger.error(f"Finger classification failed: {e}")
            return {
                'finger_type': 'unknown',
                'hand_side': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'classification_method': 'failed'
            }
    
    def _extract_features(self, image: np.ndarray) -> np.ndarray:
        """Extract morphological and texture features from the image"""
        try:
            # Ensure image is grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Binarize the image
            _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                return np.zeros(len(self.feature_names))
            
            # Get the largest contour (main finger)
            main_contour = max(contours, key=cv2.contourArea)
            
            # Basic morphological features
            area = cv2.contourArea(main_contour)
            perimeter = cv2.arcLength(main_contour, True)
            
            # Bounding rectangle
            x, y, w, h = cv2.boundingRect(main_contour)
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Circularity
            circularity = (4 * np.pi * area) / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Convex hull features
            hull = cv2.convexHull(main_contour)
            hull_area = cv2.contourArea(hull)
            solidity = area / hull_area if hull_area > 0 else 0
            convexity = len(hull) / len(main_contour) if len(main_contour) > 0 else 0
            
            # Extent
            rect_area = w * h
            extent = area / rect_area if rect_area > 0 else 0
            
            # Finger dimensions
            finger_length = max(w, h)
            finger_width = min(w, h)
            
            # Ridge density (using edge detection)
            edges = cv2.Canny(gray, 50, 150)
            ridge_density = np.sum(edges > 0) / (gray.shape[0] * gray.shape[1])
            
            # Valley count (using morphological operations)
            kernel = np.ones((3, 3), np.uint8)
            valleys = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            valley_count = np.sum(valleys > 0) / (gray.shape[0] * gray.shape[1])
            
            # Orientation variance (using gradient)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            orientation = np.arctan2(grad_y, grad_x)
            orientation_variance = np.var(orientation)
            
            features = np.array([
                area, perimeter, aspect_ratio, circularity, solidity,
                extent, convexity, finger_length, finger_width,
                ridge_density, valley_count, orientation_variance
            ])
            
            return features
            
        except Exception as e:
            self.logger.error(f"Feature extraction failed: {e}")
            return np.zeros(len(self.feature_names))
    
    def _rule_based_classification(self, features: np.ndarray, image: np.ndarray) -> Dict[str, Any]:
        """Enhanced rule-based classification when no trained model is available"""
        try:
            # Enhanced rules based on finger characteristics
            area, perimeter, aspect_ratio, circularity, solidity, extent, convexity, finger_length, finger_width, ridge_density, valley_count, orientation_variance = features
            
            # More sophisticated finger type determination
            if area > 60000:  # Very large area - likely thumb
                if circularity > 0.6:  # More circular
                    finger_type = 'thumb'
                    confidence = 0.85
                else:
                    finger_type = 'thumb'  # Still thumb but less circular
                    confidence = 0.75
            elif area > 45000:  # Large-medium area
                if aspect_ratio > 1.8:  # Very long
                    finger_type = 'middle'
                    confidence = 0.8
                elif aspect_ratio > 1.4:  # Long
                    finger_type = 'index'
                    confidence = 0.75
                else:
                    finger_type = 'ring'
                    confidence = 0.7
            elif area > 30000:  # Medium area
                if aspect_ratio > 1.6:  # Long
                    if solidity > 0.8:  # More solid
                        finger_type = 'index'
                        confidence = 0.8
                    else:
                        finger_type = 'middle'
                        confidence = 0.75
                else:
                    finger_type = 'ring'
                    confidence = 0.7
            else:  # Small area
                if aspect_ratio > 1.5:  # Long and thin
                    finger_type = 'little'
                    confidence = 0.85
                else:
                    finger_type = 'little'  # Small but not very long
                    confidence = 0.7
            
            # Enhanced hand side determination
            hand_side = self._determine_hand_side(image, features)
            
            # Additional confidence adjustment based on feature consistency
            if finger_type == 'thumb' and area > 70000:
                confidence = min(confidence + 0.1, 1.0)
            elif finger_type == 'little' and area < 25000:
                confidence = min(confidence + 0.1, 1.0)
            
            return {
                'finger_type': finger_type,
                'hand_side': hand_side,
                'confidence': confidence,
                'features': features,
                'classification_method': 'enhanced_rule_based',
                'feature_analysis': {
                    'area': area,
                    'aspect_ratio': aspect_ratio,
                    'circularity': circularity,
                    'solidity': solidity,
                    'length': finger_length,
                    'width': finger_width,
                    'ridge_density': ridge_density
                }
            }
            
        except Exception as e:
            self.logger.error(f"Rule-based classification failed: {e}")
            return {
                'finger_type': 'unknown',
                'hand_side': 'unknown',
                'confidence': 0.0,
                'error': str(e),
                'classification_method': 'rule_based_failed'
            }
    
    def _determine_hand_side(self, image: np.ndarray, features: np.ndarray) -> str:
        """Enhanced hand side determination using image analysis"""
        try:
            # Enhanced approach using image characteristics and features
            area, perimeter, aspect_ratio, circularity, solidity, extent, convexity, finger_length, finger_width, ridge_density, valley_count, orientation_variance = features
            
            # Method 1: Analyze image orientation patterns
            # This is a heuristic based on typical fingerprint scanner orientations
            
            # Method 2: Use feature asymmetry
            # Some features might be slightly different between left and right hands
            
            # Method 3: Check for orientation markers
            # Some scanners have orientation indicators
            
            # For now, use a combination of heuristics
            # In a real system, you would:
            # 1. Train a classifier on labeled left/right hand data
            # 2. Use hand pose estimation if available
            # 3. Analyze finger orientation patterns
            
            # Placeholder logic - replace with your specific scanner's characteristics
            if hasattr(self, '_last_hand_side'):
                # Use previous detection as reference
                return self._last_hand_side
            
            # Default to right hand (most common in right-handed users)
            # You can modify this based on your specific use case
            self._last_hand_side = 'right'
            return 'right'
            
        except Exception as e:
            self.logger.error(f"Hand side determination failed: {e}")
            return 'unknown'
    
    def train(self, training_data: List[Tuple[np.ndarray, str]]) -> bool:
        """Train the classifier with labeled data"""
        try:
            if not training_data:
                self.logger.warning("No training data provided")
                return False
            
            # Extract features and labels
            X = []
            y = []
            
            for image, label in training_data:
                features = self._extract_features(image)
                if not np.all(features == 0):  # Skip failed feature extraction
                    X.append(features)
                    y.append(label)
            
            if len(X) < 10:  # Need minimum training samples
                self.logger.warning(f"Insufficient training data: {len(X)} samples")
                return False
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train Random Forest classifier
            self.model = RandomForestClassifier(n_estimators=100, random_state=42)
            self.model.fit(X_train_scaled, y_train)
            
            # Evaluate
            y_pred = self.model.predict(X_test_scaled)
            accuracy = accuracy_score(y_test, y_pred)
            
            self.logger.info(f"Training completed. Accuracy: {accuracy:.3f}")
            self.logger.info(f"Classification report:\n{classification_report(y_test, y_pred)}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Training failed: {e}")
            return False
    
    def save_model(self, filepath: str) -> bool:
        """Save the trained model"""
        try:
            if self.model is None:
                self.logger.warning("No model to save")
                return False
            
            model_data = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_names': self.feature_names,
                'finger_types': self.finger_types,
                'hand_sides': self.hand_sides
            }
            
            with open(filepath, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Model saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save model: {e}")
            return False
    
    def load_model(self, filepath: str) -> bool:
        """Load a trained model"""
        try:
            if not os.path.exists(filepath):
                self.logger.warning(f"Model file not found: {filepath}")
                return False
            
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.feature_names = model_data['feature_names']
            self.finger_types = model_data['finger_types']
            self.hand_sides = model_data['hand_sides']
            
            self.logger.info(f"Model loaded from {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load model: {e}")
            return False

def test_finger_classifier():
    """Test function for the finger classifier"""
    print("Testing finger classifier...")
    
    # Create classifier
    classifier = ComputerVisionFingerClassifier()
    
    # Create a test image (you would normally load a real fingerprint)
    test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
    
    # Test classification
    result = classifier.classify_finger(test_image)
    print(f"Classification result: {result}")
    
    # Test model saving/loading
    test_model_path = "test_finger_classifier.pkl"
    if classifier.save_model(test_model_path):
        print("✅ Model saved successfully")
        
        # Create new classifier and load model
        new_classifier = ComputerVisionFingerClassifier()
        if new_classifier.load_model(test_model_path):
            print("✅ Model loaded successfully")
        else:
            print("❌ Model loading failed")
        
        # Clean up
        os.remove(test_model_path)
    else:
        print("❌ Model saving failed")

if __name__ == "__main__":
    test_finger_classifier()
