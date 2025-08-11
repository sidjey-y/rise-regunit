#!/usr/bin/env python3
"""
Test script for photo quality check functionality
Run this to test the quality check system
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from face_detector import FaceDetector
from config_manager import ConfigManager
import cv2
import numpy as np

def test_quality_check():
    print("=" * 60)
    print("PHOTO QUALITY CHECK TEST")
    print("=" * 60)
    
    try:
        # Initialize components
        print("1. Initializing face detector...")
        config_manager = ConfigManager()
        face_detector = FaceDetector(config_manager)
        
        # Test with a sample image (you can replace this with your own image)
        print("2. Testing quality check with sample image...")
        
        # Create a test image (or load an existing one)
        test_image_path = "aproved_img/lastname_firstname_20250811_170417.jpg"
        
        if os.path.exists(test_image_path):
            print(f"   Using existing image: {test_image_path}")
            test_image = cv2.imread(test_image_path)
        else:
            print("   Creating test image...")
            # Create a test image with known characteristics
            test_image = np.zeros((480, 640, 3), dtype=np.uint8)
            # Add some test content
            cv2.rectangle(test_image, (100, 100), (540, 380), (255, 255, 255), -1)
            cv2.circle(test_image, (320, 240), 80, (0, 0, 255), -1)
        
        if test_image is None:
            print("❌ Failed to load/create test image")
            return
        
        print(f"   Image shape: {test_image.shape}")
        
        # Create mock landmarks (you would normally get these from face detection)
        print("3. Creating mock landmarks for testing...")
        h, w = test_image.shape[:2]
        mock_landmarks = np.array([
            [w//2 - 50, h//2 - 50],  # Left eye
            [w//2 + 50, h//2 - 50],  # Right eye
            [w//2, h//2 + 50],       # Nose
            [w//2, h//2 + 100],      # Mouth
            [w//2 - 80, h//2 - 80],  # Left face boundary
            [w//2 + 80, h//2 - 80],  # Right face boundary
            [w//2 - 80, h//2 + 120], # Left chin
            [w//2 + 80, h//2 + 120], # Right chin
        ], dtype=np.float32)
        
        print(f"   Mock landmarks shape: {mock_landmarks.shape}")
        
        # Test quality check
        print("4. Running quality check...")
        quality_issues = face_detector.analyze_captured_photo_quality(test_image, mock_landmarks)
        
        # Display results
        print("5. Quality Check Results:")
        print("=" * 40)
        if quality_issues:
            print(f"❌ {len(quality_issues)} quality issues found:")
            for i, issue in enumerate(quality_issues, 1):
                print(f"   {i}. {issue}")
            print("\nRecommendation: Photo needs improvement")
        else:
            print("✅ No quality issues found!")
            print("Recommendation: Photo is approved")
        
        print("=" * 60)
        print("Quality check test completed!")
        
    except Exception as e:
        print(f"❌ Error during test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_quality_check()
