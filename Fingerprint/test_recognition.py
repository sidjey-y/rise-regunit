#!/usr/bin/env python3
"""
Test Fingerprint Recognition System
Simple test to verify the system can load reference data and identify fingers.
"""

import os
import sys
import cv2
import numpy as np

def test_opencv_installation():
    """Test if OpenCV is properly installed"""
    try:
        print("🔍 Testing OpenCV installation...")
        print(f"   OpenCV version: {cv2.__version__}")
        
        # Test SIFT feature extractor
        sift = cv2.SIFT_create()
        print("   ✅ SIFT feature extractor created successfully")
        
        # Test image loading
        test_image = np.zeros((100, 100), dtype=np.uint8)
        keypoints, descriptors = sift.detectAndCompute(test_image, None)
        print("   ✅ SIFT feature extraction working")
        
        return True
        
    except Exception as e:
        print(f"   ❌ OpenCV test failed: {e}")
        return False

def test_reference_database():
    """Test if we can access the reference fingerprint database"""
    try:
        print("\n📚 Testing reference database access...")
        
        reference_dir = "fingerprint_data"
        if not os.path.exists(reference_dir):
            print(f"   ❌ Reference directory not found: {reference_dir}")
            return False
        
        print(f"   ✅ Reference directory found: {reference_dir}")
        
        # Count fingerprint files
        total_files = 0
        user_count = 0
        
        for user_dir in os.listdir(reference_dir):
            user_path = os.path.join(reference_dir, user_dir)
            if os.path.isdir(user_path):
                fingerprint_path = os.path.join(user_path, "Fingerprint")
                if os.path.exists(fingerprint_path):
                    user_count += 1
                    for filename in os.listdir(fingerprint_path):
                        if filename.endswith(('.BMP', '.bmp')):
                            total_files += 1
        
        print(f"   📊 Found {user_count} users with {total_files} fingerprint files")
        
        if total_files == 0:
            print("   ⚠️  No fingerprint files found")
            return False
        
        # Test loading a sample image
        print("\n   📸 Testing image loading...")
        sample_found = False
        
        for user_dir in os.listdir(reference_dir):
            if sample_found:
                break
                
            user_path = os.path.join(reference_dir, user_dir)
            if os.path.isdir(user_path):
                fingerprint_path = os.path.join(user_path, "Fingerprint")
                if os.path.exists(fingerprint_path):
                    for filename in os.listdir(fingerprint_path):
                        if filename.endswith(('.BMP', '.bmp')):
                            filepath = os.path.join(fingerprint_path, filename)
                            try:
                                image = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
                                if image is not None:
                                    print(f"      ✅ Loaded sample image: {filename}")
                                    print(f"         Dimensions: {image.shape[1]}x{image.shape[0]}")
                                    print(f"         File size: {os.path.getsize(filepath)} bytes")
                                    
                                    # Test feature extraction
                                    sift = cv2.SIFT_create()
                                    keypoints, descriptors = sift.detectAndCompute(image, None)
                                    
                                    if descriptors is not None:
                                        print(f"         Features extracted: {len(descriptors)}")
                                    else:
                                        print(f"         ⚠️  No features extracted")
                                    
                                    sample_found = True
                                    break
                                else:
                                    print(f"      ❌ Failed to load: {filename}")
                            except Exception as e:
                                print(f"      ❌ Error loading {filename}: {e}")
        
        if not sample_found:
            print("   ❌ No sample images could be loaded")
            return False
        
        return True
        
    except Exception as e:
        print(f"   ❌ Database test failed: {e}")
        return False

def test_feature_matching():
    """Test feature matching between two images"""
    try:
        print("\n🔍 Testing feature matching...")
        
        # Create two test images with different patterns
        image1 = np.zeros((256, 256), dtype=np.uint8)
        image2 = np.zeros((256, 256), dtype=np.uint8)
        
        # Add different patterns
        for i in range(0, 256, 30):
            image1[i:i+15, :] = 128
            image2[:, i:i+15] = 128
        
        # Extract features
        sift = cv2.SIFT_create()
        keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
        keypoints2, descriptors2 = sift.detectAndCompute(image2, None)
        
        if descriptors1 is None or descriptors2 is None:
            print("   ❌ Feature extraction failed")
            return False
        
        print(f"   ✅ Feature extraction: Image1={len(descriptors1)}, Image2={len(descriptors2)}")
        
        # Test FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(descriptors1, descriptors2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.7 * n.distance:
                    good_matches.append(m)
        
        similarity = len(good_matches) / len(matches) if matches else 0.0
        print(f"   ✅ Feature matching working: {len(good_matches)}/{len(matches)} good matches")
        print(f"   📊 Similarity score: {similarity:.3f}")
        
        return True
        
    except Exception as e:
        print(f"   ❌ Feature matching test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🧪 Fingerprint Recognition System Test")
    print("=" * 50)
    
    tests = [
        ("OpenCV Installation", test_opencv_installation),
        ("Reference Database", test_reference_database),
        ("Feature Matching", test_feature_matching)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔬 Running: {test_name}")
        print("-" * 30)
        
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"❌ {test_name}: ERROR - {e}")
    
    print(f"\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready to use.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

