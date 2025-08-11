#!/usr/bin/env python3
"""
Test Script for Fingerprint System Components
Run this to test individual components without full enrollment
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_finger_classifier():
    """Test the finger classification system"""
    print("🔍 Testing Finger Classifier...")
    try:
        from finger_classifier import ComputerVisionFingerClassifier
        import numpy as np
        
        # Create a test image (simulated fingerprint)
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        classifier = ComputerVisionFingerClassifier()
        result = classifier.classify_finger(test_image)
        
        print(f"✅ Classification result: {result['finger_type']}")
        print(f"   Hand side: {result['hand_side']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        print(f"   Method: {result['classification_method']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Finger classifier test failed: {e}")
        return False

def test_finger_validator():
    """Test the finger validation system"""
    print("\n🔍 Testing Finger Validator...")
    try:
        from finger_validator import FingerValidator
        import numpy as np
        
        # Create a test image
        test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)
        
        validator = FingerValidator()
        result = validator.validate_finger_scan(test_image, "index", "right")
        
        print(f"✅ Validation result: {result['is_valid']}")
        print(f"   Expected: right index")
        print(f"   Detected: {result['detected_finger']}")
        print(f"   Confidence: {result['confidence']:.2f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Finger validator test failed: {e}")
        return False

def test_hardware_connection():
    """Test hardware connection (without full initialization)"""
    print("\n🔍 Testing Hardware Connection...")
    try:
        import serial
        
        # Test if COM4 is accessible
        try:
            ser = serial.Serial('COM4', 57600, timeout=1)
            ser.close()
            print("✅ COM4 port is accessible")
            return True
        except Exception as e:
            print(f"❌ COM4 port not accessible: {e}")
            print("   Make sure your fingerprint scanner is connected to COM4")
            return False
            
    except ImportError:
        print("❌ PySerial not installed. Install with: pip install pyserial")
        return False
    except Exception as e:
        print(f"❌ Hardware test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🧪 Fingerprint System Component Tests")
    print("=" * 50)
    
    tests = [
        ("Finger Classifier", test_finger_classifier),
        ("Finger Validator", test_finger_validator),
        ("Hardware Connection", test_hardware_connection)
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 Testing: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 Test Results Summary:")
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"   {test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\nOverall: {passed}/{len(results)} tests passed")
    
    if passed == len(results):
        print("🎉 All tests passed! System is ready for enrollment.")
        print("\nTo start enrollment, run: python run_enrollment.py")
    else:
        print("⚠️  Some tests failed. Please check the errors above.")
    
    return passed == len(results)

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
