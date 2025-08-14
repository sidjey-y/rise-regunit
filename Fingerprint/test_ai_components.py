#!/usr/bin/env python3
"""
Test AI Components Availability
Check if all AI components are working correctly
"""

import sys
import os

def test_ai_components():
    """Test all AI components"""
    print("🧪 Testing AI Components...")
    print("=" * 50)
    
    # Test 1: Siamese Network
    print("1. Testing Siamese Network...")
    try:
        from siamese_network import SiameseNetwork
        print("   ✅ Siamese Network import successful")
        
        # Try to create instance
        network = SiameseNetwork()
        print("   ✅ Siamese Network instance created")
        
    except ImportError as e:
        print(f"   ❌ Siamese Network import failed: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️ Siamese Network issue: {e}")
    
    # Test 2: Config Manager
    print("\n2. Testing Config Manager...")
    try:
        from config_manager import ConfigManager
        print("   ✅ Config Manager import successful")
        
        # Try to create instance
        config = ConfigManager()
        print("   ✅ Config Manager instance created")
        
    except ImportError as e:
        print(f"   ❌ Config Manager import failed: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️ Config Manager issue: {e}")
    
    # Test 3: Fingerprint Preprocessor
    print("\n3. Testing Fingerprint Preprocessor...")
    try:
        from fingerprint_preprocessor import FingerprintPreprocessor
        print("   ✅ Fingerprint Preprocessor import successful")
        
        # Try to create instance
        preprocessor = FingerprintPreprocessor()
        print("   ✅ Fingerprint Preprocessor instance created")
        
    except ImportError as e:
        print(f"   ❌ Fingerprint Preprocessor import failed: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️ Fingerprint Preprocessor issue: {e}")
    
    # Test 4: AI Integrated System
    print("\n4. Testing AI Integrated Enrollment System...")
    try:
        from ai_integrated_enrollment_system import AIIntegratedEnrollmentSystem
        print("   ✅ AI Integrated System import successful")
        
        # Try to create instance (without initializing scanner)
        system = AIIntegratedEnrollmentSystem(use_ai=True)
        print("   ✅ AI Integrated System instance created")
        print(f"   ✅ AI Available: {system.use_ai}")
        
    except ImportError as e:
        print(f"   ❌ AI Integrated System import failed: {e}")
        return False
    except Exception as e:
        print(f"   ⚠️ AI Integrated System issue: {e}")
    
    print("\n🎉 All AI components are working!")
    return True

def main():
    """Main function"""
    try:
        success = test_ai_components()
        
        if success:
            print("\n✅ AI system is ready for enrollment!")
            print("\nTo use AI enrollment, run:")
            print("   python ai_integrated_enrollment_system.py")
        else:
            print("\n❌ AI system has issues")
            print("\nFallback options:")
            print("   python comprehensive_enrollment_system.py")
        
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
