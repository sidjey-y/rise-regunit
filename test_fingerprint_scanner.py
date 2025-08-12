#!/usr/bin/env python3
"""
Simple test script for the fingerprint scanner
Run this to test if your hardware fingerprint scanner is working.
"""

import cv2
import sys
import os

# Add Fingerprint directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'Fingerprint'))

def test_basic_camera():
    """Test basic camera functionality"""
    print("Testing basic camera functionality...")
    
    # Try different device indices
    for device_index in [0, 1, 2]:
        print(f"\nTrying camera device {device_index}...")
        
        try:
            cap = cv2.VideoCapture(device_index)
            if not cap.isOpened():
                print(f"  ❌ Device {device_index} not accessible")
                continue
            
            # Get camera properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            
            print(f"  ✅ Device {device_index} accessible")
            print(f"    Resolution: {width}x{height}")
            print(f"    FPS: {fps}")
            
            # Try to capture a frame
            ret, frame = cap.read()
            if ret:
                print(f"    ✅ Frame capture successful: {frame.shape}")
                
                # Show the frame briefly
                cv2.imshow(f"Camera {device_index} Test", frame)
                print("    Press any key to continue...")
                cv2.waitKey(3000)  # Wait 3 seconds
                cv2.destroyAllWindows()
                
            else:
                print(f"    ❌ Frame capture failed")
            
            cap.release()
            
        except Exception as e:
            print(f"  ❌ Error with device {device_index}: {e}")
    
    print("\nBasic camera test completed!")

def test_fingerprint_scanner():
    """Test the fingerprint scanner specifically"""
    print("\nTesting fingerprint scanner...")
    
    try:
        from Fingerprint.hardware_scanner import OpticalFingerprintScanner, FingerprintScannerInterface
        
        # Try to initialize scanner
        scanner = OpticalFingerprintScanner(device_index=0)
        if scanner.initialize():
            print("✅ Fingerprint scanner initialized successfully!")
            
            # Test capture
            print("Testing fingerprint capture...")
            fingerprint = scanner.capture_fingerprint()
            
            if fingerprint is not None:
                print(f"✅ Fingerprint captured: {fingerprint.shape}")
                
                # Show the captured fingerprint
                cv2.imshow("Captured Fingerprint", fingerprint)
                print("Press any key to continue...")
                cv2.waitKey(3000)
                cv2.destroyAllWindows()
                
            else:
                print("❌ Fingerprint capture failed")
            
            scanner.cleanup()
            
        else:
            print("❌ Fingerprint scanner initialization failed")
            
    except ImportError as e:
        print(f"❌ Could not import fingerprint scanner: {e}")
    except Exception as e:
        print(f"❌ Fingerprint scanner test failed: {e}")

def test_integrated_system():
    """Test the integrated system"""
    print("\nTesting integrated system...")
    
    try:
        from integrated_system import IntegratedBiometricSystem
        
        # Create system
        system = IntegratedBiometricSystem()
        
        # Initialize
        if system.initialize():
            print("✅ Integrated system initialized successfully!")
            
            # Show status
            system._show_system_status()
            
            # Cleanup
            system.cleanup()
            
        else:
            print("❌ Integrated system initialization failed")
            
    except ImportError as e:
        print(f"❌ Could not import integrated system: {e}")
    except Exception as e:
        print(f"❌ Integrated system test failed: {e}")

def main():
    """Main test function"""
    print("🔍 FINGERPRINT SCANNER TEST SUITE")
    print("="*50)
    
    print("This script will test:")
    print("1. Basic camera functionality")
    print("2. Fingerprint scanner specifically")
    print("3. Integrated system")
    print("="*50)
    
    try:
        # Test 1: Basic camera
        test_basic_camera()
        
        # Test 2: Fingerprint scanner
        test_fingerprint_scanner()
        
        # Test 3: Integrated system
        test_integrated_system()
        
        print("\n🎉 All tests completed!")
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite error: {e}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()






