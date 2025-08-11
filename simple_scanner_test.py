#!/usr/bin/env python3
"""
Simple scanner accessibility test - no camera windows
Tests if the fingerprint scanner hardware is readable/accessible
"""

import cv2
import sys
import os

def test_scanner_accessibility():
    """Test if scanner hardware is accessible without opening windows"""
    print("🔍 Testing Scanner Accessibility (No Windows)")
    print("=" * 50)
    
    # Test different device indices
    for device_index in [0, 1, 2, 3]:
        print(f"\nTesting device index {device_index}...")
        
        try:
            # Try to open the device
            cap = cv2.VideoCapture(device_index)
            
            if not cap.isOpened():
                print(f"  ❌ Device {device_index}: Not accessible")
                continue
            
            # Get device properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            brightness = cap.get(cv2.CAP_PROP_BRIGHTNESS)
            contrast = cap.get(cv2.CAP_PROP_CONTRAST)
            exposure = cap.get(cv2.CAP_PROP_EXPOSURE)
            
            print(f"  ✅ Device {device_index}: ACCESSIBLE")
            print(f"    Resolution: {width}x{height}")
            print(f"    FPS: {fps}")
            print(f"    Brightness: {brightness}")
            print(f"    Contrast: {contrast}")
            print(f"    Exposure: {exposure}")
            
            # Try to read a frame (without displaying)
            ret, frame = cap.read()
            if ret:
                print(f"    ✅ Frame read successful: {frame.shape}")
                print(f"    Frame data type: {frame.dtype}")
                print(f"    Frame range: {frame.min()} to {frame.max()}")
            else:
                print(f"    ❌ Frame read failed")
            
            # Close the device
            cap.release()
            
        except Exception as e:
            print(f"  ❌ Device {device_index}: Error - {e}")
    
    print("\n" + "=" * 50)
    print("Scanner accessibility test completed!")

def test_opencv_devices():
    """Test what OpenCV can detect"""
    print("\n🔍 OpenCV Device Detection Test")
    print("=" * 30)
    
    try:
        # Try to get backend info
        backend = cv2.getBackendName()
        print(f"OpenCV backend: {backend}")
        
        # Try to get build information
        build_info = cv2.getBuildInformation()
        print("OpenCV build info available: ✅")
        
    except Exception as e:
        print(f"OpenCV info error: {e}")

def main():
    """Main test function"""
    print("🔍 SIMPLE SCANNER ACCESSIBILITY TEST")
    print("This test checks if your scanner hardware is accessible")
    print("No camera windows will be opened")
    print("=" * 60)
    
    try:
        # Test 1: Scanner accessibility
        test_scanner_accessibility()
        
        # Test 2: OpenCV device detection
        test_opencv_devices()
        
        print("\n🎉 All tests completed!")
        print("\nIf you see 'ACCESSIBLE' for any device, your scanner is working!")
        print("If all devices show 'Not accessible', there might be a hardware/driver issue.")
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()





