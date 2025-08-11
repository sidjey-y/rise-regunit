#!/usr/bin/env python3
"""
Hardware Fingerprint Scanner Test Script
Tests the PyFingerprint library with a dedicated fingerprint scanner hardware.
"""

import sys
import time
import logging
from typing import Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_pyfingerprint_import():
    """Test if PyFingerprint library is available"""
    try:
        from pyfingerprint.pyfingerprint import PyFingerprint
        logger.info("✅ PyFingerprint library imported successfully")
        return True
    except ImportError as e:
        logger.error(f"❌ PyFingerprint library not found: {e}")
        logger.info("Please install it with: pip install pyfingerprint")
        return False

def test_serial_connection():
    """Test serial port connection"""
    try:
        import serial
        logger.info("✅ PySerial library imported successfully")
        
        # Test COM4 connection
        try:
            ser = serial.Serial('COM4', 57600, timeout=1)
            ser.close()
            logger.info("✅ COM4 port is accessible")
            return True
        except serial.SerialException as e:
            logger.error(f"❌ COM4 port not accessible: {e}")
            logger.info("Please check:")
            logger.info("1. Device is connected to COM4")
            logger.info("2. No other application is using the port")
            logger.info("3. Device drivers are installed")
            return False
            
    except ImportError:
        logger.error("❌ PySerial library not found")
        logger.info("Please install it with: pip install pyserial")
        return False

def test_fingerprint_scanner():
    """Test the fingerprint scanner hardware"""
    if not test_pyfingerprint_import():
        return False
    
    if not test_serial_connection():
        return False
    
    try:
        from pyfingerprint.pyfingerprint import PyFingerprint
        
        logger.info("🔌 Initializing fingerprint scanner...")
        logger.info("Parameters: COM4, 57600, 0xFFFFFFFF, 0x00000000")
        
        # Initialize scanner
        f = PyFingerprint('COM4', 57600, 0xFFFFFFFF, 0x00000000)
        
        if f.verifyPassword():
            logger.info("✅ Scanner password verified successfully")
        else:
            logger.error("❌ Scanner password verification failed")
            return False
        
        # Get scanner information
        logger.info("📊 Scanner Information:")
        logger.info(f"  - Address: {f.getAddress()}")
        logger.info(f"  - System ID: {f.getSystemId()}")
        logger.info(f"  - Library Version: {f.getLibraryVersion()}")
        logger.info(f"  - Template Count: {f.getTemplateCount()}")
        logger.info(f"  - Storage Capacity: {f.getStorageCapacity()}")
        
        # Test sensor status
        logger.info("🔍 Testing sensor status...")
        if f.verifyPassword():
            logger.info("✅ Sensor is ready")
        else:
            logger.error("❌ Sensor not ready")
            return False
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Scanner initialization failed: {e}")
        return False

def test_fingerprint_operations():
    """Test basic fingerprint operations"""
    try:
        from pyfingerprint.pyfingerprint import PyFingerprint
        
        logger.info("🔌 Testing fingerprint operations...")
        
        # Initialize scanner
        f = PyFingerprint('COM4', 57600, 0xFFFFFFFF, 0x00000000)
        
        if not f.verifyPassword():
            logger.error("❌ Cannot verify password")
            return False
        
        # Test enrollment process
        logger.info("📝 Testing enrollment process...")
        logger.info("Place your finger on the sensor...")
        
        # Wait for finger to be placed
        while not f.readImage():
            time.sleep(0.1)
        
        logger.info("✅ Finger detected on sensor")
        
        # Convert to characteristics
        f.convertImage(0x01)
        
        # Check if finger is already enrolled
        result = f.searchTemplate()
        positionNumber = result[0]
        
        if positionNumber == -1:
            logger.info("✅ New fingerprint - ready for enrollment")
            
            # Test template creation
            f.createTemplate()
            positionNumber = f.storeTemplate()
            logger.info(f"✅ Template stored at position {positionNumber}")
            
        else:
            logger.info(f"✅ Fingerprint already enrolled at position {positionNumber}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Fingerprint operations failed: {e}")
        return False

def test_verification():
    """Test fingerprint verification"""
    try:
        from pyfingerprint.pyfingerprint import PyFingerprint
        
        logger.info("🔍 Testing fingerprint verification...")
        
        # Initialize scanner
        f = PyFingerprint('COM4', 57600, 0xFFFFFFFF, 0x00000000)
        
        if not f.verifyPassword():
            logger.error("❌ Cannot verify password")
            return False
        
        # Wait for finger
        logger.info("Place your finger on the sensor for verification...")
        while not f.readImage():
            time.sleep(0.1)
        
        logger.info("✅ Finger detected")
        
        # Convert to characteristics
        f.convertImage(0x01)
        
        # Search for matching template
        result = f.searchTemplate()
        positionNumber = result[0]
        accuracyScore = result[1]
        
        if positionNumber == -1:
            logger.info("❌ No matching fingerprint found")
            return False
        else:
            logger.info(f"✅ Fingerprint verified! Position: {positionNumber}, Score: {accuracyScore}")
            return True
        
    except Exception as e:
        logger.error(f"❌ Verification failed: {e}")
        return False

def run_comprehensive_test():
    """Run comprehensive hardware test"""
    logger.info("🚀 Starting comprehensive fingerprint scanner hardware test...")
    logger.info("=" * 60)
    
    tests = [
        ("PyFingerprint Import", test_pyfingerprint_import),
        ("Serial Connection", test_serial_connection),
        ("Scanner Initialization", test_fingerprint_scanner),
        ("Basic Operations", test_fingerprint_operations),
        ("Verification", test_verification)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n🧪 Running: {test_name}")
        logger.info("-" * 40)
        
        try:
            result = test_func()
            results.append((test_name, result))
            
            if result:
                logger.info(f"✅ {test_name}: PASSED")
            else:
                logger.info(f"❌ {test_name}: FAILED")
                
        except Exception as e:
            logger.error(f"❌ {test_name}: ERROR - {e}")
            results.append((test_name, False))
        
        time.sleep(1)  # Brief pause between tests
    
    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 TEST SUMMARY")
    logger.info("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status} - {test_name}")
    
    logger.info(f"\nOverall Result: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Hardware is working correctly.")
    else:
        logger.info("⚠️  Some tests failed. Check the logs above for details.")
    
    return passed == total

def main():
    """Main function"""
    try:
        success = run_comprehensive_test()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
