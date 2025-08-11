#!/usr/bin/env python3
"""
Hardware Fingerprint Scanner Test
Tests PyFingerprint library with COM4 configuration
"""

import sys
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_pyfingerprint():
    """Test PyFingerprint library"""
    try:
        from pyfingerprint.pyfingerprint import PyFingerprint
        logger.info("✅ PyFingerprint imported successfully")
        return True
    except ImportError:
        logger.error("❌ PyFingerprint not found. Install with: pip install pyfingerprint")
        return False

def test_serial():
    """Test serial connection"""
    try:
        import serial
        logger.info("✅ PySerial imported successfully")
        
        # Test COM4
        try:
            ser = serial.Serial('COM4', 57600, timeout=1)
            ser.close()
            logger.info("✅ COM4 accessible")
            return True
        except:
            logger.error("❌ COM4 not accessible")
            return False
    except ImportError:
        logger.error("❌ PySerial not found. Install with: pip install pyserial")
        return False

def test_scanner():
    """Test scanner hardware"""
    try:
        from pyfingerprint.pyfingerprint import PyFingerprint
        
        logger.info("🔌 Initializing scanner...")
        f = PyFingerprint('COM4', 57600, 0xFFFFFFFF, 0x00000000)
        
        if f.verifyPassword():
            logger.info("✅ Scanner ready")
            logger.info(f"Template count: {f.getTemplateCount()}")
            return True
        else:
            logger.error("❌ Scanner not ready")
            return False
            
    except Exception as e:
        logger.error(f"❌ Scanner error: {e}")
        return False

def main():
    """Run tests"""
    logger.info("🚀 Testing fingerprint scanner hardware...")
    
    tests = [
        ("PyFingerprint", test_pyfingerprint),
        ("Serial Connection", test_serial),
        ("Scanner Hardware", test_scanner)
    ]
    
    for name, test in tests:
        logger.info(f"\n🧪 Testing: {name}")
        if not test():
            logger.error(f"❌ {name} test failed")
            return False
        logger.info(f"✅ {name} test passed")
    
    logger.info("\n🎉 All tests passed!")
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
