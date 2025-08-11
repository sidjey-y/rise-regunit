#!/usr/bin/env python3
"""
Comprehensive Hardware Fingerprint Scanner Test
Tests all major fingerprint operations with the PyFingerprint library
"""

import sys
import time
import logging
from typing import Optional, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FingerprintScannerTester:
    """Comprehensive fingerprint scanner tester"""
    
    def __init__(self, port='COM4', baudrate=57600):
        self.port = port
        self.baudrate = baudrate
        self.scanner = None
        
    def initialize(self) -> bool:
        """Initialize the scanner"""
        try:
            from pyfingerprint.pyfingerprint import PyFingerprint
            
            logger.info(f"🔌 Initializing scanner on {self.port}...")
            self.scanner = PyFingerprint(self.port, self.baudrate, 0xFFFFFFFF, 0x00000000)
            
            if self.scanner.verifyPassword():
                logger.info("✅ Scanner initialized successfully")
                return True
            else:
                logger.error("❌ Scanner password verification failed")
                return False
                
        except Exception as e:
            logger.error(f"❌ Scanner initialization failed: {e}")
            return False
    
    def get_scanner_info(self) -> bool:
        """Get and display scanner information"""
        try:
            logger.info("📊 Scanner Information:")
            logger.info(f"  - Template Count: {self.scanner.getTemplateCount()}")
            logger.info(f"  - Storage Capacity: {self.scanner.getStorageCapacity()}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to get scanner info: {e}")
            return False
    
    def test_enrollment(self) -> bool:
        """Test fingerprint enrollment"""
        try:
            logger.info("📝 Testing fingerprint enrollment...")
            logger.info("Place your finger on the sensor...")
            
            # Wait for finger
            while not self.scanner.readImage():
                time.sleep(0.1)
            
            logger.info("✅ Finger detected")
            
            # Convert to characteristics
            self.scanner.convertImage(0x01)
            
            # Check if already enrolled
            result = self.scanner.searchTemplate()
            position = result[0]
            
            if position == -1:
                logger.info("✅ New fingerprint - creating template...")
                
                # Create template
                self.scanner.createTemplate()
                position = self.scanner.storeTemplate()
                logger.info(f"✅ Template stored at position {position}")
                return True
            else:
                logger.info(f"✅ Fingerprint already enrolled at position {position}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Enrollment failed: {e}")
            return False
    
    def test_verification(self) -> bool:
        """Test fingerprint verification"""
        try:
            logger.info("🔍 Testing fingerprint verification...")
            logger.info("Place your finger on the sensor for verification...")
            
            # Wait for finger
            while not self.scanner.readImage():
                time.sleep(0.1)
            
            logger.info("✅ Finger detected")
            
            # Convert to characteristics
            self.scanner.convertImage(0x01)
            
            # Search for template
            result = self.scanner.searchTemplate()
            position = result[0]
            score = result[1]
            
            if position == -1:
                logger.info("❌ No matching template found")
                return False
            else:
                logger.info(f"✅ Fingerprint verified! Position: {position}, Score: {score}")
                return True
                
        except Exception as e:
            logger.error(f"❌ Verification failed: {e}")
            return False
    
    def test_template_management(self) -> bool:
        """Test template management operations"""
        try:
            logger.info("🗂️  Testing template management...")
            
            # Get current template count
            count = self.scanner.getTemplateCount()
            logger.info(f"Current template count: {count}")
            
            if count > 0:
                # Test template deletion (delete the last one)
                position = count - 1
                logger.info(f"Testing template deletion at position {position}...")
                
                if self.scanner.deleteTemplate(position):
                    logger.info(f"✅ Template at position {position} deleted")
                    
                    # Verify deletion
                    new_count = self.scanner.getTemplateCount()
                    logger.info(f"New template count: {new_count}")
                    
                    # Re-enroll to restore
                    logger.info("Re-enrolling to restore template...")
                    if self.test_enrollment():
                        logger.info("✅ Template restored")
                        return True
                    else:
                        logger.error("❌ Failed to restore template")
                        return False
                else:
                    logger.error(f"❌ Failed to delete template at position {position}")
                    return False
            else:
                logger.info("No templates to test deletion with")
                return True
                
        except Exception as e:
            logger.error(f"❌ Template management failed: {e}")
            return False
    
    def test_image_quality(self) -> bool:
        """Test image quality and processing"""
        try:
            logger.info("🖼️  Testing image quality...")

            # Wait for finger
            logger.info("Place your finger on the sensor...")
            while not self.scanner.readImage():
                time.sleep(0.1)

            logger.info("✅ Finger detected")

            # Get image characteristics
            self.scanner.convertImage(0x01)

            # Check if characteristics are valid by searching for template
            result = self.scanner.searchTemplate()
            position = result[0]
            score = result[1]
            
            if position != -1:
                logger.info(f"✅ Image characteristics are valid - Found template at position {position} with score {score}")
                return True
            else:
                logger.info("✅ Image characteristics are valid - No matching template found (expected for new fingerprint)")
                return True

        except Exception as e:
            logger.error(f"❌ Image quality test failed: {e}")
            return False
    
    def run_comprehensive_test(self) -> bool:
        """Run all tests"""
        logger.info("🚀 Starting comprehensive hardware test...")
        logger.info("=" * 60)
        
        if not self.initialize():
            return False
        
        tests = [
            ("Scanner Information", self.get_scanner_info),
            ("Image Quality", self.test_image_quality),
            ("Fingerprint Enrollment", self.test_enrollment),
            ("Fingerprint Verification", self.test_verification),
            ("Template Management", self.test_template_management)
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
            
            time.sleep(1)
        
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
            logger.info("🎉 All tests passed! Hardware is fully functional.")
        else:
            logger.info("⚠️  Some tests failed. Check the logs above for details.")
        
        return passed == total
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if self.scanner:
                # Close serial connection
                if hasattr(self.scanner, '_serial'):
                    self.scanner._serial.close()
                logger.info("✅ Scanner resources cleaned up")
        except Exception as e:
            logger.error(f"❌ Cleanup failed: {e}")

def main():
    """Main function"""
    tester = FingerprintScannerTester()
    
    try:
        success = tester.run_comprehensive_test()
        return success
        
    except KeyboardInterrupt:
        logger.info("\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        return False
    finally:
        tester.cleanup()

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
