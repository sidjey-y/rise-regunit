#!/usr/bin/env python3
"""
Test if Scanner is Clean
Quick test to verify no templates exist before enrollment
"""

import sys
import time
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_scanner_clean():
    """Test if the scanner has been properly cleaned of all templates"""
    try:
        from pyfingerprint.pyfingerprint import PyFingerprint
        
        logger.info("🧪 TESTING SCANNER CLEANLINESS")
        logger.info("=" * 50)
        
        # Initialize scanner
        logger.info("Initializing scanner...")
        scanner = PyFingerprint('COM4', 57600, 0xFFFFFFFF, 0x00000000)
        
        if not scanner.verifyPassword():
            logger.error("❌ Scanner password verification failed")
            return False
        
        logger.info("✅ Scanner connected successfully")
        
        # Check template count
        template_count = scanner.getTemplateCount()
        logger.info(f"📊 Current template count: {template_count}")
        
        if template_count == 0:
            logger.info("🎉 SCANNER IS CLEAN!")
            logger.info("   No templates found - ready for fresh enrollment")
            return True
        else:
            logger.warning(f"⚠️  SCANNER NOT CLEAN!")
            logger.warning(f"   Found {template_count} remaining templates")
            
            # Try to find which positions have templates
            found_positions = []
            logger.info("🔍 Scanning for template positions...")
            
            for pos in range(min(template_count + 50, 1000)):  # Check a reasonable range
                try:
                    scanner.loadTemplate(pos)
                    found_positions.append(pos)
                    logger.info(f"   📍 Template found at position {pos}")
                    
                    if len(found_positions) >= 20:  # Limit output
                        logger.info("   ... (limiting output to first 20 found)")
                        break
                except:
                    continue
            
            logger.warning(f"Found templates at {len(found_positions)} positions")
            logger.warning("Please run 'python clear_and_restart.py' again")
            return False
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    success = test_scanner_clean()
    
    if success:
        print("\n✅ SCANNER IS READY FOR ENROLLMENT!")
        print("You can now run 'python run_enrollment.py' safely")
    else:
        print("\n❌ SCANNER NEEDS CLEANING!")
        print("Run 'python clear_and_restart.py' first")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
