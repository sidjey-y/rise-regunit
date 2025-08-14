#!/usr/bin/env python3
"""
Test and demonstrate AI-enhanced duplicate detection
"""

import logging
from ai_enhanced_enrollment_system import AIEnhancedEnrollmentSystem, Hand, FingerType

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_ai_duplicate_detection():
    """Test AI-enhanced duplicate detection system"""
    
    logger.info("🤖 AI-ENHANCED FINGERPRINT DUPLICATE DETECTION TEST")
    logger.info("=" * 70)
    logger.info("This test demonstrates the Siamese Neural Network for duplicate detection")
    logger.info("")
    
    # Initialize AI-enhanced system
    system = AIEnhancedEnrollmentSystem(use_ai=True)
    
    logger.info("🔧 System Configuration:")
    logger.info(f"   AI Mode: {'🤖 ENABLED' if system.use_ai else '🔧 TRADITIONAL'}")
    logger.info(f"   Scanner Port: {system.port}")
    logger.info(f"   Session Timeout: {system.session_timeout} seconds")
    logger.info("")
    
    if system.use_ai:
        logger.info("🧠 AI Components Available:")
        logger.info("   ✅ Siamese Neural Network")
        logger.info("   ✅ Fingerprint Preprocessor") 
        logger.info("   ✅ Configuration Manager")
        logger.info("   ✅ Advanced Embedding Extraction")
        logger.info("")
        
        logger.info("🎯 AI Capabilities:")
        logger.info("   • Deep learning-based similarity analysis")
        logger.info("   • 128-dimensional embedding space")
        logger.info("   • Contrastive loss optimization")
        logger.info("   • Higher accuracy than traditional methods")
        logger.info("   • Adaptive thresholds based on finger type")
        logger.info("")
    else:
        logger.warning("⚠️ AI components not available - using traditional methods")
        logger.info("   Install required packages: tensorflow, opencv-python")
        logger.info("")
    
    try:
        # Ask user what they want to test
        logger.info("🔍 Test Options:")
        logger.info("1. Full AI-enhanced enrollment (10 fingers)")
        logger.info("2. AI vs Traditional comparison demo")
        logger.info("3. Check AI model status")
        logger.info("")
        
        choice = input("Select test option (1/2/3): ").strip()
        
        if choice == "1":
            logger.info("🚀 Starting full AI-enhanced enrollment...")
            success = system.run_ai_enhanced_enrollment()
            
            if success:
                logger.info("✅ AI-enhanced enrollment completed successfully!")
            else:
                logger.error("❌ AI-enhanced enrollment failed")
                
        elif choice == "2":
            logger.info("🔄 AI vs Traditional comparison demo")
            logger.info("This would compare AI and traditional duplicate detection methods")
            logger.info("(Demo mode - requires actual fingerprint scanner)")
            
        elif choice == "3":
            logger.info("🔍 AI Model Status Check:")
            if system.siamese_network:
                logger.info("   ✅ Siamese network initialized")
                logger.info("   ✅ Model architecture ready")
                if system.siamese_network.model:
                    logger.info("   ✅ Model loaded and ready for inference")
                    logger.info(f"   📊 Model summary available")
                else:
                    logger.info("   ⚠️ Model needs to be built/loaded")
            else:
                logger.info("   ❌ Siamese network not available")
        else:
            logger.info("Invalid choice")
            
    except KeyboardInterrupt:
        logger.info("\nTest interrupted by user")
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        system.cleanup()

def main():
    """Main function"""
    try:
        test_ai_duplicate_detection()
    except Exception as e:
        logger.error(f"Test suite failed: {e}")

if __name__ == "__main__":
    main()
