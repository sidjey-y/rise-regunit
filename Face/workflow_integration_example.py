#!/usr/bin/env python3

import os
import logging
from automated_face_verification import verify_captured_face_automated

def setup_logging():
    """Setup logging for the workflow"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def automated_workflow_example():
    """
    Example of how to integrate face verification into your existing workflow:
    
    compliance > liveness > autocapture > face_verification > save_photo > success
    """
    
    # Setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Simulate your workflow steps
    logger.info("=== AUTOMATED WORKFLOW START ===")
    
    # Step 1: Compliance check (your existing code)
    logger.info("1. ✅ Compliance check passed")
    
    # Step 2: Liveness detection (your existing code)
    logger.info("2. ✅ Liveness detection passed")
    
    # Step 3: Auto-capture photo (your existing code)
    captured_image_path = "raw_pic/face_embeddings.jpg"  # Using your actual image for testing
    logger.info(f"3. ✅ Photo captured: {captured_image_path}")
    
    # Step 4: Face verification (NEW - integrated step)
    logger.info("4. 🔍 Starting face verification...")
    
    try:
        # Call the automated face verification
        verification_results = verify_captured_face_automated(
            captured_image_path=captured_image_path,
            approved_dir="aproved_img",
            raw_pic_dir="raw_pic",
            similarity_threshold=0.6
        )
        
        # Check if verification passed
        if verification_results['verification_passed']:
            logger.info("4. ✅ Face verification PASSED")
            logger.info(f"   Best match: {verification_results['best_match']}")
            logger.info(f"   Similarity score: {verification_results['similarity_score']:.3f}")
            
            # Step 5: Save photo (your existing code)
            logger.info("5. ✅ Photo saved successfully")
            
            # Step 6: Success
            logger.info("6. 🎉 WORKFLOW COMPLETED SUCCESSFULLY")
            
            return True
            
        else:
            logger.error("4. ❌ Face verification FAILED")
            logger.error(f"   Best match: {verification_results['best_match']}")
            logger.error(f"   Similarity score: {verification_results['similarity_score']:.3f}")
            logger.error("   Workflow stopped - face verification failed")
            
            return False
            
    except Exception as e:
        logger.error(f"4. ❌ Face verification ERROR: {e}")
        logger.error("   Workflow stopped - face verification error")
        return False

# Function to integrate into your existing camera interface
def integrate_with_camera_interface(captured_image_path: str) -> bool:
    """
    Function to integrate face verification into your existing camera interface.
    Call this function after autocapture in your main workflow.
    
    Args:
        captured_image_path: Path to the captured image
        
    Returns:
        bool: True if verification passed, False otherwise
    """
    try:
        # Perform face verification
        verification_results = verify_captured_face_automated(
            captured_image_path=captured_image_path,
            approved_dir="aproved_img",
            raw_pic_dir="raw_pic",
            similarity_threshold=0.6
        )
        
        # Return verification status
        return verification_results['verification_passed']
        
    except Exception as e:
        logging.error(f"Face verification error: {e}")
        return False

# Example usage in your existing code
def example_usage_in_existing_code():
    """
    Example of how to add this to your existing camera_interface.py or main_try.py
    """
    
    # Your existing compliance and liveness code here...
    
    # After autocapture, add this:
    captured_image = "path/to/captured/image.jpg"
    
    # Call the verification function
    if integrate_with_camera_interface(captured_image):
        print("✅ Face verification passed - saving photo")
        # Your existing save photo code here...
        print("🎉 Success!")
    else:
        print("❌ Face verification failed - retry needed")
        # Your existing retry logic here...

if __name__ == "__main__":
    # Run the example workflow
    automated_workflow_example() 