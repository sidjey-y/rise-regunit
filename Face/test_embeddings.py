#!/usr/bin/env python3

import os
import sys
import logging
from face_embedding_extractor import FaceEmbeddingExtractor

def test_face_embeddings():
    """Test face embedding extraction functionality"""
    
    # Setup basic logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Test image path (you can change this to your actual image)
    test_image = "face_embeddings.jpg"
    approved_dir = "aproved_img"
    
    # Check if test image exists
    if not os.path.exists(test_image):
        logger.error(f"Test image not found: {test_image}")
        logger.info("Please place your face image as 'face_embeddings.jpg' in the current directory")
        return False
    
    # Check if approved directory exists
    if not os.path.exists(approved_dir):
        logger.error(f"Approved directory not found: {approved_dir}")
        return False
    
    try:
        # Initialize face embedding extractor
        logger.info("Initializing face embedding extractor...")
        extractor = FaceEmbeddingExtractor(model_name="VGG-Face", distance_metric="cosine")
        
        # Test 1: Verify face quality
        logger.info("Testing face quality verification...")
        quality = extractor.verify_face_quality(test_image)
        print(f"Face Quality Assessment: {quality}")
        
        # Test 2: Extract embedding from test image
        logger.info("Testing embedding extraction...")
        embedding = extractor.extract_embedding(test_image)
        
        if embedding is not None:
            print(f"✓ Successfully extracted embedding (shape: {embedding.shape})")
        else:
            print("✗ Failed to extract embedding")
            return False
        
        # Test 3: Extract embeddings from approved images
        logger.info("Testing approved images embedding extraction...")
        approved_embeddings = extractor.extract_embeddings_from_directory(approved_dir)
        
        if approved_embeddings:
            print(f"✓ Extracted {len(approved_embeddings)} embeddings from approved images")
            for filename in approved_embeddings.keys():
                print(f"  - {filename}")
        else:
            print("✗ No embeddings extracted from approved images")
            return False
        
        # Test 4: Compare test image with approved images
        logger.info("Testing image comparison...")
        comparison_results = extractor.compare_with_approved_images(test_image, approved_dir)
        
        if 'error' in comparison_results:
            print(f"✗ Comparison failed: {comparison_results['error']}")
            return False
        
        print(f"✓ Comparison completed successfully")
        print(f"  - Approval Status: {'APPROVED' if comparison_results['is_approved'] else 'NOT APPROVED'}")
        
        if comparison_results['best_match']:
            best = comparison_results['best_match']
            print(f"  - Best Match: {best['filename']} (similarity: {best['similarity_score']:.4f})")
        
        # Test 5: Save and load embeddings
        logger.info("Testing embedding save/load functionality...")
        test_embeddings = {test_image: embedding}
        
        # Save embeddings
        if extractor.save_embeddings(test_embeddings, "test_embeddings.json"):
            print("✓ Successfully saved embeddings")
            
            # Load embeddings
            loaded_embeddings = extractor.load_embeddings("test_embeddings.json")
            if loaded_embeddings:
                print("✓ Successfully loaded embeddings")
                print(f"  - Loaded {len(loaded_embeddings)} embeddings")
            else:
                print("✗ Failed to load embeddings")
        
        # Clean up test file
        if os.path.exists("test_embeddings.json"):
            os.remove("test_embeddings.json")
        
        print("\n" + "="*50)
        print("ALL TESTS PASSED! ✓")
        print("="*50)
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {e}")
        return False

if __name__ == "__main__":
    success = test_face_embeddings()
    sys.exit(0 if success else 1) 