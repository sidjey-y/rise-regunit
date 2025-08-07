#!/usr/bin/env python3

import os
import sys
import logging
import argparse
from face_embedding_extractor import FaceEmbeddingExtractor

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('face_embeddings.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )

#main function for face extraction of embeddings
#perform comparison
def main():
    parser = argparse.ArgumentParser(description='Face Embedding Extraction and Comparison')
    parser.add_argument('--image', type=str, default='face_embeddings.jpg',
                       help='Path to the face image for embedding extraction')
    parser.add_argument('--approved-dir', type=str, default='aproved_img',
                       help='Directory containing approved images')
    parser.add_argument('--model', type=str, default='VGG-Face',
                       choices=['VGG-Face', 'Facenet', 'OpenFace', 'DeepID', 'ArcFace'],
                       help='DeepFace model to use')
    parser.add_argument('--metric', type=str, default='cosine',
                       choices=['cosine', 'euclidean', 'manhattan'],
                       help='Distance metric for comparison')
    parser.add_argument('--threshold', type=float, default=0.6,
                       help='Similarity threshold for matching')
    parser.add_argument('--save-embeddings', action='store_true',
                       help='Save extracted embeddings to file')
    parser.add_argument('--output-dir', type=str, default='embeddings',
                       help='Output directory for saved embeddings')
    
    args = parser.parse_args()
    
    #setup logging
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Check if input image exists
    if not os.path.exists(args.image):
        logger.error(f"Input image not found: {args.image}")
        sys.exit(1)
    
    # Check if approved directory exists
    if not os.path.exists(args.approved_dir):
        logger.error(f"Approved directory not found: {args.approved_dir}")
        sys.exit(1)
    
    try:
        # initialize face embedding extractor
        logger.info(f"Initializing face embedding extractor with {args.model} model")
        extractor = FaceEmbeddingExtractor(
            model_name=args.model,
            distance_metric=args.metric
        )
        
        # Set custom threshold
        extractor.similarity_threshold = args.threshold
        
        # Verify face quality of input image
        logger.info(f"Verifying face quality of {args.image}")
        quality = extractor.verify_face_quality(args.image)
        
        if 'error' in quality:
            logger.error(f"Face quality verification failed: {quality['error']}")
            sys.exit(1)
        
        logger.info(f"Face quality assessment: {quality}")
        
        if not quality['is_suitable']:
            logger.warning("Image may not be suitable for embedding extraction")
            logger.warning(f"Recommendations: {quality['recommendations']}")
        
        # Extract embedding from input image
        logger.info(f"Extracting embedding from {args.image}")
        input_embedding = extractor.extract_embedding(args.image)
        
        if input_embedding is None:
            logger.error("Failed to extract embedding from input image")
            sys.exit(1)
        
        logger.info(f"Successfully extracted embedding (shape: {input_embedding.shape})")
        
        # Extract embeddings from approved images
        logger.info(f"Extracting embeddings from approved images in {args.approved_dir}")
        approved_embeddings = extractor.extract_embeddings_from_directory(args.approved_dir)
        
        if not approved_embeddings:
            logger.error("No embeddings extracted from approved images")
            sys.exit(1)
        
        logger.info(f"Extracted {len(approved_embeddings)} embeddings from approved images")
        
        # compare input image with approved images
        logger.info("Comparing input image with approved images")
        comparison_results = extractor.compare_with_approved_images(
            args.image, 
            args.approved_dir
        )
        
        # Display results
        print("\n" + "="*60)
        print("FACE EMBEDDING COMPARISON RESULTS")
        print("="*60)
        print(f"Input Image: {args.image}")
        print(f"Model Used: {args.model}")
        print(f"Distance Metric: {args.metric}")
        print(f"Similarity Threshold: {args.threshold}")
        print(f"Timestamp: {comparison_results['timestamp']}")
        
        if 'error' in comparison_results:
            print(f"\nERROR: {comparison_results['error']}")
        else:
            print(f"\nTotal Approved Images: {len(comparison_results['matches'])}")
            print(f"Approval Status: {'✓ APPROVED' if comparison_results['is_approved'] else '✗ NOT APPROVED'}")
            
            if comparison_results['best_match']:
                best = comparison_results['best_match']
                print(f"\nBest Match:")
                print(f"  File: {best['filename']}")
                print(f"  Similarity Score: {best['similarity_score']:.4f}")
            
            print(f"\nDetailed Comparison Results:")
            print("-" * 40)
            for match in comparison_results['matches']:
                status = "✓ MATCH" if match['is_match'] else "✗ NO MATCH"
                print(f"{match['filename']}: {match['similarity_score']:.4f} ({status})")
        
        # Save embeddings if requested
        if args.save_embeddings:
            os.makedirs(args.output_dir, exist_ok=True)
            
            # Save input embedding
            input_embeddings = {os.path.basename(args.image): input_embedding}
            input_path = os.path.join(args.output_dir, 'input_embeddings.json')
            if extractor.save_embeddings(input_embeddings, input_path):
                logger.info(f"Saved input embeddings to {input_path}")
            
            # Save approved embeddings
            approved_path = os.path.join(args.output_dir, 'approved_embeddings.json')
            if extractor.save_embeddings(approved_embeddings, approved_path):
                logger.info(f"Saved approved embeddings to {approved_path}")
            
            # Save comparison results
            results_path = os.path.join(args.output_dir, 'comparison_results.json')
            import json
            with open(results_path, 'w') as f:
                # Convert numpy arrays to lists for JSON serialization
                serializable_results = comparison_results.copy()
                if 'matches' in serializable_results:
                    for match in serializable_results['matches']:
                        if 'similarity_score' in match:
                            match['similarity_score'] = float(match['similarity_score'])
                json.dump(serializable_results, f, indent=2)
            logger.info(f"Saved comparison results to {results_path}")
        
        print("\n" + "="*60)
        logger.info("Face embedding extraction and comparison completed successfully")
        
    except Exception as e:
        logger.error(f"Error during face embedding extraction: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 