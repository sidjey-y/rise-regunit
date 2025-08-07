#!/usr/bin/env python3

import cv2
import sys
import os
import argparse
import logging
import numpy as np
import json
from typing import Optional, Dict, Any, List
from config_manager import ConfigManager
from fingerprint_preprocessor import FingerprintPreprocessor
from minutiae_extractor import MinutiaeExtractor
from fingerprint_database import FingerprintDatabase

class FingerprintUniquenessChecker:
    """
    Main system class that orchestrates all fingerprint uniqueness checking components.
    Follows OOP principles and uses YAML configuration.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the fingerprint uniqueness checker system.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = config_path
        self.config_manager = None
        self.preprocessor = None
        self.minutiae_extractor = None
        self.database = None
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def initialize(self) -> bool:
        """
        Initialize all system components.
        
        Returns:
            bool: True if initialization successful, False otherwise
        """
        try:
            # Initialize configuration manager
            self.config_manager = ConfigManager(self.config_path)
            
            # Validate configuration
            if not self.config_manager.validate_config():
                self.logger.error("Configuration validation failed")
                return False
            
            # Initialize preprocessor
            self.preprocessor = FingerprintPreprocessor(self.config_manager)
            if not self.preprocessor.initialize():
                self.logger.error("Preprocessor initialization failed")
                return False
            
            # Initialize minutiae extractor
            self.minutiae_extractor = MinutiaeExtractor()
            
            # Initialize database
            self.database = FingerprintDatabase(self.config_manager)
            if not self.database.initialize():
                self.logger.error("Database initialization failed")
                return False
            
            self.logger.info("Fingerprint uniqueness checker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"System initialization failed: {e}")
            return False
    
    def process_fingerprint(self, image_path: str, subject_id: str, 
                          finger_type: str, hand_side: str) -> Dict[str, Any]:
        """
        Process a single fingerprint for uniqueness checking.
        
        Args:
            image_path: Path to fingerprint image
            subject_id: Subject identifier
            finger_type: Type of finger (thumb, index, middle, ring, little)
            hand_side: Hand side (left, right)
            
        Returns:
            dict: Processing results including uniqueness check
        """
        try:
            # Load image
            image = cv2.imread(image_path)
            if image is None:
                return {'success': False, 'error': f'Could not load image: {image_path}'}
            
            # Preprocess image
            preprocess_results = self.preprocessor.process(image)
            if not preprocess_results:
                return {'success': False, 'error': 'Preprocessing failed'}
            
            # Extract minutiae points
            minutiae = self.minutiae_extractor.extract_minutiae(
                preprocess_results['resized_image']
            )
            
            if not minutiae:
                return {'success': False, 'error': 'Minutiae extraction failed'}
            
            # Check quality
            quality_score = self.preprocessor.get_quality_score(image)
            
            # Check for duplicates in database
            duplicate_check = self.database.process({
                'operation': 'check_duplicate',
                'minutiae': minutiae,
                'subject_id': subject_id
            })
            
            # Store in database if no duplicates found
            if duplicate_check.get('success') and not duplicate_check.get('is_duplicate'):
                store_result = self.database.process({
                    'operation': 'store',
                    'subject_id': subject_id,
                    'finger_type': finger_type,
                    'hand_side': hand_side,
                    'minutiae': minutiae,
                    'metadata': {
                        'image_path': image_path,
                        'original_size': image.shape,
                        'processed_size': preprocess_results['resized_image'].shape,
                        'minutiae_count': len(minutiae)
                    },
                    'quality_score': quality_score
                })
                
                if not store_result.get('success'):
                    return {'success': False, 'error': 'Failed to store fingerprint'}
            
            # Prepare results
            results = {
                'success': True,
                'subject_id': subject_id,
                'finger_type': finger_type,
                'hand_side': hand_side,
                'quality_score': quality_score,
                'is_duplicate': duplicate_check.get('is_duplicate', False),
                'duplicate_count': duplicate_check.get('duplicate_count', 0),
                'max_similarity': duplicate_check.get('max_similarity', 0.0),
                'best_match': duplicate_check.get('best_match'),
                'stored': not duplicate_check.get('is_duplicate', False),
                'minutiae_count': len(minutiae),
                'minutiae': minutiae,  # Include minutiae data for duplicate detection
                'preprocessing_metadata': preprocess_results['metadata']
            }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error processing fingerprint: {e}")
            return {'success': False, 'error': str(e)}
    
    def compare_fingerprints(self, image_path1: str, image_path2: str) -> Dict[str, Any]:
        try:
            # Load images
            image1 = cv2.imread(image_path1)
            image2 = cv2.imread(image_path2)
            
            if image1 is None or image2 is None:
                return {'success': False, 'error': 'Could not load one or both images'}
            
            # Preprocess images
            preprocess1 = self.preprocessor.process(image1)
            preprocess2 = self.preprocessor.process(image2)
            
            if not preprocess1 or not preprocess2:
                return {'success': False, 'error': 'Preprocessing failed'}
            
            # Compare using Siamese network
            comparison_result = self.siamese_network.process({
                'fingerprint1': preprocess1['model_input'],
                'fingerprint2': preprocess2['model_input']
            })
            
            # Add quality scores
            quality1 = self.preprocessor.get_quality_score(image1)
            quality2 = self.preprocessor.get_quality_score(image2)
            
            results = {
                'success': True,
                'similarity_score': comparison_result.get('similarity_score', 0.0),
                'is_same_person': comparison_result.get('is_same_person', False),
                'confidence': comparison_result.get('confidence', 0.0),
                'quality_score_1': quality1,
                'quality_score_2': quality2,
                'average_quality': (quality1 + quality2) / 2.0
            }
            return results
            
        except Exception as e:
            self.logger.error(f"Error comparing fingerprints: {e}")
            return {'success': False, 'error': str(e)}
    
    def train_model(self, dataset_path: str) -> Dict[str, Any]:
        try:
            self.logger.info("Training functionality requires implementation of data loading")
            return {
                'success': False,
                'error': 'Training functionality not fully implemented'
            }
            
        except Exception as e:
            self.logger.error(f"Error during training: {e}")
            return {'success': False, 'error': str(e)}
    
    def get_system_status(self) -> dict:
        """
        Get system status information.
        
        Returns:
            dict: System status
        """
        status = {
            'config_loaded': self.config_manager is not None,
            'preprocessor_initialized': self.preprocessor is not None and self.preprocessor.is_initialized(),
            'siamese_network_initialized': self.siamese_network is not None and self.siamese_network.is_initialized(),
            'database_initialized': self.database is not None and self.database.is_initialized()
        }
        
        if self.config_manager:
            status['config_path'] = self.config_path
            status['config_valid'] = self.config_manager.validate_config()
        
        if self.database:
            status['database_stats'] = self.database.get_database_stats()
        
        return status
    
    def cleanup(self) -> None:
        """Cleanup all system resources"""
        try:
            if self.preprocessor:
                self.preprocessor.cleanup()
            if self.siamese_network:
                self.siamese_network.cleanup()
            if self.database:
                self.database.cleanup()
            self.logger.info("System cleanup completed")
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    def __enter__(self):
        """Context manager entry"""
        if not self.initialize():
            raise RuntimeError("Failed to initialize fingerprint uniqueness checker")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()

def main():
    """Main application entry point"""
    parser = argparse.ArgumentParser(description='Fingerprint Uniqueness Checker')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--mode', type=str, choices=['process', 'compare', 'train', 'status'],
                       default='status', help='Operation mode')
    parser.add_argument('--image', type=str, help='Path to fingerprint image')
    parser.add_argument('--image2', type=str, help='Path to second fingerprint image (for compare mode)')
    parser.add_argument('--subject-id', type=str, help='Subject identifier')
    parser.add_argument('--finger-type', type=str, choices=['thumb', 'index', 'middle', 'ring', 'little'],
                       help='Type of finger')
    parser.add_argument('--hand-side', type=str, choices=['left', 'right'], help='Hand side')
    parser.add_argument('--dataset', type=str, help='Path to training dataset (for train mode)')
    
    args = parser.parse_args()
    
    # Create and run system
    with FingerprintUniquenessChecker(args.config) as checker:
        if args.mode == 'status':
            # Show system status
            status = checker.get_system_status()
            print("System Status:")
            for key, value in status.items():
                print(f"  {key}: {value}")
            return
        
        elif args.mode == 'process':
            if not all([args.image, args.subject_id, args.finger_type, args.hand_side]):
                print("Error: --image, --subject-id, --finger-type, and --hand-side are required for process mode")
                return
            
            result = checker.process_fingerprint(
                args.image, args.subject_id, args.finger_type, args.hand_side
            )
            print("Processing Result:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        
        elif args.mode == 'compare':
            if not all([args.image, args.image2]):
                print("Error: --image and --image2 are required for compare mode")
                return
            
            result = checker.compare_fingerprints(args.image, args.image2)
            print("Comparison Result:")
            for key, value in result.items():
                print(f"  {key}: {value}")
        
        elif args.mode == 'train':
            if not args.dataset:
                print("Error: --dataset is required for train mode")
                return
            
            result = checker.train_model(args.dataset)
            print("Training Result:")
            for key, value in result.items():
                print(f"  {key}: {value}")

if __name__ == "__main__":
    main() 

    #to do
    # 