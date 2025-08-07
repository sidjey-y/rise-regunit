#!/usr/bin/env python3

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import numpy as np
from collections import defaultdict
import time

from bozorth3_matcher import Bozorth3Matcher
from minutiae_extractor import MinutiaeExtractor
from fingerprint_preprocessor import FingerprintPreprocessor
from config_manager import ConfigManager

class DuplicateFingerprintDetector:
    """
    Detector for finding duplicate fingerprints within the Fingerprint folder.
    Uses Bozorth3 algorithm, xytheta format, and Boyer-Moore pattern matching.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config_manager = None
        self.bozorth3_matcher = None
        self.minutiae_extractor = None
        self.preprocessor = None
        self.logger = self._setup_logging()
        
        # Duplicate detection parameters
        self.similarity_threshold = 0.85
        self.min_match_score = 12
        self.max_processing_time = 300  # 5 minutes per comparison
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for duplicate detection"""
        logger = logging.getLogger('DuplicateDetector')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('duplicate_detection.log')
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
    
    def initialize(self) -> bool:
        """Initialize the duplicate detection system"""
        try:
            # Initialize configuration manager
            self.config_manager = ConfigManager(self.config_path)
            
            # Initialize components
            self.preprocessor = FingerprintPreprocessor(self.config_manager)
            self.minutiae_extractor = MinutiaeExtractor()
            
            # Initialize Bozorth3 matcher with configuration
            bozorth3_config = {
                'max_distance': 20.0,
                'max_angle_diff': 30.0,
                'min_matches': self.min_match_score,
                'max_score': 100.0
            }
            self.bozorth3_matcher = Bozorth3Matcher(bozorth3_config)
            
            self.logger.info("Duplicate fingerprint detector initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize duplicate detector: {e}")
            return False
    
    def scan_fingerprint_folder(self, base_path: str = "fingerprint_data") -> List[Dict[str, Any]]:
        """
        Scan the fingerprint data folder and collect all fingerprint files
        
        Args:
            base_path: Path to fingerprint data directory
            
        Returns:
            List of fingerprint file information
        """
        fingerprint_files = []
        base_path = Path(base_path)
        
        if not base_path.exists():
            self.logger.error(f"Base path does not exist: {base_path}")
            return fingerprint_files
        
        self.logger.info(f"Scanning fingerprint data in: {base_path}")
        
        # Scan all subject folders
        for subject_folder in sorted(base_path.iterdir()):
            if not subject_folder.is_dir():
                continue
                
            try:
                subject_id = subject_folder.name
                fingerprint_folder = subject_folder / "Fingerprint"
                
                if not fingerprint_folder.exists():
                    continue
                
                # Process all fingerprint files in the folder
                for fingerprint_file in fingerprint_folder.glob("*.BMP"):
                    file_info = self._parse_filename(fingerprint_file, subject_id)
                    if file_info:
                        fingerprint_files.append(file_info)
                        
            except Exception as e:
                self.logger.error(f"Error processing subject folder {subject_folder}: {e}")
        
        self.logger.info(f"Found {len(fingerprint_files)} fingerprint files")
        return fingerprint_files
    
    def _parse_filename(self, file_path: Path, subject_id: str) -> Optional[Dict[str, Any]]:
        """Parse fingerprint filename to extract metadata"""
        try:
            filename = file_path.stem
            parts = filename.split('__')
            
            if len(parts) >= 3:
                file_subject_id = parts[0]
                gender = parts[1]
                finger_info = parts[2]
                
                # Parse finger information
                finger_parts = finger_info.split('_')
                
                if len(finger_parts) >= 3:
                    hand_side = finger_parts[0]  # Left or Right
                    finger_type = finger_parts[1]  # thumb, index, middle, ring, little
                    
                    return {
                        'file_path': str(file_path),
                        'subject_id': file_subject_id,
                        'gender': gender,
                        'hand_side': hand_side,
                        'finger_type': finger_type,
                        'filename': filename,
                        'folder_subject_id': subject_id
                    }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error parsing filename {file_path}: {e}")
            return None
    
    def extract_minutiae_from_file(self, file_info: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
        """
        Extract minutiae points from a fingerprint file
        
        Args:
            file_info: File information dictionary
            
        Returns:
            List of minutiae points or None if extraction fails
        """
        try:
            import cv2
            
            # Load image
            image = cv2.imread(file_info['file_path'])
            if image is None:
                self.logger.error(f"Could not load image: {file_info['file_path']}")
                return None
            
            # Preprocess image
            preprocess_results = self.preprocessor.process(image)
            if not preprocess_results:
                self.logger.error(f"Preprocessing failed for: {file_info['file_path']}")
                return None
            
            # Extract minutiae
            minutiae = self.minutiae_extractor.extract_minutiae(image)
            
            if not minutiae:
                self.logger.warning(f"No minutiae extracted from: {file_info['file_path']}")
                return None
            
            self.logger.debug(f"Extracted {len(minutiae)} minutiae from {file_info['filename']}")
            return minutiae
            
        except Exception as e:
            self.logger.error(f"Error extracting minutiae from {file_info['file_path']}: {e}")
            return None
    
    def detect_duplicates(self, fingerprint_files: List[Dict[str, Any]], 
                         output_dir: str = "duplicate_results") -> Dict[str, Any]:
        """
        Detect duplicate fingerprints using Bozorth3 and Boyer-Moore
        
        Args:
            fingerprint_files: List of fingerprint file information
            output_dir: Directory to save results
            
        Returns:
            Dictionary with duplicate detection results
        """
        self.logger.info(f"Starting duplicate detection for {len(fingerprint_files)} files")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        # Extract minutiae for all files
        minutiae_cache = {}
        for file_info in fingerprint_files:
            minutiae = self.extract_minutiae_from_file(file_info)
            if minutiae:
                minutiae_cache[file_info['file_path']] = minutiae
        
        self.logger.info(f"Successfully extracted minutiae from {len(minutiae_cache)} files")
        
        # Find duplicates
        duplicates = []
        processed_pairs = set()
        
        file_paths = list(minutiae_cache.keys())
        total_comparisons = len(file_paths) * (len(file_paths) - 1) // 2
        comparison_count = 0
        
        self.logger.info(f"Starting {total_comparisons} fingerprint comparisons")
        
        for i, file_path1 in enumerate(file_paths):
            for j, file_path2 in enumerate(file_paths[i+1:], i+1):
                comparison_count += 1
                
                if comparison_count % 100 == 0:
                    self.logger.info(f"Progress: {comparison_count}/{total_comparisons} comparisons")
                
                # Create unique pair identifier
                pair_id = tuple(sorted([file_path1, file_path2]))
                if pair_id in processed_pairs:
                    continue
                
                processed_pairs.add(pair_id)
                
                # Get file information
                file_info1 = next(f for f in fingerprint_files if f['file_path'] == file_path1)
                file_info2 = next(f for f in fingerprint_files if f['file_path'] == file_path2)
                
                # Compare fingerprints using Bozorth3
                match_result = self.bozorth3_matcher.match_fingerprints(
                    minutiae_cache[file_path1],
                    minutiae_cache[file_path2]
                )
                
                # Check if it's a duplicate
                if match_result['is_match']:
                    duplicate_info = {
                        'file1': file_info1,
                        'file2': file_info2,
                        'match_score': match_result['match_score'],
                        'correspondence_count': match_result['correspondence_count'],
                        'pattern_match': match_result['pattern_match'],
                        'template_points': match_result['template_points'],
                        'query_points': match_result['query_points']
                    }
                    duplicates.append(duplicate_info)
                    
                    self.logger.warning(f"Duplicate detected: {file_info1['filename']} <-> {file_info2['filename']} (Score: {match_result['match_score']:.2f})")
        
        # Generate results
        results = {
            'total_files': len(fingerprint_files),
            'files_with_minutiae': len(minutiae_cache),
            'total_comparisons': comparison_count,
            'duplicates_found': len(duplicates),
            'duplicate_pairs': duplicates,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Save results
        self._save_results(results, output_path)
        
        self.logger.info(f"Duplicate detection completed. Found {len(duplicates)} duplicate pairs")
        return results
    
    def _save_results(self, results: Dict[str, Any], output_path: Path) -> None:
        """Save duplicate detection results to files"""
        try:
            # Save detailed results as JSON
            results_file = output_path / "duplicate_detection_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary report
            summary_file = output_path / "duplicate_summary.txt"
            with open(summary_file, 'w') as f:
                f.write("DUPLICATE FINGERPRINT DETECTION SUMMARY\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"Detection Date: {results['timestamp']}\n")
                f.write(f"Total Files Scanned: {results['total_files']}\n")
                f.write(f"Files with Minutiae: {results['files_with_minutiae']}\n")
                f.write(f"Total Comparisons: {results['total_comparisons']}\n")
                f.write(f"Duplicates Found: {results['duplicates_found']}\n\n")
                
                if results['duplicates_found'] > 0:
                    f.write("DUPLICATE PAIRS:\n")
                    f.write("-" * 30 + "\n")
                    for i, duplicate in enumerate(results['duplicate_pairs'], 1):
                        f.write(f"\n{i}. Duplicate Pair:\n")
                        f.write(f"   File 1: {duplicate['file1']['filename']}\n")
                        f.write(f"   File 2: {duplicate['file2']['filename']}\n")
                        f.write(f"   Match Score: {duplicate['match_score']:.2f}\n")
                        f.write(f"   Correspondence Count: {duplicate['correspondence_count']}\n")
                        f.write(f"   Pattern Match: {duplicate['pattern_match']}\n")
                else:
                    f.write("No duplicates found.\n")
            
            # Save xytheta format files for duplicates
            if results['duplicates_found'] > 0:
                xytheta_dir = output_path / "xytheta_files"
                xytheta_dir.mkdir(exist_ok=True)
                
                for i, duplicate in enumerate(results['duplicate_pairs']):
                    # Extract minutiae again for xytheta conversion
                    minutiae1 = self.extract_minutiae_from_file(duplicate['file1'])
                    minutiae2 = self.extract_minutiae_from_file(duplicate['file2'])
                    
                    if minutiae1:
                        xytheta_file1 = xytheta_dir / f"duplicate_{i+1}_file1.xytheta"
                        self.bozorth3_matcher.save_xytheta_format(minutiae1, str(xytheta_file1))
                    
                    if minutiae2:
                        xytheta_file2 = xytheta_dir / f"duplicate_{i+1}_file2.xytheta"
                        self.bozorth3_matcher.save_xytheta_format(minutiae2, str(xytheta_file2))
            
            self.logger.info(f"Results saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {e}")
    
    def analyze_duplicate_patterns(self, duplicates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze patterns in detected duplicates
        
        Args:
            duplicates: List of duplicate pairs
            
        Returns:
            Analysis results
        """
        analysis = {
            'total_duplicates': len(duplicates),
            'subject_duplicates': defaultdict(int),
            'finger_type_duplicates': defaultdict(int),
            'hand_side_duplicates': defaultdict(int),
            'score_distribution': [],
            'same_subject_duplicates': 0,
            'cross_subject_duplicates': 0
        }
        
        for duplicate in duplicates:
            file1 = duplicate['file1']
            file2 = duplicate['file2']
            
            # Count by subject
            analysis['subject_duplicates'][file1['subject_id']] += 1
            analysis['subject_duplicates'][file2['subject_id']] += 1
            
            # Count by finger type
            analysis['finger_type_duplicates'][file1['finger_type']] += 1
            analysis['finger_type_duplicates'][file2['finger_type']] += 1
            
            # Count by hand side
            analysis['hand_side_duplicates'][file1['hand_side']] += 1
            analysis['hand_side_duplicates'][file2['hand_side']] += 1
            
            # Score distribution
            analysis['score_distribution'].append(duplicate['match_score'])
            
            # Same vs cross subject
            if file1['subject_id'] == file2['subject_id']:
                analysis['same_subject_duplicates'] += 1
            else:
                analysis['cross_subject_duplicates'] += 1
        
        return analysis
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        try:
            if self.preprocessor:
                self.preprocessor.cleanup()
            self.logger.info("Duplicate detector cleanup completed")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")

def main():
    """Main function for command-line usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Detect duplicate fingerprints using Bozorth3 and Boyer-Moore')
    parser.add_argument('--data-path', default='fingerprint_data', 
                       help='Path to fingerprint data directory')
    parser.add_argument('--output-dir', default='duplicate_results',
                       help='Output directory for results')
    parser.add_argument('--config', default='config.yaml',
                       help='Configuration file path')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = DuplicateFingerprintDetector(args.config)
    
    if not detector.initialize():
        print("Failed to initialize duplicate detector")
        return
    
    try:
        # Scan fingerprint files
        fingerprint_files = detector.scan_fingerprint_folder(args.data_path)
        
        if not fingerprint_files:
            print("No fingerprint files found")
            return
        
        # Detect duplicates
        results = detector.detect_duplicates(fingerprint_files, args.output_dir)
        
        # Print summary
        print(f"\nDuplicate Detection Summary:")
        print(f"Total Files: {results['total_files']}")
        print(f"Files with Minutiae: {results['files_with_minutiae']}")
        print(f"Total Comparisons: {results['total_comparisons']}")
        print(f"Duplicates Found: {results['duplicates_found']}")
        
        if results['duplicates_found'] > 0:
            print(f"\nDuplicate pairs saved to: {args.output_dir}")
        
    except KeyboardInterrupt:
        print("\nDuplicate detection interrupted by user")
    except Exception as e:
        print(f"Error during duplicate detection: {e}")
    finally:
        detector.cleanup()

if __name__ == "__main__":
    main() 