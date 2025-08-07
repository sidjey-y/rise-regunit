#!/usr/bin/env python3

import os
import sys
import argparse
import json
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
from pathlib import Path
import logging

# Import the main fingerprint checker
from main import FingerprintUniquenessChecker

class BatchFingerprintProcessor:
    """
    Batch processor for fingerprint uniqueness checking across multiple folders.
    Processes all fingerprint images in the dataset and provides comprehensive reporting.
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.checker = None
        self.results = []
        self.logger = self._setup_logging()
        
    def _setup_logging(self) -> logging.Logger:
        """Setup logging for batch processing"""
        logger = logging.getLogger('BatchProcessor')
        logger.setLevel(logging.INFO)
        
        # Create handlers
        c_handler = logging.StreamHandler()
        f_handler = logging.FileHandler('batch_processing.log')
        
        # Create formatters and add it to handlers
        log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(log_format)
        f_handler.setFormatter(log_format)
        
        # Add handlers to the logger
        logger.addHandler(c_handler)
        logger.addHandler(f_handler)
        
        return logger
    
    def initialize(self) -> bool:
        """Initialize the fingerprint checker system"""
        try:
            self.checker = FingerprintUniquenessChecker(self.config_path)
            return self.checker.initialize()
        except Exception as e:
            self.logger.error(f"Failed to initialize system: {e}")
            return False
    
    def get_fingerprint_files(self, base_path: str) -> List[Dict[str, str]]:
        """
        Scan the fingerprint data directory and return all fingerprint files with metadata.
        
        Args:
            base_path: Base path to fingerprint data directory
            
        Returns:
            List of dictionaries containing file information
        """
        fingerprint_files = []
        base_path = Path(base_path)
        
        if not base_path.exists():
            self.logger.error(f"Base path does not exist: {base_path}")
            return fingerprint_files
        
        # Scan all subject folders (1-48)
        for subject_folder in sorted(base_path.iterdir()):
            if not subject_folder.is_dir():
                continue
                
            try:
                subject_id = subject_folder.name
                fingerprint_folder = subject_folder / "Fingerprint"
                
                if not fingerprint_folder.exists():
                    self.logger.warning(f"No Fingerprint folder found in {subject_id}")
                    continue
                
                # Process all fingerprint files in the folder
                for fingerprint_file in fingerprint_folder.glob("*.BMP"):
                    # Parse filename to extract metadata
                    filename = fingerprint_file.stem
                    parts = filename.split('__')
                    
                    if len(parts) == 2:
                        file_subject_id = parts[0]
                        finger_info = parts[1]  # e.g., "M_Left_index_finger"
                        
                        # Parse finger information
                        finger_parts = finger_info.split('_')
                        
                        if len(finger_parts) >= 4:
                            # Format: M_Left_index_finger
                            gender = finger_parts[0]  # M/F
                            hand_side = finger_parts[1]  # Left/Right
                            finger_type = finger_parts[2]  # index, middle, ring, little, thumb
                            
                            fingerprint_files.append({
                                'file_path': str(fingerprint_file),
                                'subject_id': subject_id,
                                'gender': gender,
                                'hand_side': hand_side.lower(),
                                'finger_type': finger_type.lower(),
                                'filename': filename
                            })
                        else:
                            self.logger.warning(f"Could not parse finger info from {filename} (parts: {finger_parts})")
                    else:
                        self.logger.warning(f"Could not parse filename: {filename} (parts: {parts})")
                        
            except Exception as e:
                self.logger.error(f"Error processing folder {subject_folder}: {e}")
                continue
        
        self.logger.info(f"Found {len(fingerprint_files)} fingerprint files")
        return fingerprint_files
    
    def process_single_file(self, file_info: Dict[str, str]) -> Dict[str, Any]:
        """
        Process a single fingerprint file.
        
        Args:
            file_info: Dictionary containing file information
            
        Returns:
            Processing results
        """
        try:
            result = self.checker.process_fingerprint(
                image_path=file_info['file_path'],
                subject_id=file_info['subject_id'],
                finger_type=file_info['finger_type'],
                hand_side=file_info['hand_side']
            )
            
            # Add file metadata to result
            result.update({
                'filename': file_info['filename'],
                'gender': file_info['gender'],
                'processing_time': datetime.now().isoformat()
            })
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {file_info['file_path']}: {e}")
            return {
                'success': False,
                'error': str(e),
                'filename': file_info['filename'],
                'file_path': file_info['file_path']
            }
    
    def process_all_files(self, base_path: str, interactive: bool = False) -> List[Dict[str, Any]]:
        """
        Process all fingerprint files in the dataset.
        
        Args:
            base_path: Base path to fingerprint data directory
            interactive: Whether to ask for confirmation before processing
            
        Returns:
            List of processing results
        """
        fingerprint_files = self.get_fingerprint_files(base_path)
        
        if not fingerprint_files:
            self.logger.error("No fingerprint files found")
            return []
        
        if interactive:
            print(f"\nFound {len(fingerprint_files)} fingerprint files to process.")
            response = input("Do you want to proceed with processing? (y/n): ").lower().strip()
            if response != 'y':
                print("Processing cancelled.")
                return []
        
        results = []
        total_files = len(fingerprint_files)
        
        print(f"\nStarting batch processing of {total_files} files...")
        
        # First pass: Process all files and collect minutiae
        processed_files = []
        for i, file_info in enumerate(fingerprint_files, 1):
            print(f"\nProcessing file {i}/{total_files}: {file_info['filename']}")
            
            result = self.process_single_file(file_info)
            results.append(result)
            
            # Store successful results for duplicate detection
            if result['success']:
                processed_files.append({
                    'file_info': file_info,
                    'result': result,
                    'minutiae': result.get('minutiae', [])
                })
            
            # Print progress
            if result['success']:
                print(f"  ✓ Success - Quality: {result.get('quality_score', 'N/A'):.3f}, "
                      f"Minutiae: {result.get('minutiae_count', 0)}")
            else:
                print(f"  ✗ Failed - Error: {result.get('error', 'Unknown error')}")
            
            # Progress indicator
            if i % 10 == 0:
                print(f"  Progress: {i}/{total_files} ({i/total_files*100:.1f}%)")
        
        # Second pass: Comprehensive duplicate detection
        print(f"\n{'='*60}")
        print("COMPREHENSIVE DUPLICATE DETECTION")
        print(f"{'='*60}")
        
        duplicates_found = self.detect_all_duplicates(processed_files)
        
        # Update results with duplicate information
        for duplicate in duplicates_found:
            for result in results:
                if result.get('filename') == duplicate['file1']['filename']:
                    result['is_duplicate'] = True
                    result['duplicate_of'] = duplicate['file2']['filename']
                    result['similarity_score'] = duplicate['similarity']
                    break
        
        return results
    
    def detect_all_duplicates(self, processed_files: List[Dict]) -> List[Dict]:
        """
        Detect duplicates by comparing all fingerprints against each other.
        
        Args:
            processed_files: List of processed file information with minutiae
            
        Returns:
            List of duplicate pairs found
        """
        duplicates = []
        similarities = []  # Track all similarities for analysis
        total_comparisons = len(processed_files) * (len(processed_files) - 1) // 2
        comparison_count = 0
        
        print(f"Comparing {len(processed_files)} fingerprints ({total_comparisons} comparisons)...")
        
        # Remove artificial limit - let it complete all comparisons
        for i, file1 in enumerate(processed_files):
            for j, file2 in enumerate(processed_files[i+1:], i+1):
                comparison_count += 1
                
                if comparison_count % 1000 == 0:  # Show progress every 1000 comparisons
                    print(f"  Progress: {comparison_count}/{total_comparisons} comparisons ({comparison_count/total_comparisons*100:.1f}%)")
                
                # Skip if same subject and finger type (should be similar)
                if (file1['file_info']['subject_id'] == file2['file_info']['subject_id'] and
                    file1['file_info']['finger_type'] == file2['file_info']['finger_type'] and
                    file1['file_info']['hand_side'] == file2['file_info']['hand_side']):
                    continue
                
                # Compare minutiae
                similarity = self.compare_minutiae(
                    file1['minutiae'], 
                    file2['minutiae']
                )
                
                # Track all similarities for analysis
                similarities.append(similarity)
                
                # Check if it's a duplicate (lowered threshold for better detection)
                if similarity > 0.6:  # Lowered from 0.8 to 0.6 for more sensitive detection
                    duplicate_info = {
                        'file1': file1['file_info'],
                        'file2': file2['file_info'],
                        'similarity': similarity,
                        'file1_subject': file1['file_info']['subject_id'],
                        'file2_subject': file2['file_info']['subject_id'],
                        'file1_finger': f"{file1['file_info']['hand_side']}_{file1['file_info']['finger_type']}",
                        'file2_finger': f"{file2['file_info']['hand_side']}_{file2['file_info']['finger_type']}"
                    }
                    duplicates.append(duplicate_info)
        
        # Print similarity statistics
        if similarities:
            similarities.sort(reverse=True)
            print(f"\n📊 SIMILARITY STATISTICS:")
            print(f"  - Total comparisons: {len(similarities)}")
            print(f"  - Highest similarity: {similarities[0]:.3f}")
            print(f"  - Top 10 similarities: {[f'{s:.3f}' for s in similarities[:10]]}")
            print(f"  - Average similarity: {sum(similarities)/len(similarities):.3f}")
            print(f"  - Similarities > 0.5: {len([s for s in similarities if s > 0.5])}")
            print(f"  - Similarities > 0.6: {len([s for s in similarities if s > 0.6])}")
            print(f"  - Similarities > 0.7: {len([s for s in similarities if s > 0.7])}")
            print(f"  - Similarities > 0.8: {len([s for s in similarities if s > 0.8])}")
            
            # Show potential duplicates (high similarities)
            high_similarities = [s for s in similarities if s > 0.5]
            if high_similarities:
                print(f"\n🔍 POTENTIAL DUPLICATES (similarity > 0.5):")
                print(f"  - Found {len(high_similarities)} high-similarity pairs")
                print(f"  - Highest: {high_similarities[0]:.3f}")
                if len(high_similarities) > 1:
                    print(f"  - Second highest: {high_similarities[1]:.3f}")
        
        print(f"\nDuplicate detection completed. Found {len(duplicates)} duplicate pairs.")
        
        # Show clean duplicate summary
        if duplicates:
            print(f"\n📋 DUPLICATE SUMMARY:")
            print("-" * 60)
            for i, dup in enumerate(duplicates, 1):
                print(f"{i:2d}. {dup['file1']['filename']} (Subject {dup['file1_subject']})")
                print(f"    ↳ Duplicate of: {dup['file2']['filename']} (Subject {dup['file2_subject']})")
                print(f"    ↳ Similarity: {dup['similarity']:.3f}")
                print()
        else:
            print(f"\n✅ No duplicates found!")
        
        return duplicates
    
    def compare_minutiae(self, minutiae1: List[Dict], minutiae2: List[Dict]) -> float:
        """
        Compare two sets of minutiae points and return similarity score.
        Optimized for faster duplicate detection.
        
        Args:
            minutiae1: First set of minutiae points
            minutiae2: Second set of minutiae points
            
        Returns:
            Similarity score between 0 and 1
        """
        try:
            if not minutiae1 or not minutiae2:
                return 0.0
            
            # Quick check for excessive minutiae points
            if len(minutiae1) > 50 or len(minutiae2) > 50:
                return 0.0
            
            # Extract coordinates and angles
            points1 = [(m.get('x', 0), m.get('y', 0), m.get('theta', 0)) for m in minutiae1]
            points2 = [(m.get('x', 0), m.get('y', 0), m.get('theta', 0)) for m in minutiae2]
            
            # Quick count similarity
            count_similarity = min(len(points1), len(points2)) / max(len(points1), len(points2))
            
            # Early exit if count similarity is too low
            if count_similarity < 0.3:
                return 0.0
            
            # Optimized spatial similarity calculation
            spatial_similarity = 0.0
            if len(points1) > 0 and len(points2) > 0:
                # Use fewer points for faster comparison
                max_points = min(8, len(points1), len(points2))
                total_similarity = 0
                comparisons = 0
                
                for p1 in points1[:max_points]:
                    best_similarity = 0.0
                    for p2 in points2[:max_points]:
                        # Quick distance calculation
                        dist = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5
                        
                        # Quick angle difference
                        angle_diff = abs(p1[2] - p2[2]) % 360
                        if angle_diff > 180:
                            angle_diff = 360 - angle_diff
                        
                        angle_similarity = 1 - (angle_diff / 180)
                        
                        # Combined similarity
                        combined_similarity = (1 - dist/200) * 0.7 + angle_similarity * 0.3
                        combined_similarity = max(0, combined_similarity)
                        
                        best_similarity = max(best_similarity, combined_similarity)
                    
                    total_similarity += best_similarity
                    comparisons += 1
                
                if comparisons > 0:
                    spatial_similarity = total_similarity / comparisons
            
            # Quick pattern similarity
            pattern_similarity = 0.0
            if len(points1) >= 3 and len(points2) >= 3:
                # Use first 3 points for center calculation
                center1 = (sum(p[0] for p in points1[:3])/3, sum(p[1] for p in points1[:3])/3)
                center2 = (sum(p[0] for p in points2[:3])/3, sum(p[1] for p in points2[:3])/3)
                
                center_dist = ((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)**0.5
                pattern_similarity = max(0, 1 - center_dist/300)
            
            # Combined similarity score
            similarity = (count_similarity * 0.3 + spatial_similarity * 0.5 + pattern_similarity * 0.2)
            
            return min(1.0, max(0.0, similarity))
            
        except Exception as e:
            return 0.0
    
    def generate_report(self, results: List[Dict[str, Any]], output_dir: str = "results") -> None:
        """
        Generate comprehensive reports from processing results.
        
        Args:
            results: List of processing results
            output_dir: Directory to save reports
        """
        if not results:
            self.logger.warning("No results to generate report")
            return
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate summary statistics
        successful_results = [r for r in results if r.get('success', False)]
        failed_results = [r for r in results if not r.get('success', False)]
        
        summary = {
            'total_files': len(results),
            'successful': len(successful_results),
            'failed': len(failed_results),
            'success_rate': len(successful_results) / len(results) * 100,
            'duplicates_found': len([r for r in successful_results if r.get('is_duplicate', False)]),
            'average_quality': sum(r.get('quality_score', 0) for r in successful_results) / len(successful_results) if successful_results else 0,
            'processing_time': datetime.now().isoformat()
        }
        
        # Save summary report
        summary_file = output_path / f"batch_summary_{timestamp}.json"
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Save detailed results
        results_file = output_path / f"batch_results_{timestamp}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Generate CSV report
        csv_file = output_path / f"batch_results_{timestamp}.csv"
        if successful_results:
            fieldnames = ['filename', 'subject_id', 'finger_type', 'hand_side', 'gender', 
                         'quality_score', 'is_duplicate', 'max_similarity', 'duplicate_count', 
                         'stored', 'processing_time']
            
            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for result in successful_results:
                    row = {field: result.get(field, '') for field in fieldnames}
                    writer.writerow(row)
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"BATCH PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total files processed: {summary['total_files']}")
        print(f"Successful: {summary['successful']}")
        print(f"Failed: {summary['failed']}")
        print(f"Success rate: {summary['success_rate']:.1f}%")
        print(f"Duplicates found: {summary['duplicates_found']}")
        print(f"Average quality score: {summary['average_quality']:.3f}")
        print(f"\nReports saved to: {output_path}")
        print(f"  - Summary: {summary_file}")
        print(f"  - Detailed results: {results_file}")
        if successful_results:
            print(f"  - CSV report: {csv_file}")
        print(f"{'='*60}")
    
    def process_specific_subjects(self, base_path: str, subject_ids: List[str], interactive: bool = False) -> List[Dict[str, Any]]:
        """
        Process specific subject folders.
        
        Args:
            base_path: Base path to fingerprint data directory
            subject_ids: List of subject IDs to process
            interactive: Whether to ask for confirmation
            
        Returns:
            List of processing results
        """
        all_files = self.get_fingerprint_files(base_path)
        target_files = [f for f in all_files if f['subject_id'] in subject_ids]
        
        if not target_files:
            self.logger.error(f"No files found for subjects: {subject_ids}")
            return []
        
        if interactive:
            print(f"\nFound {len(target_files)} files for subjects {subject_ids}")
            response = input("Do you want to proceed? (y/n): ").lower().strip()
            if response != 'y':
                return []
        
        results = []
        for file_info in target_files:
            result = self.process_single_file(file_info)
            results.append(result)
        
        return results
    
    def test_threshold(self, processed_files: List[Dict], threshold: float) -> List[Dict]:
        """
        Test duplicate detection with a specific threshold.
        
        Args:
            processed_files: List of processed file information with minutiae
            threshold: Similarity threshold to test
            
        Returns:
            List of duplicate pairs found with the given threshold
        """
        duplicates = []
        
        for i, file1 in enumerate(processed_files):
            for j, file2 in enumerate(processed_files[i+1:], i+1):
                # Skip if same subject and finger type (should be similar)
                if (file1['file_info']['subject_id'] == file2['file_info']['subject_id'] and
                    file1['file_info']['finger_type'] == file2['file_info']['finger_type'] and
                    file1['file_info']['hand_side'] == file2['file_info']['hand_side']):
                    continue
                
                # Compare minutiae
                similarity = self.compare_minutiae(
                    file1['minutiae'], 
                    file2['minutiae']
                )
                
                # Check if it's a duplicate with the given threshold
                if similarity > threshold:
                    duplicate_info = {
                        'file1': file1['file_info'],
                        'file2': file2['file_info'],
                        'similarity': similarity,
                        'file1_subject': file1['file_info']['subject_id'],
                        'file2_subject': file2['file_info']['subject_id'],
                        'file1_finger': f"{file1['file_info']['hand_side']}_{file1['file_info']['finger_type']}",
                        'file2_finger': f"{file2['file_info']['hand_side']}_{file2['file_info']['finger_type']}"
                    }
                    duplicates.append(duplicate_info)
        
        return duplicates
    
    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.checker:
            self.checker.cleanup()

def main():
    """Main entry point for batch processing"""
    parser = argparse.ArgumentParser(description='Batch Fingerprint Processing')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--base-path', type=str, 
                       default='fingerprint_data',
                       help='Base path to fingerprint data directory')
    parser.add_argument('--subjects', type=str, nargs='+',
                       help='Specific subject IDs to process (e.g., 1 2 3)')
    parser.add_argument('--interactive', action='store_true',
                       help='Interactive mode with user confirmation')
    parser.add_argument('--output-dir', type=str, default='results',
                       help='Output directory for reports')
    parser.add_argument('--mode', type=str, choices=['all', 'specific', 'scan'],
                       default='all', help='Processing mode')
    
    args = parser.parse_args()
    
    # Initialize batch processor
    processor = BatchFingerprintProcessor(args.config)
    
    if not processor.initialize():
        print("Failed to initialize fingerprint system")
        return
    
    try:
        if args.mode == 'scan':
            # Just scan and show what would be processed
            files = processor.get_fingerprint_files(args.base_path)
            print(f"\nFound {len(files)} fingerprint files:")
            for file_info in files[:10]:  # Show first 10
                print(f"  {file_info['filename']} (Subject {file_info['subject_id']})")
            if len(files) > 10:
                print(f"  ... and {len(files) - 10} more files")
            
        elif args.mode == 'specific':
            if not args.subjects:
                print("Error: --subjects required for specific mode")
                return
            results = processor.process_specific_subjects(args.base_path, args.subjects, args.interactive)
            processor.generate_report(results, args.output_dir)
            
        else:  # all mode
            results = processor.process_all_files(args.base_path, args.interactive)
            processor.generate_report(results, args.output_dir)
    
    finally:
        processor.cleanup()

if __name__ == "__main__":
    main() 