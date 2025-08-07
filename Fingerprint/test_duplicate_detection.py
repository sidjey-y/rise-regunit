#!/usr/bin/env python3

import os
import sys
import logging
from pathlib import Path
import time

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from duplicate_detector import DuplicateFingerprintDetector
from bozorth3_matcher import Bozorth3Matcher
from minutiae_extractor import MinutiaeExtractor
import cv2
import numpy as np

def setup_logging():
    """Setup logging for testing"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('test_duplicate_detection.log')
        ]
    )

def test_bozorth3_matcher():
    """Test the Bozorth3 matcher with sample data"""
    print("\n" + "="*60)
    print("Testing Bozorth3 Matcher")
    print("="*60)
    
    # Create sample minutiae data
    template_minutiae = [
        {'x': 100, 'y': 150, 'theta': 45.0, 'type': 'ending'},
        {'x': 200, 'y': 250, 'theta': 90.0, 'type': 'bifurcation'},
        {'x': 300, 'y': 350, 'theta': 135.0, 'type': 'ending'},
        {'x': 400, 'y': 450, 'theta': 180.0, 'type': 'bifurcation'},
        {'x': 500, 'y': 550, 'theta': 225.0, 'type': 'ending'}
    ]
    
    # Create similar query minutiae (slight variations)
    query_minutiae = [
        {'x': 102, 'y': 148, 'theta': 47.0, 'type': 'ending'},
        {'x': 198, 'y': 252, 'theta': 88.0, 'type': 'bifurcation'},
        {'x': 302, 'y': 348, 'theta': 137.0, 'type': 'ending'},
        {'x': 398, 'y': 452, 'theta': 182.0, 'type': 'bifurcation'},
        {'x': 498, 'y': 552, 'theta': 227.0, 'type': 'ending'}
    ]
    
    # Create different query minutiae (should not match)
    different_minutiae = [
        {'x': 50, 'y': 50, 'theta': 0.0, 'type': 'ending'},
        {'x': 150, 'y': 150, 'theta': 45.0, 'type': 'bifurcation'},
        {'x': 250, 'y': 250, 'theta': 90.0, 'type': 'ending'},
        {'x': 350, 'y': 350, 'theta': 135.0, 'type': 'bifurcation'},
        {'x': 450, 'y': 450, 'theta': 180.0, 'type': 'ending'}
    ]
    
    # Initialize Bozorth3 matcher
    matcher = Bozorth3Matcher()
    
    # Test similar fingerprints (should match)
    print("Testing similar fingerprints...")
    result1 = matcher.match_fingerprints(template_minutiae, query_minutiae)
    print(f"Match Score: {result1['match_score']:.2f}")
    print(f"Is Match: {result1['is_match']}")
    print(f"Correspondence Count: {result1['correspondence_count']}")
    print(f"Pattern Match: {result1['pattern_match']}")
    
    # Test different fingerprints (should not match)
    print("\nTesting different fingerprints...")
    result2 = matcher.match_fingerprints(template_minutiae, different_minutiae)
    print(f"Match Score: {result2['match_score']:.2f}")
    print(f"Is Match: {result2['is_match']}")
    print(f"Correspondence Count: {result2['correspondence_count']}")
    print(f"Pattern Match: {result2['pattern_match']}")
    
    # Test xytheta format conversion
    print("\nTesting xytheta format conversion...")
    xytheta_points = matcher.convert_to_xytheta(template_minutiae)
    print(f"Converted {len(xytheta_points)} points to xytheta format")
    for i, (x, y, theta) in enumerate(xytheta_points[:3]):  # Show first 3
        print(f"  Point {i+1}: x={x}, y={y}, theta={theta:.2f}")
    
    # Test pattern creation
    print("\nTesting pattern creation...")
    pattern = matcher.create_minutiae_pattern(xytheta_points)
    print(f"Created pattern: {pattern[:50]}...")  # Show first 50 chars
    
    return result1, result2

def test_boyer_moore():
    """Test the Boyer-Moore string matching algorithm"""
    print("\n" + "="*60)
    print("Testing Boyer-Moore Algorithm")
    print("="*60)
    
    from bozorth3_matcher import BoyerMooreMatcher
    
    bm = BoyerMooreMatcher()
    
    # Test cases
    test_cases = [
        ("ABCDEFGHIJKLMNOP", "DEF", [3]),  # Pattern in middle
        ("ABCDEFGHIJKLMNOP", "XYZ", []),   # Pattern not found
        ("AAAAAA", "AAA", [0, 1, 2, 3]),   # Multiple matches
        ("", "ABC", []),                   # Empty text
        ("ABC", "", []),                   # Empty pattern
    ]
    
    for text, pattern, expected in test_cases:
        result = bm.search(text, pattern)
        print(f"Text: '{text}', Pattern: '{pattern}'")
        print(f"  Expected: {expected}, Got: {result}")
        print(f"  {'✓ PASS' if result == expected else '✗ FAIL'}")

def test_minutiae_extraction():
    """Test minutiae extraction from a sample fingerprint"""
    print("\n" + "="*60)
    print("Testing Minutiae Extraction")
    print("="*60)
    
    # Create a synthetic fingerprint image for testing
    def create_synthetic_fingerprint():
        """Create a synthetic fingerprint image for testing"""
        img = np.zeros((400, 400), dtype=np.uint8)
        
        # Add some ridge patterns
        for i in range(0, 400, 20):
            cv2.line(img, (i, 0), (i, 400), 255, 2)
        
        # Add some minutiae-like features
        cv2.circle(img, (100, 100), 3, 255, -1)  # Ridge ending
        cv2.circle(img, (200, 200), 3, 255, -1)  # Ridge ending
        cv2.circle(img, (300, 300), 3, 255, -1)  # Ridge ending
        
        # Add some noise
        noise = np.random.randint(0, 50, (400, 400), dtype=np.uint8)
        img = cv2.add(img, noise)
        
        return img
    
    try:
        # Create synthetic image
        synthetic_img = create_synthetic_fingerprint()
        
        # Save for inspection
        cv2.imwrite('test_synthetic_fingerprint.bmp', synthetic_img)
        print("Created synthetic fingerprint image: test_synthetic_fingerprint.bmp")
        
        # Extract minutiae
        extractor = MinutiaeExtractor()
        minutiae = extractor.extract_minutiae(synthetic_img)
        
        print(f"Extracted {len(minutiae)} minutiae points")
        
        if minutiae:
            print("Sample minutiae points:")
            for i, point in enumerate(minutiae[:5]):  # Show first 5
                print(f"  Point {i+1}: x={point['x']}, y={point['y']}, theta={point['theta']:.2f}")
        
        return minutiae
        
    except Exception as e:
        print(f"Error in minutiae extraction test: {e}")
        return []

def test_duplicate_detection_on_sample_data():
    """Test duplicate detection on a small sample of real data"""
    print("\n" + "="*60)
    print("Testing Duplicate Detection on Sample Data")
    print("="*60)
    
    # Check if fingerprint data exists
    data_path = "fingerprint_data"
    if not Path(data_path).exists():
        print(f"Fingerprint data directory '{data_path}' not found.")
        print("Skipping real data test.")
        return
    
    try:
        # Initialize detector
        detector = DuplicateFingerprintDetector()
        
        if not detector.initialize():
            print("Failed to initialize duplicate detector")
            return
        
        # Scan for fingerprint files
        fingerprint_files = detector.scan_fingerprint_folder(data_path)
        
        if not fingerprint_files:
            print("No fingerprint files found in the data directory")
            return
        
        print(f"Found {len(fingerprint_files)} fingerprint files")
        
        # Limit to first 10 files for quick testing
        test_files = fingerprint_files[:10]
        print(f"Testing with first {len(test_files)} files")
        
        # Test duplicate detection
        start_time = time.time()
        results = detector.detect_duplicates(test_files, "test_duplicate_results")
        end_time = time.time()
        
        print(f"\nDuplicate Detection Results:")
        print(f"Processing Time: {end_time - start_time:.2f} seconds")
        print(f"Total Files: {results['total_files']}")
        print(f"Files with Minutiae: {results['files_with_minutiae']}")
        print(f"Total Comparisons: {results['total_comparisons']}")
        print(f"Duplicates Found: {results['duplicates_found']}")
        
        if results['duplicates_found'] > 0:
            print("\nDuplicate pairs found:")
            for i, duplicate in enumerate(results['duplicate_pairs'], 1):
                print(f"  {i}. {duplicate['file1']['filename']} <-> {duplicate['file2']['filename']}")
                print(f"     Score: {duplicate['match_score']:.2f}")
        
        detector.cleanup()
        
    except Exception as e:
        print(f"Error in duplicate detection test: {e}")

def main():
    """Main test function"""
    print("Fingerprint Duplicate Detection Test Suite")
    print("Using Bozorth3, xytheta format, and Boyer-Moore algorithm")
    
    # Setup logging
    setup_logging()
    
    try:
        # Test Bozorth3 matcher
        test_bozorth3_matcher()
        
        # Test Boyer-Moore algorithm
        test_boyer_moore()
        
        # Test minutiae extraction
        test_minutiae_extraction()
        
        # Test duplicate detection on real data
        test_duplicate_detection_on_sample_data()
        
        print("\n" + "="*60)
        print("All tests completed!")
        print("="*60)
        
    except Exception as e:
        print(f"Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 