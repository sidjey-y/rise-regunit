#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from batch_process import BatchFingerprintProcessor

def test_folder48_specific_duplicates():
    """Test duplicate detection specifically for folder 48 left index vs left little finger"""
    print("=" * 60)
    print("FOLDER 48 DUPLICATE DETECTION TEST")
    print("Testing left index vs left little finger (should be same fingerprint)")
    print("=" * 60)
    
    # Initialize processor
    processor = BatchFingerprintProcessor()
    
    if not processor.initialize():
        print("❌ Failed to initialize fingerprint system")
        return
    
    print("✅ System initialized successfully!")
    
    # Get files from folder 48
    all_files = processor.get_fingerprint_files("fingerprint_data")
    folder48_files = [f for f in all_files if f['subject_id'] == '48']
    
    print(f"\n📁 Found {len(folder48_files)} files in folder 48:")
    for file_info in folder48_files:
        print(f"  - {file_info['filename']} ({file_info['hand_side']} {file_info['finger_type']})")
    
    # Find the specific files we want to compare
    left_index = None
    left_little = None
    
    for file_info in folder48_files:
        if file_info['hand_side'] == 'left' and file_info['finger_type'] == 'index':
            left_index = file_info
        elif file_info['hand_side'] == 'left' and file_info['finger_type'] == 'little':
            left_little = file_info
    
    if not left_index or not left_little:
        print("❌ Could not find left index and/or left little finger files")
        return
    
    print(f"\n🎯 Testing duplicate detection between:")
    print(f"  File 1: {left_index['filename']}")
    print(f"  File 2: {left_little['filename']}")
    
    # Process both files
    print(f"\n🔍 Processing files...")
    
    result1 = processor.process_single_file(left_index)
    result2 = processor.process_single_file(left_little)
    
    if not result1['success'] or not result2['success']:
        print("❌ Failed to process one or both files")
        if not result1['success']:
            print(f"  Error in {left_index['filename']}: {result1.get('error')}")
        if not result2['success']:
            print(f"  Error in {left_little['filename']}: {result2.get('error')}")
        return
    
    print(f"✅ Both files processed successfully")
    print(f"  {left_index['filename']}: {result1.get('minutiae_count', 0)} minutiae")
    print(f"  {left_little['filename']}: {result2.get('minutiae_count', 0)} minutiae")
    
    # Compare minutiae directly
    print(f"\n🔍 Comparing minutiae...")
    
    minutiae1 = result1.get('minutiae', [])
    minutiae2 = result2.get('minutiae', [])
    
    similarity = processor.compare_minutiae(minutiae1, minutiae2)
    
    print(f"\n📊 COMPARISON RESULTS:")
    print(f"  Similarity Score: {similarity:.4f}")
    print(f"  Threshold: 0.8")
    print(f"  Is Duplicate: {'YES' if similarity > 0.8 else 'NO'}")
    
    if similarity > 0.8:
        print(f"  ✅ DUPLICATE DETECTED!")
    else:
        print(f"  ❌ NO DUPLICATE DETECTED (similarity too low)")
    
    # Show detailed minutiae comparison
    print(f"\n🔍 DETAILED MINUTIAE ANALYSIS:")
    print(f"  File 1 minutiae count: {len(minutiae1)}")
    print(f"  File 2 minutiae count: {len(minutiae2)}")
    
    if minutiae1 and minutiae2:
        print(f"\n  First 3 minutiae points from each file:")
        print(f"  File 1 ({left_index['filename']}):")
        for i, m in enumerate(minutiae1[:3]):
            print(f"    Point {i+1}: x={m['x']:.1f}, y={m['y']:.1f}, theta={m['theta']:.1f}")
        
        print(f"  File 2 ({left_little['filename']}):")
        for i, m in enumerate(minutiae2[:3]):
            print(f"    Point {i+1}: x={m['x']:.1f}, y={m['y']:.1f}, theta={m['theta']:.1f}")
    
    # Test with different thresholds
    print(f"\n🔍 TESTING DIFFERENT THRESHOLDS:")
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    for threshold in thresholds:
        is_dup = similarity > threshold
        print(f"  Threshold {threshold:.1f}: {'DUPLICATE' if is_dup else 'NOT DUPLICATE'}")
    
    # Cleanup
    processor.cleanup()

if __name__ == "__main__":
    test_folder48_specific_duplicates()
