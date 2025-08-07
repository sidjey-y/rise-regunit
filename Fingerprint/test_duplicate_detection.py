#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from batch_process import BatchFingerprintProcessor

def test_duplicate_detection():
    """Test duplicate detection specifically on folders 46-48"""
    print("=" * 60)
    print("DUPLICATE DETECTION TEST")
    print("Testing folders 46-48 for intentional duplicates")
    print("=" * 60)
    
    # Initialize processor
    processor = BatchFingerprintProcessor()
    
    if not processor.initialize():
        print("❌ Failed to initialize fingerprint system")
        return
    
    print("✅ System initialized successfully!")
    
    # Get all fingerprint files
    all_files = processor.get_fingerprint_files("fingerprint_data")
    
    if not all_files:
        print("❌ No fingerprint files found!")
        return
    
    # Filter for folders 46-48
    test_files = [f for f in all_files if f['subject_id'] in ['46', '47', '48']]
    
    print(f"\n📁 Found {len(test_files)} files in folders 46-48:")
    for file_info in test_files:
        print(f"  - {file_info['filename']} (Subject {file_info['subject_id']})")
    
    # Process test files
    print(f"\n🔍 Processing {len(test_files)} test files...")
    processed_files = []
    
    for i, file_info in enumerate(test_files, 1):
        print(f"\nProcessing {i}/{len(test_files)}: {file_info['filename']}")
        
        result = processor.process_single_file(file_info)
        
        if result['success']:
            processed_files.append({
                'file_info': file_info,
                'result': result,
                'minutiae': result.get('minutiae', [])
            })
            print(f"  ✓ Success - Minutiae: {result.get('minutiae_count', 0)}")
        else:
            print(f"  ✗ Failed - Error: {result.get('error', 'Unknown error')}")
    
    # Run duplicate detection
    print(f"\n{'='*60}")
    print("DUPLICATE DETECTION ON TEST FOLDERS")
    print(f"{'='*60}")
    
    duplicates = processor.detect_all_duplicates(processed_files)
    
    # Summary
    print(f"\n{'='*60}")
    print("TEST RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Files processed: {len(processed_files)}")
    print(f"Duplicates found: {len(duplicates)}")
    
    if duplicates:
        print(f"\n🚨 DUPLICATES DETECTED:")
        for i, dup in enumerate(duplicates, 1):
            print(f"\n{i}. Similarity: {dup['similarity']:.3f}")
            print(f"   File 1: {dup['file1']['filename']} (Subject {dup['file1_subject']})")
            print(f"   File 2: {dup['file2']['filename']} (Subject {dup['file2_subject']})")
            print(f"   Finger 1: {dup['file1_finger']}")
            print(f"   Finger 2: {dup['file2_finger']}")
    else:
        print("\n✅ No duplicates detected in test folders")
    
    # Cleanup
    processor.cleanup()

if __name__ == "__main__":
    test_duplicate_detection() 