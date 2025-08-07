#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from batch_process import BatchFingerprintProcessor

def debug_minutiae_structure():
    """Debug script to check minutiae data structure"""
    print("=" * 60)
    print("MINUTIAE DATA STRUCTURE DEBUG")
    print("=" * 60)
    
    # Initialize processor
    processor = BatchFingerprintProcessor()
    
    if not processor.initialize():
        print("❌ Failed to initialize fingerprint system")
        return
    
    print("✅ System initialized successfully!")
    
    # Get a single file from folder 48
    all_files = processor.get_fingerprint_files("fingerprint_data")
    folder48_files = [f for f in all_files if f['subject_id'] == '48']
    
    if not folder48_files:
        print("❌ No files found in folder 48")
        return
    
    # Process first file
    test_file = folder48_files[0]
    print(f"\n🔍 Processing file: {test_file['filename']}")
    
    result = processor.process_single_file(test_file)
    
    if not result['success']:
        print(f"❌ Failed to process file: {result.get('error')}")
        return
    
    minutiae = result.get('minutiae', [])
    print(f"✅ Successfully extracted {len(minutiae)} minutiae points")
    
    if minutiae:
        print(f"\n📊 MINUTIAE DATA STRUCTURE:")
        print(f"Type: {type(minutiae)}")
        print(f"Length: {len(minutiae)}")
        
        # Show first minutiae point structure
        first_point = minutiae[0]
        print(f"\n🔍 FIRST MINUTIAE POINT:")
        print(f"Type: {type(first_point)}")
        print(f"Keys: {list(first_point.keys())}")
        
        for key, value in first_point.items():
            print(f"  {key}: {value} (type: {type(value)})")
        
        # Show first 3 points
        print(f"\n📋 FIRST 3 MINUTIAE POINTS:")
        for i, point in enumerate(minutiae[:3]):
            print(f"  Point {i+1}: {point}")
        
        # Test field access
        print(f"\n🧪 TESTING FIELD ACCESS:")
        try:
            x_values = [m['x'] for m in minutiae[:3]]
            y_values = [m['y'] for m in minutiae[:3]]
            theta_values = [m['theta'] for m in minutiae[:3]]
            print(f"  x values: {x_values}")
            print(f"  y values: {y_values}")
            print(f"  theta values: {theta_values}")
            print(f"  ✅ Field access successful!")
        except KeyError as e:
            print(f"  ❌ KeyError: {e}")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    # Cleanup
    processor.cleanup()

if __name__ == "__main__":
    debug_minutiae_structure()
