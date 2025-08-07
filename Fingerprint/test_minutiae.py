#!/usr/bin/env python3

import os
import sys
from pathlib import Path
import cv2
import numpy as np
from minutiae_extractor import MinutiaeExtractor

def test_fingerprint_structure():
    """Test the fingerprint data structure"""
    print("=" * 60)
    print("FINGERPRINT DATA STRUCTURE TEST")
    print("=" * 60)
    
    # Check if fingerprint_data exists
    fingerprint_data = Path("fingerprint_data")
    if not fingerprint_data.exists():
        print("❌ fingerprint_data folder not found in current directory")
        print("Current directory:", Path.cwd())
        print("Available directories:")
        for item in Path.cwd().iterdir():
            if item.is_dir():
                print(f"  - {item.name}")
        return False
    
    print(f"✅ Found fingerprint_data at: {fingerprint_data}")
    
    # Count folders
    folders = [f for f in fingerprint_data.iterdir() if f.is_dir()]
    print(f"📁 Found {len(folders)} folders")
    
    if len(folders) == 0:
        print("❌ No folders found in fingerprint_data")
        return False
    
    # Check first few folders
    for i, folder in enumerate(sorted(folders)[:3]):
        print(f"\n📂 Folder {folder.name}:")
        
        # Check for Fingerprint subfolder
        fingerprint_folder = folder / "Fingerprint"
        if fingerprint_folder.exists():
            files = list(fingerprint_folder.glob("*.BMP"))
            print(f"  ✅ Fingerprint folder found with {len(files)} .BMP files")
            
            # Show first few filenames
            for file in files[:3]:
                print(f"    - {file.name}")
        else:
            print(f"  ❌ No Fingerprint folder found")
            # List what's in the folder
            items = list(folder.iterdir())
            print(f"  📋 Contains: {[item.name for item in items[:5]]}")
    
    return True

def test_minutiae_extraction():
    """Test minutiae extraction on a sample fingerprint"""
    print("\n" + "=" * 60)
    print("MINUTIAE EXTRACTION TEST")
    print("=" * 60)
    
    # Find a sample fingerprint file
    fingerprint_data = Path("fingerprint_data")
    if not fingerprint_data.exists():
        print("❌ fingerprint_data not found")
        return False
    
    # Look for the first available fingerprint
    sample_file = None
    for folder in fingerprint_data.iterdir():
        if folder.is_dir():
            fingerprint_folder = folder / "Fingerprint"
            if fingerprint_folder.exists():
                files = list(fingerprint_folder.glob("*.BMP"))
                if files:
                    sample_file = files[0]
                    break
    
    if not sample_file:
        print("❌ No fingerprint files found")
        return False
    
    print(f"✅ Testing with sample file: {sample_file.name}")
    
    try:
        # Load the image
        image = cv2.imread(str(sample_file))
        if image is None:
            print("❌ Could not load image")
            return False
        
        print(f"✅ Loaded image: {image.shape}")
        
        # Initialize minutiae extractor
        extractor = MinutiaeExtractor()
        
        # Extract minutiae
        minutiae = extractor.extract_minutiae(image)
        
        if minutiae:
            print(f"✅ Extracted {len(minutiae)} minutiae points")
            
            # Show first few minutiae
            print("\n📊 Sample minutiae points:")
            for i, point in enumerate(minutiae[:5]):
                print(f"  {i+1}. x={point['x']}, y={point['y']}, theta={point['theta']:.3f}, type={point['type']}")
            
            # Save to file
            output_file = f"minutiae_test_{sample_file.stem}.txt"
            if extractor.save_minutiae_to_file(minutiae, output_file):
                print(f"✅ Saved minutiae to: {output_file}")
            
            return True
        else:
            print("❌ No minutiae extracted")
            return False
            
    except Exception as e:
        print(f"❌ Error testing minutiae extraction: {e}")
        return False

def test_batch_processing():
    """Test the batch processing system"""
    print("\n" + "=" * 60)
    print("BATCH PROCESSING TEST")
    print("=" * 60)
    
    try:
        from batch_process import BatchFingerprintProcessor
        
        # Initialize processor
        processor = BatchFingerprintProcessor()
        
        if not processor.initialize():
            print("❌ Failed to initialize fingerprint system")
            return False
        
        print("✅ System initialized successfully")
        
        # Test file scanning
        files = processor.get_fingerprint_files("fingerprint_data")
        
        if files:
            print(f"✅ Found {len(files)} fingerprint files")
            print("Sample files:")
            for file_info in files[:3]:
                print(f"  - {file_info['filename']} (Subject {file_info['subject_id']})")
        else:
            print("❌ No fingerprint files found")
            return False
        
        processor.cleanup()
        return True
        
    except Exception as e:
        print(f"❌ Error testing batch processing: {e}")
        return False

if __name__ == "__main__":
    print("🧪 FINGERPRINT MINUTIAE SYSTEM TEST")
    print("=" * 60)
    
    # Test 1: Structure
    structure_ok = test_fingerprint_structure()
    
    if structure_ok:
        # Test 2: Minutiae extraction
        minutiae_ok = test_minutiae_extraction()
        
        if minutiae_ok:
            # Test 3: Batch processing
            batch_ok = test_batch_processing()
            
            if batch_ok:
                print("\n" + "=" * 60)
                print("✅ ALL TESTS PASSED!")
                print("✅ Your fingerprint system is ready for processing")
                print("✅ Run: python run_batch.py")
                print("=" * 60)
            else:
                print("\n❌ Batch processing test failed")
        else:
            print("\n❌ Minutiae extraction test failed")
    else:
        print("\n❌ Structure test failed") 