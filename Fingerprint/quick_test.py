#!/usr/bin/env python3

import os
from pathlib import Path

def test_fingerprint_structure():
    """Test the fingerprint data structure"""
    print("Testing fingerprint data structure...")
    
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

def test_batch_processing():
    """Test the batch processing script"""
    print("\n" + "="*50)
    print("Testing batch processing...")
    
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
    print("="*60)
    print("FINGERPRINT DATA STRUCTURE TEST")
    print("="*60)
    
    # Test structure
    structure_ok = test_fingerprint_structure()
    
    if structure_ok:
        # Test batch processing
        batch_ok = test_batch_processing()
        
        if batch_ok:
            print("\n" + "="*60)
            print("✅ ALL TESTS PASSED!")
            print("✅ Your fingerprint data is ready for processing")
            print("✅ Run: python run_batch.py")
            print("="*60)
        else:
            print("\n❌ Batch processing test failed")
    else:
        print("\n❌ Structure test failed") 