#!/usr/bin/env python3

import os
from pathlib import Path

def find_fingerprint_data():
    """Find the fingerprint data directory"""
    print("Searching for fingerprint data...")
    
    # Check current directory
    current_dir = Path.cwd()
    print(f"Current directory: {current_dir}")
    
    # List all directories in current location
    print("\nDirectories in current location:")
    for item in current_dir.iterdir():
        if item.is_dir():
            print(f"  - {item.name}")
    
    # Check if fingerprint_data exists in current directory
    fingerprint_data = current_dir / "fingerprint_data"
    if fingerprint_data.exists():
        print(f"\n✅ Found fingerprint_data at: {fingerprint_data}")
        return str(fingerprint_data)
    
    # Check if we're in the regunit directory
    if current_dir.name == "regunit":
        fingerprint_data = current_dir / "fingerprint_data"
        if fingerprint_data.exists():
            print(f"\n✅ Found fingerprint_data at: {fingerprint_data}")
            return str(fingerprint_data)
    
    # Check if we're in the Fingerprint subdirectory
    if current_dir.name == "Fingerprint":
        # Go up to parent and look for fingerprint_data
        fingerprint_data = current_dir.parent / "fingerprint_data"
        if fingerprint_data.exists():
            print(f"\n✅ Found fingerprint_data at: {fingerprint_data}")
            return str(fingerprint_data)
    
    # Check if any directory contains fingerprint_data
    for item in current_dir.iterdir():
        if item.is_dir():
            fingerprint_data = item / "fingerprint_data"
            if fingerprint_data.exists():
                print(f"\n✅ Found fingerprint_data at: {fingerprint_data}")
                return str(fingerprint_data)
    
    print("\n❌ Could not find fingerprint_data directory")
    return None

def scan_fingerprint_structure(base_path):
    """Scan the fingerprint data structure"""
    if not base_path:
        return
    
    base_path = Path(base_path)
    print(f"\nScanning: {base_path}")
    
    if not base_path.exists():
        print("❌ Path does not exist!")
        return
    
    # Count folders
    folders = [f for f in base_path.iterdir() if f.is_dir()]
    print(f"📁 Found {len(folders)} folders")
    
    if len(folders) == 0:
        print("❌ No folders found!")
        return
    
    # Check first few folders
    for i, folder in enumerate(sorted(folders)[:5]):
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

if __name__ == "__main__":
    fingerprint_path = find_fingerprint_data()
    scan_fingerprint_structure(fingerprint_path) 