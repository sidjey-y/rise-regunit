#!/usr/bin/env python3

import os
import sys
from pathlib import Path
from batch_process import BatchFingerprintProcessor

def main():
    """Interactive batch processing script"""
    print("=" * 60)
    print("FINGERPRINT BATCH PROCESSING SYSTEM")
    print("=" * 60)
    
    # Initialize processor
    processor = BatchFingerprintProcessor()
    
    if not processor.initialize():
        print("❌ Failed to initialize fingerprint system")
        print("Please check your configuration and dependencies.")
        return
    
    print("✅ System initialized successfully!")
    
    while True:
        print("\n" + "=" * 40)
        print("BATCH PROCESSING OPTIONS")
        print("=" * 40)
        print("1. Scan and show all fingerprint files")
        print("2. Process ALL 48 folders (all fingerprints)")
        print("3. Process specific subject folders")
        print("4. Process with interactive confirmation")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == '1':
            # Scan mode
            print("\n🔍 Scanning fingerprint data directory...")
            files = processor.get_fingerprint_files("fingerprint_data")
            
            if files:
                print(f"\n📁 Found {len(files)} fingerprint files:")
                print("-" * 50)
                
                # Group by subject
                subjects = {}
                for file_info in files:
                    subject_id = file_info['subject_id']
                    if subject_id not in subjects:
                        subjects[subject_id] = []
                    subjects[subject_id].append(file_info)
                
                for subject_id in sorted(subjects.keys(), key=int):
                    file_count = len(subjects[subject_id])
                    print(f"Subject {subject_id}: {file_count} files")
                
                print(f"\n📊 Summary:")
                print(f"  - Total subjects: {len(subjects)}")
                print(f"  - Total files: {len(files)}")
                print(f"  - Average files per subject: {len(files)/len(subjects):.1f}")
            else:
                print("❌ No fingerprint files found!")
                
        elif choice == '2':
            # Process all files
            print("\n⚠️  WARNING: This will process ALL fingerprint files!")
            print("This may take a long time depending on the number of files.")
            
            confirm = input("Are you sure you want to proceed? (yes/no): ").lower().strip()
            if confirm == 'yes':
                print("\n🚀 Starting batch processing of all files...")
                results = processor.process_all_files("fingerprint_data", interactive=False)
                processor.generate_report(results, "results")
            else:
                print("❌ Processing cancelled.")
                
        elif choice == '3':
            # Process specific subjects
            print("\n📝 Enter subject IDs to process (separated by spaces)")
            print("Example: 1 2 3 4 5")
            
            subjects_input = input("Subject IDs: ").strip()
            if subjects_input:
                subject_ids = subjects_input.split()
                print(f"\n🎯 Processing subjects: {subject_ids}")
                
                results = processor.process_specific_subjects(
                    "fingerprint_data",
                    subject_ids,
                    interactive=False
                )
                processor.generate_report(results, "results")
            else:
                print("❌ No subjects specified.")
                
        elif choice == '4':
            # Interactive mode
            print("\n🔍 Scanning for fingerprint files...")
            files = processor.get_fingerprint_files("fingerprint_data")
            
            if files:
                print(f"\n📁 Found {len(files)} fingerprint files to process.")
                
                # Ask for confirmation
                confirm = input("Do you want to proceed with processing? (y/n): ").lower().strip()
                if confirm == 'y':
                    print("\n🚀 Starting interactive batch processing...")
                    results = processor.process_all_files("fingerprint_data", interactive=True)
                    processor.generate_report(results, "results")
                else:
                    print("❌ Processing cancelled.")
            else:
                print("❌ No fingerprint files found!")
                
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
            
        else:
            print("❌ Invalid choice. Please enter a number between 1-5.")
    
    # Cleanup
    processor.cleanup()

if __name__ == "__main__":
    main() 