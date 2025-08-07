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
        print("Failed to initialize fingerprint system")
        print("Please check your configuration and dependencies.")
        return
    
    print("✅ System initialized successfully!")
    
    try:
        while True:
            print("\n" + "=" * 40)
            print("BATCH PROCESSING OPTIONS")
            print("=" * 40)
            print("1. Scan and show all fingerprint files")
            print("2. Process ALL 48 folders (all fingerprints)")
            print("3. Process specific subject folders")
            print("4. Process with interactive confirmation")
            print("5. Test duplicate detection with different thresholds")
            print("6. Exit")
            
            try:
                choice = input("\nEnter your choice (1-6): ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\n\n👋 Goodbye!")
                break
            
            if choice == '1':
                # Scan mode
                print("\nScanning fingerprint data directory...")
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
                    
                    print(f"\nSummary:")
                    print(f"  - Total subjects: {len(subjects)}")
                    print(f"  - Total files: {len(files)}")
                    print(f"  - Average files per subject: {len(files)/len(subjects):.1f}")
                else:
                    print("No fingerprint files found!")
                    
            elif choice == '2':
                # Process all files
                print("\nWARNING: This will process ALL fingerprint files!")
                print("This may take a long time depending on the number of files.")
                
                confirm = input("Are you sure you want to proceed? (yes/no): ").lower().strip()
                if confirm == 'yes':
                    print("\nStarting batch processing of all files...")
                    results = processor.process_all_files("fingerprint_data", interactive=False)
                    processor.generate_report(results, "results")
                else:
                    print("Processing cancelled.")
                    
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
                        print("Processing cancelled.")
                else:
                    print("No fingerprint files found!")
                    
            elif choice == '5':
                # Test duplicate detection with different thresholds
                print("\n🔍 Testing duplicate detection with different thresholds...")
                files = processor.get_fingerprint_files("fingerprint_data")
                
                if files:
                    print(f"\n📁 Found {len(files)} fingerprint files to test.")
                    
                    # Process a subset for testing (first 10 subjects)
                    test_files = [f for f in files if int(f['subject_id']) <= 10]
                    print(f"Testing with first 10 subjects ({len(test_files)} files)...")
                    
                    # Process files to get minutiae
                    processed_files = []
                    for file_info in test_files:
                        result = processor.process_single_file(file_info)
                        if result['success']:
                            processed_files.append({
                                'file_info': file_info,
                                'result': result,
                                'minutiae': result.get('minutiae', [])
                            })
                    
                    if processed_files:
                        print(f"\n✅ Successfully processed {len(processed_files)} files for testing.")
                        
                        # Test different thresholds
                        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
                        print(f"\n🧪 Testing thresholds: {thresholds}")
                        print("-" * 60)
                        
                        for threshold in thresholds:
                            duplicates = processor.test_threshold(processed_files, threshold)
                            print(f"Threshold {threshold:.1f}: {len(duplicates)} duplicates found")
                            
                            if duplicates:
                                print(f"  Examples:")
                                for dup in duplicates[:3]:  # Show first 3
                                    print(f"    {dup['file1']['filename']} ↔ {dup['file2']['filename']} (similarity: {dup['similarity']:.3f})")
                        
                        print("-" * 60)
                    else:
                        print("❌ No files could be processed for testing.")
                else:
                    print("No fingerprint files found!")
                    
            elif choice == '6':
                print("\nGoodbye!")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1-6.")
    
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ An error occurred: {e}")
    finally:
        # Cleanup
        processor.cleanup()

if __name__ == "__main__":
    main() 