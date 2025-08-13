#!/usr/bin/env python3
"""
Clear All Fingerprint Templates
This script deletes all stored fingerprint templates from the scanner.
Use this to reset the scanner and start fresh enrollment.
"""

import sys
import time
from pyfingerprint.pyfingerprint import PyFingerprint

def clear_all_templates():
    """Delete all fingerprint templates from the scanner"""
    try:
        print("🧹 Clearing all fingerprint templates...")
        print("=" * 50)
        
        # Initialize scanner
        print("🔌 Initializing scanner on COM4...")
        scanner = PyFingerprint('COM4', 57600, 0xFFFFFFFF, 0x00000000)
        
        if not scanner.verifyPassword():
            print("❌ Scanner password verification failed")
            return False
        
        print("✅ Scanner initialized successfully")
        
        # Get current template count
        template_count = scanner.getTemplateCount()
        print(f"📊 Current templates: {template_count}")
        
        if template_count == 0:
            print("✅ No templates to delete - scanner is already clean")
            return True
        
        # Confirm deletion
        print(f"\n⚠️  WARNING: This will delete ALL {template_count} templates!")
        confirm = input("Type 'YES' to confirm deletion: ")
        
        if confirm != 'YES':
            print("❌ Deletion cancelled")
            return False
        
        print(f"\n🗑️  Deleting {template_count} templates...")
        
        # Delete all templates
        deleted_count = 0
        for position in range(template_count):
            try:
                if scanner.deleteTemplate(position):
                    deleted_count += 1
                    print(f"   ✅ Deleted template at position {position}")
                else:
                    print(f"   ❌ Failed to delete template at position {position}")
                time.sleep(0.1)  # Small delay to avoid overwhelming the scanner
            except Exception as e:
                print(f"   ⚠️  Error deleting template at position {position}: {e}")
        
        # Verify deletion
        new_template_count = scanner.getTemplateCount()
        print(f"\n📊 Templates after deletion: {new_template_count}")
        
        if new_template_count == 0:
            print("🎉 All templates successfully deleted!")
            print("✅ Scanner is now clean and ready for fresh enrollment")
            return True
        else:
            print(f"⚠️  {new_template_count} templates still remain")
            return False
            
    except Exception as e:
        print(f"❌ Error clearing templates: {e}")
        return False

def main():
    """Main function"""
    print("🧹 Fingerprint Template Cleaner")
    print("=" * 40)
    
    try:
        success = clear_all_templates()
        return success
        
    except KeyboardInterrupt:
        print("\n⚠️  Operation interrupted by user")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)






