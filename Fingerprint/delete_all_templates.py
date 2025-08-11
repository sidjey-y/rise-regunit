#!/usr/bin/env python3
"""
Delete All Fingerprint Templates
This script deletes ALL saved fingerprint templates from the scanner.
WARNING: This will permanently remove all enrolled fingerprints!
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def delete_all_templates():
    """Delete all fingerprint templates from the scanner"""
    try:
        from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem
        
        print("🗑️  DELETE ALL FINGERPRINT TEMPLATES")
        print("=" * 50)
        print("⚠️  WARNING: This will permanently delete ALL saved fingerprints!")
        print("⚠️  This action cannot be undone!")
        print()
        
        # Ask for confirmation
        confirm = input("Are you sure you want to delete ALL templates? Type 'DELETE ALL' to confirm: ")
        
        if confirm != "DELETE ALL":
            print("Operation cancelled. No templates were deleted.")
            return False
        
        # Double confirmation
        print()
        print("⚠️  FINAL WARNING: You are about to delete ALL fingerprint templates!")
        final_confirm = input("Type 'YES DELETE ALL' to proceed: ")
        
        if final_confirm != "YES DELETE ALL":
            print("Operation cancelled. No templates were deleted.")
            return False
        
        print()
        print("Initializing scanner...")
        
        # Initialize the system
        enrollment_system = ComprehensiveEnrollmentSystem()
        
        if not enrollment_system.initialize():
            print("❌ Failed to initialize scanner")
            return False
        
        print("✅ Scanner initialized successfully")
        
        # Get scanner object
        scanner = enrollment_system.scanner
        
        # Count existing templates
        print("🔍 Counting existing templates...")
        template_count = 0
        
        try:
            # Try to get the number of templates
            # This is scanner-specific - PyFingerprint may have different methods
            for position in range(1000):  # Check first 1000 positions
                try:
                    # Try to load template at this position
                    scanner.loadTemplate(position)
                    template_count += 1
                    print(f"   Found template at position {position}")
                except:
                    # No template at this position
                    continue
        except Exception as e:
            print(f"⚠️  Could not count templates: {e}")
            print("   Proceeding with deletion anyway...")
        
        if template_count == 0:
            print("✅ No templates found to delete")
            return True
        
        print(f"📊 Found {template_count} template(s) to delete")
        print()
        
        # Delete all templates
        print("🗑️  Deleting all templates...")
        deleted_count = 0
        
        for position in range(1000):  # Check first 1000 positions
            try:
                # Try to load template at this position
                scanner.loadTemplate(position)
                
                # Delete the template
                try:
                    scanner.deleteTemplate(position)
                    deleted_count += 1
                    print(f"   ✅ Deleted template at position {position}")
                except Exception as e:
                    print(f"   ❌ Failed to delete template at position {position}: {e}")
                    
            except:
                # No template at this position
                continue
        
        print()
        print(f"🎉 Template deletion completed!")
        print(f"   Deleted: {deleted_count} template(s)")
        
        # Verify deletion
        print()
        print("🔍 Verifying deletion...")
        remaining_count = 0
        
        for position in range(1000):
            try:
                scanner.loadTemplate(position)
                remaining_count += 1
                print(f"   ⚠️  Template still exists at position {position}")
            except:
                continue
        
        if remaining_count == 0:
            print("✅ All templates successfully deleted!")
        else:
            print(f"⚠️  {remaining_count} template(s) still remain")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required files are in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Template deletion failed: {e}")
        return False

def main():
    """Main function"""
    print("🗑️  Fingerprint Template Deletion Utility")
    print("=" * 50)
    print("This utility will delete ALL saved fingerprint templates from your scanner.")
    print("Use this when you want to start fresh or clear all enrolled fingerprints.")
    print()
    
    success = delete_all_templates()
    
    if success:
        print("\n✅ Template deletion completed successfully!")
        print("   Your scanner is now clean and ready for new enrollments.")
    else:
        print("\n❌ Template deletion failed!")
        print("   Check the error messages above for details.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
