#!/usr/bin/env python3
"""
Clear All Templates and Restart Enrollment System
This script clears all fingerprint templates and allows you to start fresh.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def clear_and_restart():
    """Clear all templates and restart the system"""
    print("🧹 CLEAR ALL TEMPLATES AND RESTART")
    print("=" * 60)
    print("This will:")
    print("  1. Delete ALL fingerprint templates from scanner")
    print("  2. Clear any stored enrollment data")
    print("  3. Allow you to start fresh enrollment")
    print("=" * 60)
    
    try:
        from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem
        
        # Ask for confirmation
        confirm = input("\n⚠️  Are you sure you want to clear ALL templates? Type 'CLEAR ALL' to confirm: ")
        
        if confirm != "CLEAR ALL":
            print("Operation cancelled. No templates were deleted.")
            return False
        
        print("\n🗑️  Clearing all templates...")
        
        # Initialize the system
        enrollment_system = ComprehensiveEnrollmentSystem()
        
        if not enrollment_system.initialize():
            print("❌ Failed to initialize scanner")
            return False
        
        print("✅ Scanner initialized successfully")
        
        # Get scanner object
        scanner = enrollment_system.scanner
        
        # Count and delete all templates
        print("🔍 Counting existing templates...")
        template_count = 0
        deleted_count = 0
        
        # Check first 1000 positions for templates
        for position in range(1000):
            try:
                # Try to load template at this position
                scanner.loadTemplate(position)
                template_count += 1
                print(f"   Found template at position {position}")
                
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
        
        print(f"\n📊 Template deletion completed!")
        print(f"   Found: {template_count} template(s)")
        print(f"   Deleted: {deleted_count} template(s)")
        
        # Verify deletion
        print("\n🔍 Verifying deletion...")
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
        
        print("\n🎉 System cleared successfully!")
        print("You can now run 'python run_enrollment.py' to start fresh enrollment")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all required files are in the same directory.")
        return False
    except Exception as e:
        print(f"❌ Clear and restart failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 Clear All Templates and Restart")
    print("=" * 60)
    
    success = clear_and_restart()
    
    if success:
        print("\n✅ System cleared successfully!")
        print("Ready for fresh enrollment!")
    else:
        print("\n❌ Clear and restart failed!")
        print("Please check the error messages above")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
