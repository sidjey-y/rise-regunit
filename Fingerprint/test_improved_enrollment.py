#!/usr/bin/env python3
"""
Test Improved Enrollment System
Tests the enhanced enrollment system with better scanner state management
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_improved_enrollment():
    """Test the improved enrollment system"""
    try:
        from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem
        
        print("🧪 TESTING IMPROVED ENROLLMENT SYSTEM")
        print("=" * 60)
        print("Improvements:")
        print("  ✅ Added scanner state clearing between enrollments")
        print("  ✅ Better buffer management for subsequent fingers")
        print("  ✅ Enhanced error handling for multi-finger sessions")
        print("=" * 60)
        
        # Initialize the system
        enrollment_system = ComprehensiveEnrollmentSystem()
        
        if not enrollment_system.initialize():
            print("❌ Failed to initialize scanner")
            return False
        
        print("✅ Scanner initialized successfully")
        
        # Ask for user ID
        user_id = input("\nEnter user ID for testing: ").strip()
        if not user_id:
            print("❌ User ID is required")
            return False
        
        print(f"\n🔄 Starting improved enrollment for user: {user_id}")
        print()
        print("Key improvements:")
        print("  • Scanner state is cleared after each successful enrollment")
        print("  • Buffer management prevents interference between fingers")
        print("  • Better timeout handling for subsequent finger scans")
        print()
        
        # Start enrollment
        success = enrollment_system.enroll_user(user_id)
        
        if success:
            print("\n✅ IMPROVED ENROLLMENT COMPLETED SUCCESSFULLY!")
            print("All fingers should now enroll without 'failed attempt' issues")
        else:
            print("\n❌ Enrollment failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    try:
        success = test_improved_enrollment()
        if success:
            print("\n🎉 Test completed successfully!")
        else:
            print("\n⚠️  Test encountered issues")
        
        return success
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        return False
    except Exception as e:
        print(f"\n❌ Test error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
