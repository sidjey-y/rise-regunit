#!/usr/bin/env python3
"""
Quick Enrollment Test
Simple test to verify fingerprint enrollment without AI dependencies
"""

import sys
import time
from datetime import datetime

def test_enrollment():
    """Test enrollment without AI dependencies"""
    try:
        print("🧪 QUICK ENROLLMENT TEST")
        print("=" * 50)
        print("Testing fingerprint enrollment without AI dependencies...")
        
        # Import the comprehensive system
        from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem
        
        print("✅ Import successful")
        
        # Initialize system
        system = ComprehensiveEnrollmentSystem()
        
        print("✅ System created")
        
        # Test initialization
        if not system.initialize():
            print("❌ Scanner initialization failed")
            print("   Make sure your fingerprint scanner is connected to COM4")
            return False
        
        print("✅ Scanner initialized successfully")
        
        # Get user ID
        user_id = input("\nEnter User ID for testing: ").strip()
        if not user_id:
            print("❌ User ID is required")
            return False
        
        print(f"\n🔄 Starting enrollment for user: {user_id}")
        
        # Start enrollment
        success = system.enroll_user(user_id)
        
        if success:
            print("\n✅ ENROLLMENT COMPLETED SUCCESSFULLY!")
            print("All fingers should now enroll properly")
        else:
            print("\n❌ Enrollment failed")
            
        return success
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Missing required modules")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    try:
        success = test_enrollment()
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
