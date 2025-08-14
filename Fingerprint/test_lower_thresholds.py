#!/usr/bin/env python3
"""
Test Lower Similarity Thresholds
Tests the enrollment system with reduced sensitivity for false positives
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def test_lower_thresholds():
    """Test the enrollment system with lower similarity thresholds"""
    try:
        from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem
        
        print("🔧 TESTING LOWER SIMILARITY THRESHOLDS")
        print("=" * 60)
        print("Threshold Changes:")
        print("  📉 Same hand thumb: 40% → 25% (15% reduction)")
        print("  📉 Different hand thumb: 55% → 35% (20% reduction)")
        print("  📉 Non-thumb fingers: 40% → 25% (15% reduction)")
        print()
        print("Benefits:")
        print("  ✅ Fewer false positive 'similar finger' detections")
        print("  ✅ More permissive enrollment for different fingers")
        print("  ✅ Reduces 'keeps detecting similar' issues")
        print("=" * 60)
        
        # Initialize the system
        enrollment_system = ComprehensiveEnrollmentSystem()
        
        if not enrollment_system.initialize():
            print("❌ Failed to initialize scanner")
            return False
        
        print("✅ Scanner initialized with new thresholds")
        
        # Ask for user ID
        user_id = input("\nEnter user ID for testing new thresholds: ").strip()
        if not user_id:
            print("❌ User ID is required")
            return False
        
        print(f"\n🔄 Starting enrollment with lower thresholds for user: {user_id}")
        print()
        print("Key improvements:")
        print("  • Much less sensitive similarity detection")
        print("  • Different fingers won't be flagged as 'similar'")
        print("  • Enrollment should proceed more smoothly")
        print()
        
        # Start enrollment
        success = enrollment_system.run_complete_enrollment(user_id)
        
        if success:
            print("\n✅ ENROLLMENT WITH LOWER THRESHOLDS COMPLETED!")
            print("The system should now be much less sensitive to false positives")
        else:
            print("\n❌ Enrollment failed")
            
        return success
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    """Main function"""
    try:
        success = test_lower_thresholds()
        if success:
            print("\n🎉 Lower threshold test completed successfully!")
            print("The similarity detection should now be much more reasonable")
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
