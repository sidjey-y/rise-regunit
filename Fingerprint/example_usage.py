#!/usr/bin/env python3
"""
Example Usage of the 1 User = 10 Fingers Enrollment System
This demonstrates how the system works with user management.
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def example_enrollment():
    """Example of how the enrollment system works"""
    print("🔐 EXAMPLE: 1 User = 10 Fingers Enrollment System")
    print("=" * 60)
    
    try:
        from comprehensive_enrollment_system import ComprehensiveEnrollmentSystem
        
        # Example 1: Enroll User "john_doe"
        print("\n📝 Example 1: Enrolling User 'john_doe'")
        print("-" * 40)
        
        system1 = ComprehensiveEnrollmentSystem()
        
        # This would normally prompt for user input
        # For demo purposes, we'll show the structure
        print("User ID: john_doe")
        print("Required fingers:")
        print("  - Left: thumb, index, middle, ring, little")
        print("  - Right: thumb, index, middle, ring, little")
        print("Total: 10 fingers")
        
        # Example 2: Enroll User "jane_smith" 
        print("\n📝 Example 2: Enrolling User 'jane_smith'")
        print("-" * 40)
        
        system2 = ComprehensiveEnrollmentSystem()
        
        print("User ID: jane_smith")
        print("Required fingers:")
        print("  - Left: thumb, index, middle, ring, little")
        print("  - Right: thumb, index, middle, ring, little")
        print("Total: 10 fingers")
        
        # Key Points
        print("\n🎯 Key Points:")
        print("=" * 40)
        print("✅ Each user gets exactly 10 fingers")
        print("✅ Duplication detection only within same user")
        print("✅ Different users can have same finger types")
        print("✅ Files saved as: user_<userid>_enrollment_<timestamp>.json")
        print("✅ No cross-user data conflicts")
        
        # Example JSON Structure
        print("\n📁 Example JSON Structure:")
        print("-" * 40)
        print("""
{
  "user_info": {
    "user_id": "john_doe",
    "enrollment_date": "2025-01-15",
    "completion_time": "14:30:25",
    "total_fingers_enrolled": 10,
    "session_duration_seconds": 245.7
  },
  "enrolled_fingers": {
    "left_thumb": {
      "user_id": "john_doe",
      "hand": "left",
      "finger_type": "thumb",
      "position": 0,
      "timestamp": "2025-01-15T14:25:30",
      "minutiae_points": [...],
      "raw_image_data_b64": "..."
    }
    // ... 9 more fingers
  }
}
        """)
        
        print("✅ Example completed successfully!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Example failed: {e}")
        return False

def main():
    """Main function"""
    print("🚀 1 User = 10 Fingers Enrollment System Example")
    print("=" * 60)
    
    success = example_enrollment()
    
    if success:
        print("\n✅ Example completed successfully!")
        print("Run 'python run_enrollment.py' to start actual enrollment")
    else:
        print("\n❌ Example failed!")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
