"""
Integration Example: Combining Fingerprint and Face Authentication
This example shows how to integrate the fingerprint system with the existing face recognition system
"""

import sys
import os
from typing import Dict, Optional, Tuple

# Add parent directory to path to import face recognition modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fingerprint_auth.fingerprint_main import FingerprintAuthSystem


class MultiModalAuthSystem:
    """Combined fingerprint and face authentication system"""
    
    def __init__(self, fingerprint_threshold: float = 0.7):
        """
        Initialize the multi-modal authentication system
        
        Args:
            fingerprint_threshold: Similarity threshold for fingerprint matching
        """
        self.fingerprint_system = FingerprintAuthSystem(
            simulation_mode=True, 
            threshold=fingerprint_threshold
        )
        self.face_system_available = False
        
        # Try to import face recognition modules
        try:
            # Import face recognition modules here when available
            # from face_detector import FaceDetector
            # from liveness_detector import LivenessDetector
            # self.face_detector = FaceDetector()
            # self.liveness_detector = LivenessDetector()
            # self.face_system_available = True
            print("Face recognition system integration ready (imports commented out)")
        except ImportError as e:
            print(f"Face recognition system not available: {e}")
    
    def register_user_multimodal(self) -> Dict[str, bool]:
        """
        Register user with both fingerprint and face recognition
        
        Returns:
            Dictionary with registration results for each modality
        """
        results = {
            'fingerprint': False,
            'face': False
        }
        
        print("=== Multi-Modal User Registration ===")
        print("This will register your fingerprints and face for authentication.")
        
        # Register fingerprints
        print("\n1. Fingerprint Registration")
        results['fingerprint'] = self.fingerprint_system.register_user()
        
        # Register face (when available)
        if self.face_system_available:
            print("\n2. Face Registration")
            # results['face'] = self._register_face()
            print("Face registration not implemented in this example")
        else:
            print("\n2. Face Registration (skipped - system not available)")
        
        # Summary
        print(f"\n=== Registration Summary ===")
        print(f"Fingerprint: {'✓ Success' if results['fingerprint'] else '✗ Failed'}")
        print(f"Face: {'✓ Success' if results['face'] else '✗ Failed'}")
        
        return results
    
    def authenticate_user_multimodal(self, 
                                   require_fingerprint: bool = True,
                                   require_face: bool = False) -> bool:
        """
        Authenticate user using multiple modalities
        
        Args:
            require_fingerprint: Whether fingerprint authentication is required
            require_face: Whether face authentication is required
            
        Returns:
            True if authentication successful, False otherwise
        """
        print("=== Multi-Modal Authentication ===")
        
        results = {
            'fingerprint': False,
            'face': False
        }
        
        # Fingerprint authentication
        if require_fingerprint:
            print("\n1. Fingerprint Authentication")
            results['fingerprint'] = self.fingerprint_system.authenticate_user()
        
        # Face authentication (when available)
        if require_face and self.face_system_available:
            print("\n2. Face Authentication")
            # results['face'] = self._authenticate_face()
            print("Face authentication not implemented in this example")
        elif require_face:
            print("\n2. Face Authentication (skipped - system not available)")
        
        # Determine overall result
        if require_fingerprint and require_face:
            success = results['fingerprint'] and results['face']
        elif require_fingerprint:
            success = results['fingerprint']
        elif require_face:
            success = results['face']
        else:
            success = False
        
        # Summary
        print(f"\n=== Authentication Summary ===")
        if require_fingerprint:
            print(f"Fingerprint: {'✓ Pass' if results['fingerprint'] else '✗ Fail'}")
        if require_face:
            print(f"Face: {'✓ Pass' if results['face'] else '✗ Fail'}")
        print(f"Overall: {'✓ Access Granted' if success else '✗ Access Denied'}")
        
        return success
    
    def get_system_status(self) -> Dict[str, bool]:
        """Get status of all authentication systems"""
        return {
            'fingerprint_registered': self.fingerprint_system.is_registered,
            'face_system_available': self.face_system_available,
            'fingerprint_system_ready': True
        }
    
    def show_system_info(self):
        """Display information about the multi-modal system"""
        print("=== Multi-Modal Authentication System ===")
        
        status = self.get_system_status()
        
        print(f"Fingerprint System:")
        print(f"  Status: {'Ready' if status['fingerprint_system_ready'] else 'Not Ready'}")
        print(f"  Registered: {'Yes' if status['fingerprint_registered'] else 'No'}")
        print(f"  Mode: Simulation (mouse/touchpad input)")
        
        print(f"\nFace Recognition System:")
        print(f"  Status: {'Available' if status['face_system_available'] else 'Not Available'}")
        print(f"  Integration: Ready for implementation")
        
        print(f"\nSecurity Level:")
        if status['fingerprint_registered'] and status['face_system_available']:
            print(f"  Multi-factor authentication available")
        elif status['fingerprint_registered']:
            print(f"  Single-factor authentication (fingerprint)")
        else:
            print(f"  No authentication configured")


def main():
    """Main function for multi-modal authentication system"""
    print("=== Multi-Modal Biometric Authentication System ===")
    print("This system combines fingerprint and face recognition for enhanced security.")
    
    # Initialize the system
    auth_system = MultiModalAuthSystem(fingerprint_threshold=0.7)
    
    while True:
        print("\n" + "="*60)
        print("MULTI-MODAL AUTHENTICATION MENU")
        print("="*60)
        
        # Show system status
        auth_system.show_system_info()
        
        print("\nOptions:")
        print("1. Register user (fingerprint + face)")
        print("2. Authenticate with fingerprint only")
        print("3. Authenticate with fingerprint + face")
        print("4. Fingerprint system menu")
        print("5. Show system information")
        print("6. Exit")
        
        try:
            choice = input("\nEnter your choice (1-6): ").strip()
            
            if choice == '1':
                auth_system.register_user_multimodal()
                
            elif choice == '2':
                if auth_system.authenticate_user_multimodal(require_fingerprint=True, require_face=False):
                    print("✓ Access granted!")
                else:
                    print("✗ Access denied!")
                    
            elif choice == '3':
                if auth_system.authenticate_user_multimodal(require_fingerprint=True, require_face=True):
                    print("✓ Access granted!")
                else:
                    print("✗ Access denied!")
                    
            elif choice == '4':
                # Launch fingerprint system menu
                print("\nLaunching fingerprint system...")
                from fingerprint_auth.fingerprint_main import main as fingerprint_main
                fingerprint_main()
                
            elif choice == '5':
                auth_system.show_system_info()
                
            elif choice == '6':
                print("Goodbye!")
                break
                
            else:
                print("Invalid choice. Please enter a number between 1-6.")
                
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")


if __name__ == "__main__":
    main() 