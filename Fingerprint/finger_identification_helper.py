#!/usr/bin/env python3
"""
Finger Identification Helper
Helps users identify which finger to scan next
"""

import logging
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Hand(Enum):
    LEFT = "left"
    RIGHT = "right"

class FingerType(Enum):
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    LITTLE = "little"

def show_finger_guide(hand: Hand, finger_type: FingerType):
    """Show detailed finger identification guide"""
    
    logger.info("👆 FINGER IDENTIFICATION GUIDE")
    logger.info("=" * 50)
    logger.info(f"🎯 TARGET: {hand.value.upper()} {finger_type.value.upper()}")
    logger.info("")
    
    # Hand identification
    logger.info("✋ HAND IDENTIFICATION:")
    if hand == Hand.LEFT:
        logger.info("   🫲 Use your LEFT hand")
        logger.info("   🫲 This is typically your non-writing hand")
        logger.info("   🫲 Look at your palm - thumb should be on the RIGHT side")
    else:
        logger.info("   🫱 Use your RIGHT hand") 
        logger.info("   🫱 This is typically your writing hand")
        logger.info("   🫱 Look at your palm - thumb should be on the LEFT side")
    
    logger.info("")
    
    # Finger identification
    logger.info("👆 FINGER IDENTIFICATION:")
    if finger_type == FingerType.THUMB:
        logger.info("   👍 THUMB:")
        logger.info("      • Shortest and widest finger")
        logger.info("      • Opposable to other fingers")
        logger.info("      • Used for gripping")
        logger.info("      • Place it flat on the sensor")
        
    elif finger_type == FingerType.INDEX:
        logger.info("   👉 INDEX FINGER (Pointer finger):")
        logger.info("      • Next to the thumb")
        logger.info("      • Used for pointing")
        logger.info("      • Usually the longest or second longest finger")
        logger.info("      • The finger you use to press buttons")
        
    elif finger_type == FingerType.MIDDLE:
        logger.info("   🖕 MIDDLE FINGER:")
        logger.info("      • In the center of your hand")
        logger.info("      • Usually the longest finger")
        logger.info("      • Between index and ring finger")
        logger.info("      • Easy to identify as the tallest")
        
    elif finger_type == FingerType.RING:
        logger.info("   💍 RING FINGER:")
        logger.info("      • Between middle and little finger")
        logger.info("      • Where wedding rings are worn")
        logger.info("      • Second shortest (after little finger)")
        logger.info("      • Weaker than other fingers")
        
    elif finger_type == FingerType.LITTLE:
        logger.info("   🤏 LITTLE FINGER (Pinky):")
        logger.info("      • Smallest and shortest finger")
        logger.info("      • On the outside edge of your hand")
        logger.info("      • Used for delicate tasks")
        logger.info("      • Also called 'pinky finger'")
    
    logger.info("")
    logger.info("📏 FINGER ORDER (from thumb to pinky):")
    logger.info("   1. 👍 Thumb")
    logger.info("   2. 👉 Index (pointer)")
    logger.info("   3. 🖕 Middle (tallest)")
    logger.info("   4. 💍 Ring")
    logger.info("   5. 🤏 Little (pinky)")
    logger.info("")
    logger.info("🎯 SUMMARY:")
    logger.info(f"   Hand: {hand.value.upper()}")
    logger.info(f"   Finger: {finger_type.value.upper()}")
    logger.info("=" * 50)

def finger_placement_tips():
    """General finger placement tips"""
    logger.info("📋 FINGER PLACEMENT TIPS:")
    logger.info("=" * 40)
    logger.info("✅ DO:")
    logger.info("   • Clean your finger and the scanner")
    logger.info("   • Place finger flat and centered")
    logger.info("   • Apply gentle, even pressure")
    logger.info("   • Keep finger still during scan")
    logger.info("   • Use the correct hand and finger")
    logger.info("")
    logger.info("❌ DON'T:")
    logger.info("   • Press too hard or too soft")
    logger.info("   • Move your finger during scan")
    logger.info("   • Use wet or dirty fingers")
    logger.info("   • Tilt your finger at an angle")
    logger.info("   • Rush the scanning process")
    logger.info("=" * 40)

def troubleshoot_wrong_finger():
    """Help troubleshoot wrong finger detection"""
    logger.info("🔧 WRONG FINGER TROUBLESHOOTING:")
    logger.info("=" * 50)
    logger.info("")
    logger.info("❓ If you get 'Wrong Finger Detected':")
    logger.info("")
    logger.info("1. 🔍 DOUBLE-CHECK YOUR FINGER:")
    logger.info("   • Count from thumb: 1=thumb, 2=index, 3=middle, 4=ring, 5=little")
    logger.info("   • Make sure you're using the correct finger type")
    logger.info("")
    logger.info("2. 🔍 DOUBLE-CHECK YOUR HAND:")
    logger.info("   • Look at your palm - where is your thumb?")
    logger.info("   • Left hand: thumb on right side of palm")
    logger.info("   • Right hand: thumb on left side of palm")
    logger.info("")
    logger.info("3. 🔍 CHECK FOR REPEATED FINGERS:")
    logger.info("   • Did you accidentally scan this finger before?")
    logger.info("   • Check the enrollment progress summary")
    logger.info("")
    logger.info("4. 🔍 SCANNER CLEANLINESS:")
    logger.info("   • Clean the scanner surface")
    logger.info("   • Clean your finger")
    logger.info("   • Try a different finger angle")
    logger.info("")
    logger.info("5. 🔍 FINGERPRINT QUALITY:")
    logger.info("   • Some fingers are naturally more similar")
    logger.info("   • Try pressing slightly differently")
    logger.info("   • Ensure full finger contact")
    logger.info("=" * 50)

def show_enrollment_order():
    """Show the standard enrollment order"""
    logger.info("📋 STANDARD ENROLLMENT ORDER:")
    logger.info("=" * 40)
    
    fingers = [
        (Hand.LEFT, FingerType.THUMB),
        (Hand.LEFT, FingerType.INDEX),
        (Hand.LEFT, FingerType.MIDDLE),
        (Hand.LEFT, FingerType.RING),
        (Hand.LEFT, FingerType.LITTLE),
        (Hand.RIGHT, FingerType.THUMB),
        (Hand.RIGHT, FingerType.INDEX),
        (Hand.RIGHT, FingerType.MIDDLE),
        (Hand.RIGHT, FingerType.RING),
        (Hand.RIGHT, FingerType.LITTLE)
    ]
    
    for i, (hand, finger) in enumerate(fingers, 1):
        emoji = "👍" if finger == FingerType.THUMB else "👉" if finger == FingerType.INDEX else "🖕" if finger == FingerType.MIDDLE else "💍" if finger == FingerType.RING else "🤏"
        hand_emoji = "🫲" if hand == Hand.LEFT else "🫱"
        logger.info(f"{i:2d}. {hand_emoji} {emoji} {hand.value.title()} {finger.value.title()}")
    
    logger.info("=" * 40)

def interactive_finger_helper():
    """Interactive finger identification helper"""
    logger.info("🤖 INTERACTIVE FINGER IDENTIFICATION HELPER")
    logger.info("=" * 60)
    
    try:
        # Get hand
        print("\n🫲🫱 Which hand are you trying to scan?")
        hand_input = input("Enter 'left' or 'right': ").strip().lower()
        
        if hand_input == 'left':
            hand = Hand.LEFT
        elif hand_input == 'right':
            hand = Hand.RIGHT
        else:
            logger.error("Invalid hand. Please enter 'left' or 'right'")
            return
        
        # Get finger
        print(f"\n👆 Which finger on your {hand.value} hand?")
        print("Options: thumb, index, middle, ring, little")
        finger_input = input("Enter finger type: ").strip().lower()
        
        finger_map = {
            'thumb': FingerType.THUMB,
            'index': FingerType.INDEX,
            'middle': FingerType.MIDDLE,
            'ring': FingerType.RING,
            'little': FingerType.LITTLE,
            'pinky': FingerType.LITTLE
        }
        
        if finger_input in finger_map:
            finger_type = finger_map[finger_input]
        else:
            logger.error("Invalid finger. Please enter: thumb, index, middle, ring, or little")
            return
        
        # Show detailed guide
        print("\n")
        show_finger_guide(hand, finger_type)
        
        # Ask if they want more help
        print("\n🔧 Need more help?")
        help_choice = input("Enter 'tips', 'troubleshoot', 'order', or 'done': ").strip().lower()
        
        if help_choice == 'tips':
            finger_placement_tips()
        elif help_choice == 'troubleshoot':
            troubleshoot_wrong_finger()
        elif help_choice == 'order':
            show_enrollment_order()
        else:
            logger.info("✅ Helper complete - good luck with your enrollment!")
            
    except KeyboardInterrupt:
        logger.info("\nHelper interrupted by user")
    except Exception as e:
        logger.error(f"Helper error: {e}")

def main():
    """Main function"""
    interactive_finger_helper()

if __name__ == "__main__":
    main()
