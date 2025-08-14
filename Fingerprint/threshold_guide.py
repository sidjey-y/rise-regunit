#!/usr/bin/env python3
"""
Similarity Threshold Configuration Guide
Shows current thresholds and provides easy adjustment options
"""

def show_threshold_guide():
    """Display current threshold configuration"""
    print("🎯 FINGERPRINT SIMILARITY THRESHOLD GUIDE")
    print("=" * 60)
    print()
    print("CURRENT THRESHOLDS (After Reduction):")
    print("  📏 Same hand thumb:      25% (was 40%)")
    print("  📏 Different hand thumb: 35% (was 55%)")  
    print("  📏 Non-thumb fingers:    25% (was 40%)")
    print()
    print("WHAT THESE THRESHOLDS MEAN:")
    print("  • Lower % = Less sensitive = Fewer false positives")
    print("  • Higher % = More sensitive = More duplicate detection")
    print()
    print("RECOMMENDED ADJUSTMENTS:")
    print("  🔴 If STILL too sensitive (false positives):")
    print("      → Lower to 20% for all thresholds")
    print("      → Lower to 15% for all thresholds")
    print()
    print("  🟢 If MISSING real duplicates:")
    print("      → Raise to 30% for all thresholds")
    print("      → Raise to 35% for all thresholds")
    print()
    print("THRESHOLD LOCATION IN CODE:")
    print("  📁 File: comprehensive_enrollment_system.py")
    print("  📍 Method: check_duplicate_within_user()")
    print("  📍 Lines: ~192-203")
    print()
    print("QUICK EDIT COMMANDS:")
    print("  🔧 Even lower (20%): threshold = 0.20")
    print("  🔧 Much lower (15%): threshold = 0.15")
    print("  🔧 Slightly higher (30%): threshold = 0.30")
    print()
    print("=" * 60)

def main():
    show_threshold_guide()

if __name__ == "__main__":
    main()
