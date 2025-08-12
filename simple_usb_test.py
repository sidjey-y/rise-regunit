import sys

print("🔍 Simple USB Device Test")
print("=" * 30)

# Test 1: Try to import USB libraries
print("Testing USB libraries...")

try:
    import usb.core
    print("✅ pyusb available")
except ImportError:
    print("❌ pyusb not available")

try:
    import serial
    print("✅ pyserial available")
except ImportError:
    print("❌ pyserial not available")

# Test 2: Check Windows registry
print("\nTesting Windows registry...")
try:
    import winreg
    print("✅ winreg available")
    
    # Try to access USB devices
    key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Enum\USB")
    print("✅ USB registry accessible")
    
    # Count devices
    count = 0
    try:
        i = 0
        while True:
            winreg.EnumKey(key, i)
            count += 1
            i += 1
    except WindowsError:
        pass
    
    print(f"Found {count} USB devices in registry")
    
except Exception as e:
    print(f"❌ Registry access failed: {e}")

print("\n�� Test completed!")






