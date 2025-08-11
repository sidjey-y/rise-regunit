#!/usr/bin/env python3
"""
USB Fingerprint Scanner Test
Tests for USB fingerprint scanner devices, not cameras
"""

import sys
import os

def test_usb_devices():
    """Test for USB fingerprint scanner devices"""
    print("🔍 USB Fingerprint Scanner Detection")
    print("=" * 50)
    
    try:
        # Method 1: Check if pyusb is available
        try:
            import usb.core
            import usb.util
            print("✅ pyusb library available")
            
            # Find USB devices
            devices = usb.core.find(find_all=True)
            print(f"\nFound {len(list(devices))} USB devices:")
            
            for device in devices:
                try:
                    vendor_id = device.idVendor
                    product_id = device.idProduct
                    manufacturer = usb.util.get_string(device, device.iManufacturer) if device.iManufacturer else "Unknown"
                    product = usb.util.get_string(device, device.iProduct) if device.iProduct else "Unknown"
                    
                    print(f"  📱 VID:{vendor_id:04x} PID:{product_id:04x} - {manufacturer} {product}")
                    
                    # Check if this looks like a fingerprint scanner
                    if any(keyword in product.lower() for keyword in ['fingerprint', 'scanner', 'reader', 'biometric']):
                        print(f"    🎯 POTENTIAL FINGERPRINT SCANNER!")
                        print(f"    Manufacturer: {manufacturer}")
                        print(f"    Product: {product}")
                        
                except Exception as e:
                    print(f"  ❌ Error reading device info: {e}")
                    
        except ImportError:
            print("❌ pyusb library not available")
            print("   Install with: pip install pyusb")
    
    except Exception as e:
        print(f"❌ USB detection error: {e}")
    
    print("\n" + "=" * 50)

def test_windows_devices():
    """Test for Windows device detection"""
    print("\n🔍 Windows Device Detection")
    print("=" * 30)
    
    try:
        # Method 2: Check Windows registry for USB devices
        import winreg
        
        print("Checking Windows USB device registry...")
        
        # Check USB devices in registry
        try:
            key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SYSTEM\CurrentControlSet\Enum\USB")
            print("✅ USB registry key accessible")
            
            # List subkeys (device IDs)
            i = 0
            while True:
                try:
                    device_id = winreg.EnumKey(key, i)
                    print(f"  USB Device: {device_id}")
                    i += 1
                except WindowsError:
                    break
                    
        except Exception as e:
            print(f"❌ USB registry access error: {e}")
            
    except ImportError:
        print("❌ Windows registry access not available")
    except Exception as e:
        print(f"❌ Windows device detection error: {e}")
    
    print("=" * 30)

def test_pyserial():
    """Test for serial/USB communication"""
    print("\n🔍 Serial/USB Communication Test")
    print("=" * 35)
    
    try:
        import serial.tools.list_ports
        
        print("✅ pyserial library available")
        ports = serial.tools.list_ports.comports()
        
        if ports:
            print(f"Found {len(ports)} serial ports:")
            for port in ports:
                print(f"  📡 {port.device}: {port.description}")
                if port.vid and port.pid:
                    print(f"    VID:{port.vid:04x} PID:{port.pid:04x}")
                    
                    # Check if this looks like a fingerprint scanner
                    if any(keyword in port.description.lower() for keyword in ['fingerprint', 'scanner', 'reader', 'biometric']):
                        print(f"    🎯 POTENTIAL FINGERPRINT SCANNER!")
        else:
            print("❌ No serial ports found")
            
    except ImportError:
        print("❌ pyserial library not available")
        print("   Install with: pip install pyserial")
    except Exception as e:
        print(f"❌ Serial detection error: {e}")
    
    print("=" * 35)

def main():
    """Main test function"""
    print("🔍 USB FINGERPRINT SCANNER DETECTION TEST")
    print("This test looks for USB fingerprint scanner devices")
    print("NOT cameras - actual USB hardware devices")
    print("=" * 60)
    
    try:
        # Test 1: USB device detection
        test_usb_devices()
        
        # Test 2: Windows device detection
        test_windows_devices()
        
        # Test 3: Serial/USB communication
        test_pyserial()
        
        print("\n🎉 USB scanner detection completed!")
        print("\nIf you see 'POTENTIAL FINGERPRINT SCANNER', your device is detected!")
        print("If not, the scanner might need drivers or different connection method.")
        
    except Exception as e:
        print(f"\n❌ Test error: {e}")
    
    print("\nPress Enter to exit...")
    input()

if __name__ == "__main__":
    main()





