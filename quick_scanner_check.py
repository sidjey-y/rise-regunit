import cv2

print("🔍 Quick Scanner Check")
print("=" * 30)

# Test device 0 (usually built-in camera)
print("Testing device 0...")
cap0 = cv2.VideoCapture(0)
if cap0.isOpened():
    print("✅ Device 0: ACCESSIBLE")
    ret, frame = cap0.read()
    if ret:
        print(f"   Frame shape: {frame.shape}")
    cap0.release()
else:
    print("❌ Device 0: Not accessible")

# Test device 1 (often USB devices like fingerprint scanners)
print("\nTesting device 1...")
cap1 = cv2.VideoCapture(1)
if cap1.isOpened():
    print("✅ Device 1: ACCESSIBLE")
    ret, frame = cap1.read()
    if ret:
        print(f"   Frame shape: {frame.shape}")
    cap1.release()
else:
    print("❌ Device 1: Not accessible")

# Test device 2
print("\nTesting device 2...")
cap2 = cv2.VideoCapture(2)
if cap2.isOpened():
    print("✅ Device 2: ACCESSIBLE")
    ret, frame = cap2.read()
    if ret:
        print(f"   Frame shape: {frame.shape}")
    cap2.release()
else:
    print("❌ Device 2: Not accessible")

print("\n🎉 Scanner check completed!")
print("\nIf you see 'ACCESSIBLE' for any device, your scanner is working!")



