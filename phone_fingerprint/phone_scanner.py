"""
Phone-based Fingerprint Scanner
Receives fingerprint data from a phone's fingerprint sensor via various methods:
1. Web API (Flask server)
2. Bluetooth communication
3. USB connection
4. File transfer (manual upload)
"""

import json
import os
import time
import hashlib
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
import numpy as np
from datetime import datetime
import base64
import cv2
from PIL import Image
import io

@dataclass
class PhoneFingerprintData:
    """Data structure for storing phone fingerprint information"""
    finger_id: str  # e.g., "thumb_left", "index_right"
    sensor_type: str  # "optical", "capacitive", "ultrasonic"
    raw_data: bytes  # Raw sensor data
    processed_data: Dict  # Processed fingerprint features
    quality_score: float
    timestamp: float
    device_info: Dict  # Phone model, sensor info, etc.
    metadata: Dict  # Additional metadata

class PhoneFingerprintScanner:
    def __init__(self, data_dir: str = "phone_fingerprint_data"):
        self.data_dir = data_dir
        self.ensure_data_directory()
        self.finger_names = [
            "thumb_left", "index_left", "middle_left", "ring_left", "pinky_left",
            "thumb_right", "index_right", "middle_right", "ring_right", "pinky_right"
        ]
        self.scanned_fingers = set()
        self.fingerprint_data = {}
        self.load_existing_data()
        
    def ensure_data_directory(self):
        """Create data directory if it doesn't exist"""
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
            print(f"Created data directory: {self.data_dir}")
    
    def load_existing_data(self):
        """Load existing fingerprint data from files"""
        for finger_name in self.finger_names:
            file_path = os.path.join(self.data_dir, f"{finger_name}.json")
            if os.path.exists(file_path):
                try:
                    with open(file_path, 'r') as f:
                        data_dict = json.load(f)
                        # Convert back to dataclass
                        self.fingerprint_data[finger_name] = PhoneFingerprintData(**data_dict)
                        self.scanned_fingers.add(finger_name)
                except Exception as e:
                    print(f"Error loading {finger_name}: {e}")
    
    def process_phone_fingerprint_data(self, raw_data: bytes, sensor_type: str = "optical") -> Dict:
        """
        Process raw fingerprint data from phone sensor
        This simulates the processing that would happen on the phone
        """
        # Convert raw bytes to numpy array (simulating sensor data)
        data_array = np.frombuffer(raw_data, dtype=np.uint8)
        
        # Simulate fingerprint feature extraction
        # In a real implementation, this would use actual fingerprint processing algorithms
        features = {
            "minutiae_points": self._extract_minutiae(data_array),
            "ridge_patterns": self._extract_ridge_patterns(data_array),
            "core_points": self._find_core_points(data_array),
            "delta_points": self._find_delta_points(data_array),
            "quality_metrics": self._calculate_quality_metrics(data_array)
        }
        
        return features
    
    def _extract_minutiae(self, data_array: np.ndarray) -> List[Dict]:
        """Extract minutiae points (ridge endings and bifurcations)"""
        # Simulate minutiae extraction
        num_points = min(50, len(data_array) // 100)
        minutiae = []
        
        for i in range(num_points):
            x = int(data_array[i * 10] % 200)
            y = int(data_array[i * 10 + 1] % 200)
            angle = float(data_array[i * 10 + 2] % 360)
            minutiae.append({
                "x": x, "y": y, "angle": angle,
                "type": "ending" if i % 2 == 0 else "bifurcation"
            })
        
        return minutiae
    
    def _extract_ridge_patterns(self, data_array: np.ndarray) -> Dict:
        """Extract ridge pattern information"""
        # Simulate ridge pattern analysis
        return {
            "ridge_count": len(data_array) // 100,
            "ridge_density": len(data_array) / 1000.0,
            "pattern_type": ["loop", "whorl", "arch"][len(data_array) % 3],
            "orientation_map": self._generate_orientation_map(data_array)
        }
    
    def _find_core_points(self, data_array: np.ndarray) -> List[Dict]:
        """Find core points (center of fingerprint)"""
        # Simulate core point detection
        core_points = []
        for i in range(min(3, len(data_array) // 200)):
            x = int(data_array[i * 50] % 200)
            y = int(data_array[i * 50 + 1] % 200)
            core_points.append({"x": x, "y": y, "confidence": 0.8 + (i * 0.1)})
        
        return core_points
    
    def _find_delta_points(self, data_array: np.ndarray) -> List[Dict]:
        """Find delta points (triangular regions)"""
        # Simulate delta point detection
        delta_points = []
        for i in range(min(2, len(data_array) // 300)):
            x = int(data_array[i * 75] % 200)
            y = int(data_array[i * 75 + 1] % 200)
            delta_points.append({"x": x, "y": y, "confidence": 0.7 + (i * 0.1)})
        
        return delta_points
    
    def _generate_orientation_map(self, data_array: np.ndarray) -> List[List[float]]:
        """Generate orientation map for ridge patterns"""
        # Simulate orientation map
        size = 20
        orientation_map = []
        for i in range(size):
            row = []
            for j in range(size):
                idx = (i * size + j) % len(data_array)
                angle = float(data_array[idx] % 180)
                row.append(angle)
            orientation_map.append(row)
        
        return orientation_map
    
    def _calculate_quality_metrics(self, data_array: np.ndarray) -> Dict:
        """Calculate quality metrics for the fingerprint"""
        # Simulate quality assessment
        return {
            "clarity": min(1.0, len(data_array) / 1000.0),
            "coverage": min(1.0, len(data_array) / 800.0),
            "contrast": 0.7 + (len(data_array) % 100) / 1000.0,
            "overall_score": min(1.0, len(data_array) / 1200.0)
        }
    
    def receive_fingerprint_from_phone(self, finger_name: str, raw_data: bytes, 
                                     sensor_type: str = "optical", 
                                     device_info: Dict = None) -> Optional[PhoneFingerprintData]:
        """
        Receive and process fingerprint data from phone
        """
        if finger_name not in self.finger_names:
            print(f"Invalid finger name: {finger_name}")
            return None
        
        if finger_name in self.scanned_fingers:
            print(f"Finger {finger_name} already scanned. Skipping...")
            return None
        
        # Process the raw data
        processed_data = self.process_phone_fingerprint_data(raw_data, sensor_type)
        
        # Calculate quality score
        quality_score = processed_data["quality_metrics"]["overall_score"]
        
        # Create fingerprint data object
        fingerprint_data = PhoneFingerprintData(
            finger_id=finger_name,
            sensor_type=sensor_type,
            raw_data=raw_data,
            processed_data=processed_data,
            quality_score=quality_score,
            timestamp=time.time(),
            device_info=device_info or {"model": "Unknown", "sensor": sensor_type},
            metadata={"source": "phone_sensor", "version": "1.0"}
        )
        
        # Save the data
        self.save_fingerprint_data(finger_name, fingerprint_data)
        self.scanned_fingers.add(finger_name)
        self.fingerprint_data[finger_name] = fingerprint_data
        
        print(f"Successfully processed {finger_name} (Quality: {quality_score:.2f})")
        return fingerprint_data
    
    def save_fingerprint_data(self, finger_name: str, data: PhoneFingerprintData):
        """Save fingerprint data to JSON file"""
        file_path = os.path.join(self.data_dir, f"{finger_name}.json")
        
        # Convert dataclass to dict, handling bytes serialization
        data_dict = asdict(data)
        data_dict["raw_data"] = base64.b64encode(data.raw_data).decode('utf-8')
        
        with open(file_path, 'w') as f:
            json.dump(data_dict, f, indent=2)
    
    def load_fingerprint_data(self, finger_name: str) -> Optional[PhoneFingerprintData]:
        """Load fingerprint data from JSON file"""
        file_path = os.path.join(self.data_dir, f"{finger_name}.json")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'r') as f:
                data_dict = json.load(f)
                # Convert base64 back to bytes
                data_dict["raw_data"] = base64.b64decode(data_dict["raw_data"])
                return PhoneFingerprintData(**data_dict)
        except Exception as e:
            print(f"Error loading {finger_name}: {e}")
            return None
    
    def get_registration_status(self) -> Dict[str, bool]:
        """Get status of which fingers have been registered"""
        return {finger: finger in self.scanned_fingers for finger in self.finger_names}
    
    def reset_registration(self):
        """Delete all stored fingerprint data"""
        for finger_name in self.finger_names:
            file_path = os.path.join(self.data_dir, f"{finger_name}.json")
            if os.path.exists(file_path):
                os.remove(file_path)
        
        self.scanned_fingers.clear()
        self.fingerprint_data.clear()
        print("All fingerprint data has been reset.")
    
    def generate_test_data(self, finger_name: str) -> bytes:
        """
        Generate test fingerprint data for demonstration
        In a real implementation, this would come from the phone's sensor
        """
        # Generate realistic-looking fingerprint data
        np.random.seed(hash(finger_name) % 1000)
        data_size = np.random.randint(800, 1200)
        test_data = np.random.bytes(data_size)
        return test_data
    
    def simulate_phone_scan(self, finger_name: str) -> Optional[PhoneFingerprintData]:
        """
        Simulate receiving data from a phone's fingerprint sensor
        This is for testing purposes
        """
        print(f"Simulating phone fingerprint scan for {finger_name}...")
        
        # Generate test data
        raw_data = self.generate_test_data(finger_name)
        
        # Simulate device info
        device_info = {
            "model": "Test Phone",
            "sensor": "optical",
            "resolution": "1920x1080",
            "os": "Android 12"
        }
        
        return self.receive_fingerprint_from_phone(
            finger_name, raw_data, "optical", device_info
        ) 