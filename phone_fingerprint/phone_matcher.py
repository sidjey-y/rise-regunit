"""
Phone-based Fingerprint Matcher
Compares fingerprint data received from phone sensors for authentication
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
from phone_scanner import PhoneFingerprintScanner, PhoneFingerprintData
import json
import time

@dataclass
class PhoneMatchResult:
    """Result of phone fingerprint matching"""
    is_match: bool
    similarity_score: float
    matched_finger: Optional[str]
    confidence: float
    match_details: Dict
    processing_time: float

class PhoneFingerprintMatcher:
    def __init__(self, scanner: PhoneFingerprintScanner, threshold: float = 0.75):
        self.scanner = scanner
        self.threshold = threshold
        
    def match_fingerprint(self, test_data: PhoneFingerprintData, 
                         reference_data: Dict[str, PhoneFingerprintData]) -> PhoneMatchResult:
        """
        Match a test fingerprint against reference fingerprints
        """
        start_time = time.time()
        
        best_match = None
        best_score = 0.0
        best_finger = None
        
        for finger_name, ref_data in reference_data.items():
            score = self.calculate_similarity(test_data, ref_data)
            if score > best_score:
                best_score = score
                best_finger = finger_name
                best_match = ref_data
        
        processing_time = time.time() - start_time
        
        is_match = best_score >= self.threshold
        confidence = min(1.0, best_score / self.threshold) if is_match else 0.0
        
        match_details = {
            "best_match_finger": best_finger,
            "best_match_score": best_score,
            "threshold": self.threshold,
            "all_scores": {name: self.calculate_similarity(test_data, ref_data) 
                          for name, ref_data in reference_data.items()}
        }
        
        return PhoneMatchResult(
            is_match=is_match,
            similarity_score=best_score,
            matched_finger=best_finger,
            confidence=confidence,
            match_details=match_details,
            processing_time=processing_time
        )
    
    def calculate_similarity(self, data1: PhoneFingerprintData, 
                           data2: PhoneFingerprintData) -> float:
        """
        Calculate similarity between two fingerprint data objects
        Uses multiple factors for robust matching
        """
        # Weight factors for different similarity components
        weights = {
            "minutiae": 0.4,
            "ridge_patterns": 0.3,
            "quality": 0.2,
            "sensor_compatibility": 0.1
        }
        
        # Calculate individual similarity scores
        minutiae_similarity = self._calculate_minutiae_similarity(
            data1.processed_data["minutiae_points"],
            data2.processed_data["minutiae_points"]
        )
        
        ridge_similarity = self._calculate_ridge_pattern_similarity(
            data1.processed_data["ridge_patterns"],
            data2.processed_data["ridge_patterns"]
        )
        
        quality_similarity = self._calculate_quality_similarity(
            data1.processed_data["quality_metrics"],
            data2.processed_data["quality_metrics"]
        )
        
        sensor_similarity = self._calculate_sensor_compatibility(
            data1.sensor_type, data2.sensor_type,
            data1.device_info, data2.device_info
        )
        
        # Calculate weighted average
        total_similarity = (
            weights["minutiae"] * minutiae_similarity +
            weights["ridge_patterns"] * ridge_similarity +
            weights["quality"] * quality_similarity +
            weights["sensor_compatibility"] * sensor_similarity
        )
        
        return total_similarity
    
    def _calculate_minutiae_similarity(self, minutiae1: List[Dict], 
                                     minutiae2: List[Dict]) -> float:
        """
        Calculate similarity based on minutiae points
        """
        if not minutiae1 or not minutiae2:
            return 0.0
        
        # Normalize coordinates to 0-1 range
        max_coord = 200.0
        
        # Convert to numpy arrays for easier processing
        points1 = np.array([[m["x"]/max_coord, m["y"]/max_coord, m["angle"]/360.0] 
                           for m in minutiae1])
        points2 = np.array([[m["x"]/max_coord, m["y"]/max_coord, m["angle"]/360.0] 
                           for m in minutiae2])
        
        # Calculate pairwise distances
        distances = []
        for p1 in points1:
            for p2 in points2:
                # Euclidean distance for position
                pos_dist = np.sqrt(np.sum((p1[:2] - p2[:2])**2))
                # Angular distance for orientation
                angle_dist = min(abs(p1[2] - p2[2]), 1.0 - abs(p1[2] - p2[2]))
                # Combined distance
                total_dist = pos_dist + 0.5 * angle_dist
                distances.append(total_dist)
        
        if not distances:
            return 0.0
        
        # Convert distances to similarity scores
        similarities = [max(0, 1 - d) for d in distances]
        
        # Return average similarity
        return np.mean(similarities)
    
    def _calculate_ridge_pattern_similarity(self, pattern1: Dict, 
                                          pattern2: Dict) -> float:
        """
        Calculate similarity based on ridge patterns
        """
        similarities = []
        
        # Compare ridge count
        count1, count2 = pattern1["ridge_count"], pattern2["ridge_count"]
        if count1 > 0 and count2 > 0:
            count_similarity = 1 - abs(count1 - count2) / max(count1, count2)
            similarities.append(count_similarity)
        
        # Compare ridge density
        density1, density2 = pattern1["ridge_density"], pattern2["ridge_density"]
        if density1 > 0 and density2 > 0:
            density_similarity = 1 - abs(density1 - density2) / max(density1, density2)
            similarities.append(density_similarity)
        
        # Compare pattern type
        type1, type2 = pattern1["pattern_type"], pattern2["pattern_type"]
        type_similarity = 1.0 if type1 == type2 else 0.0
        similarities.append(type_similarity)
        
        # Compare orientation maps
        orient1, orient2 = pattern1["orientation_map"], pattern2["orientation_map"]
        orient_similarity = self._calculate_orientation_similarity(orient1, orient2)
        similarities.append(orient_similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_orientation_similarity(self, orient1: List[List[float]], 
                                        orient2: List[List[float]]) -> float:
        """
        Calculate similarity between orientation maps
        """
        if not orient1 or not orient2:
            return 0.0
        
        # Convert to numpy arrays
        arr1 = np.array(orient1)
        arr2 = np.array(orient2)
        
        # Ensure same size
        min_size = min(arr1.shape[0], arr2.shape[0], arr1.shape[1], arr2.shape[1])
        arr1 = arr1[:min_size, :min_size]
        arr2 = arr2[:min_size, :min_size]
        
        # Calculate correlation
        try:
            correlation = np.corrcoef(arr1.flatten(), arr2.flatten())[0, 1]
            return max(0, correlation) if not np.isnan(correlation) else 0.0
        except:
            return 0.0
    
    def _calculate_quality_similarity(self, quality1: Dict, quality2: Dict) -> float:
        """
        Calculate similarity based on quality metrics
        """
        similarities = []
        
        for metric in ["clarity", "coverage", "contrast", "overall_score"]:
            if metric in quality1 and metric in quality2:
                val1, val2 = quality1[metric], quality2[metric]
                similarity = 1 - abs(val1 - val2)
                similarities.append(similarity)
        
        return np.mean(similarities) if similarities else 0.0
    
    def _calculate_sensor_compatibility(self, sensor1: str, sensor2: str,
                                      device1: Dict, device2: Dict) -> float:
        """
        Calculate compatibility between different sensors and devices
        """
        # Same sensor type gets higher score
        sensor_similarity = 1.0 if sensor1 == sensor2 else 0.7
        
        # Same device model gets higher score
        device_similarity = 1.0 if device1.get("model") == device2.get("model") else 0.8
        
        return (sensor_similarity + device_similarity) / 2
    
    def authenticate_fingerprint(self, finger_name: str, 
                               test_data: PhoneFingerprintData) -> PhoneMatchResult:
        """
        Authenticate a fingerprint against stored data
        """
        # Load reference data for the specific finger
        reference_data = self.scanner.load_fingerprint_data(finger_name)
        
        if not reference_data:
            return PhoneMatchResult(
                is_match=False,
                similarity_score=0.0,
                matched_finger=None,
                confidence=0.0,
                match_details={"error": "No reference data found"},
                processing_time=0.0
            )
        
        # Match against the reference
        return self.match_fingerprint(test_data, {finger_name: reference_data})
    
    def verify_all_fingers(self) -> Dict[str, PhoneMatchResult]:
        """
        Verify all registered fingers by re-scanning them
        """
        results = {}
        
        for finger_name in self.scanner.scanned_fingers:
            print(f"Verifying {finger_name}...")
            
            # Simulate re-scanning the finger
            test_data = self.scanner.simulate_phone_scan(finger_name)
            if test_data:
                result = self.authenticate_fingerprint(finger_name, test_data)
                results[finger_name] = result
            else:
                results[finger_name] = PhoneMatchResult(
                    is_match=False,
                    similarity_score=0.0,
                    matched_finger=None,
                    confidence=0.0,
                    match_details={"error": "Failed to re-scan"},
                    processing_time=0.0
                )
        
        return results
    
    def get_matching_statistics(self) -> Dict:
        """
        Get statistics about the matching system
        """
        if not self.scanner.fingerprint_data:
            return {"error": "No fingerprint data available"}
        
        # Calculate self-similarity scores
        self_similarities = []
        cross_similarities = []
        
        finger_names = list(self.scanner.fingerprint_data.keys())
        
        for i, finger1 in enumerate(finger_names):
            data1 = self.scanner.fingerprint_data[finger1]
            
            # Self-similarity (should be high)
            self_sim = self.calculate_similarity(data1, data1)
            self_similarities.append(self_sim)
            
            # Cross-similarity with other fingers (should be low)
            for j, finger2 in enumerate(finger_names):
                if i != j:
                    data2 = self.scanner.fingerprint_data[finger2]
                    cross_sim = self.calculate_similarity(data1, data2)
                    cross_similarities.append(cross_sim)
        
        return {
            "total_fingers": len(finger_names),
            "average_self_similarity": np.mean(self_similarities) if self_similarities else 0.0,
            "average_cross_similarity": np.mean(cross_similarities) if cross_similarities else 0.0,
            "discrimination_ratio": (np.mean(self_similarities) / np.mean(cross_similarities) 
                                   if cross_similarities and np.mean(cross_similarities) > 0 
                                   else float('inf')),
            "threshold": self.threshold,
            "finger_names": finger_names
        } 