#!/usr/bin/env python3

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any
import logging

class MinutiaeExtractor:
    """
    Extract minutiae points (x, y, theta) from fingerprint images.
    Minutiae are the key features used in fingerprint matching.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
        
    def extract_minutiae(self, image: np.ndarray) -> List[Dict[str, Any]]:
        try:
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Enhance image
            enhanced = self._enhance_image(gray)
            
            # Extract minutiae
            minutiae = self._detect_minutiae(enhanced)
            
            self.logger.info(f"Extracted {len(minutiae)} minutiae points")
            return minutiae
            
        except Exception as e:
            self.logger.error(f"Error extracting minutiae: {e}")
            return []
    
    def _enhance_image(self, image: np.ndarray) -> np.ndarray:
        """Enhance fingerprint image for better minutiae detection"""
        # Apply histogram equalization
        enhanced = cv2.equalizeHist(image)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        return enhanced
    
    def _detect_minutiae(self, image: np.ndarray) -> List[Dict[str, Any]]:
        minutiae = []
        
        # Apply ridge detection
        ridges = self._detect_ridges(image)
        
        # Find ridge endings and bifurcations
        endings = self._find_ridge_endings(ridges)
        bifurcations = self._find_bifurcations(ridges)
        
        # Combine and filter minutiae
        all_minutiae = endings + bifurcations
        filtered_minutiae = self._filter_minutiae(all_minutiae, image.shape)
        
        for point in filtered_minutiae:
            minutiae.append({
                'x': int(point['x']),
                'y': int(point['y']),
                'theta': point['theta'],
                'type': point['type'],
                'quality': point['quality']
            })
        
        return minutiae
    
    def _detect_ridges(self, image: np.ndarray) -> np.ndarray:
        """Detect ridge patterns in the fingerprint"""
        # Apply Gabor filter for ridge detection
        kernel_size = 15
        sigma = 2.0
        theta = np.pi/4
        lambda_val = 10.0
        gamma = 0.5
        psi = 0
        
        kernel = cv2.getGaborKernel((kernel_size, kernel_size), sigma, theta, lambda_val, gamma, psi, ktype=cv2.CV_32F)
        ridges = cv2.filter2D(image, cv2.CV_8UC3, kernel)
        
        # Binarize the result
        _, binary = cv2.threshold(ridges, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        return binary
    
    def _find_ridge_endings(self, ridges: np.ndarray) -> List[Dict[str, Any]]:
        """Find ridge ending points"""
        endings = []
        
        # Use skeletonization to find endpoints
        skeleton = self._skeletonize(ridges)
        
        # Find endpoints (pixels with only one neighbor)
        for y in range(1, skeleton.shape[0] - 1):
            for x in range(1, skeleton.shape[1] - 1):
                if skeleton[y, x] > 0:
                    # Count neighbors
                    neighbors = self._count_neighbors(skeleton, x, y)
                    if neighbors == 1:  # Endpoint
                        theta = self._calculate_orientation(skeleton, x, y)
                        endings.append({
                            'x': x,
                            'y': y,
                            'theta': theta,
                            'type': 'ending',
                            'quality': 1.0
                        })
        
        return endings
    
    def _find_bifurcations(self, ridges: np.ndarray) -> List[Dict[str, Any]]:
        """Find bifurcation points"""
        bifurcations = []
        
        # Use skeletonization to find bifurcations
        skeleton = self._skeletonize(ridges)
        
        # Find bifurcations (pixels with three neighbors)
        for y in range(1, skeleton.shape[0] - 1):
            for x in range(1, skeleton.shape[1] - 1):
                if skeleton[y, x] > 0:
                    # Count neighbors
                    neighbors = self._count_neighbors(skeleton, x, y)
                    if neighbors == 3:  # Bifurcation
                        theta = self._calculate_orientation(skeleton, x, y)
                        bifurcations.append({
                            'x': x,
                            'y': y,
                            'theta': theta,
                            'type': 'bifurcation',
                            'quality': 1.0
                        })
        
        return bifurcations
    
    def _skeletonize(self, image: np.ndarray) -> np.ndarray:
        """Skeletonize the binary image"""
        # Simple skeletonization using morphological operations
        kernel = np.ones((3, 3), np.uint8)
        skeleton = np.zeros_like(image)
        
        max_iterations = 100  # Prevent infinite loops
        iteration_count = 0
        prev_pixel_count = cv2.countNonZero(image)
        
        while iteration_count < max_iterations:
            # Erode
            eroded = cv2.erode(image, kernel)
            # Open
            opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, kernel)
            # Subtract
            temp = cv2.subtract(eroded, opened)
            # Union
            skeleton = cv2.bitwise_or(skeleton, temp)
            # Update image
            image = eroded.copy()
            
            current_pixel_count = cv2.countNonZero(image)
            
            # Break if no pixels left or no change in pixel count for safety
            if current_pixel_count == 0 or current_pixel_count == prev_pixel_count:
                break
                
            prev_pixel_count = current_pixel_count
            iteration_count += 1
            
        if iteration_count >= max_iterations:
            self.logger.warning(f"Skeletonization reached maximum iterations ({max_iterations})")
        
        return skeleton
    
    def _count_neighbors(self, image: np.ndarray, x: int, y: int) -> int:
        """Count the number of neighbors for a pixel"""
        count = 0
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                if image[y + dy, x + dx] > 0:
                    count += 1
        return count
    
    def _calculate_orientation(self, image: np.ndarray, x: int, y: int) -> float:
        """Calculate the orientation of a minutiae point"""
        # Simple orientation calculation based on gradient
        try:
            # Calculate gradient
            gx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
            gy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate orientation
            orientation = np.arctan2(gy[y, x], gx[y, x])
            return float(orientation)
        except:
            return 0.0
    
    def _filter_minutiae(self, minutiae: List[Dict[str, Any]], image_shape: Tuple[int, int]) -> List[Dict[str, Any]]:
        """Filter minutiae points based on quality and distance"""
        if not minutiae:
            return []
        
        # Remove minutiae too close to edges
        margin = 10
        filtered = []
        for point in minutiae:
            x, y = point['x'], point['y']
            if margin <= x < image_shape[1] - margin and margin <= y < image_shape[0] - margin:
                filtered.append(point)
        
        # Remove minutiae too close to each other
        final_minutiae = []
        min_distance = 10
        
        for point in filtered:
            too_close = False
            for existing in final_minutiae:
                distance = np.sqrt((point['x'] - existing['x'])**2 + (point['y'] - existing['y'])**2)
                if distance < min_distance:
                    too_close = True
                    break
            
            if not too_close:
                final_minutiae.append(point)
        
        return final_minutiae
    
    def save_minutiae_to_file(self, minutiae: List[Dict[str, Any]], filename: str) -> bool:
        """Save minutiae points to a file in (x,y,theta) format"""
        try:
            with open(filename, 'w') as f:
                f.write("# Minutiae points in (x,y,theta) format\n")
                f.write("# x y theta type quality\n")
                for point in minutiae:
                    f.write(f"{point['x']} {point['y']} {point['theta']:.4f} {point['type']} {point['quality']:.2f}\n")
            return True
        except Exception as e:
            self.logger.error(f"Error saving minutiae to file: {e}")
            return False
    
    def load_minutiae_from_file(self, filename: str) -> List[Dict[str, Any]]:
        """Load minutiae points from a file"""
        minutiae = []
        try:
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 5:
                        minutiae.append({
                            'x': int(parts[0]),
                            'y': int(parts[1]),
                            'theta': float(parts[2]),
                            'type': parts[3],
                            'quality': float(parts[4])
                        })
            return minutiae
        except Exception as e:
            self.logger.error(f"Error loading minutiae from file: {e}")
            return []
    
    def compare_minutiae(self, minutiae1: List[Dict[str, Any]], minutiae2: List[Dict[str, Any]]) -> float:
        """Compare two sets of minutiae points and return similarity score"""
        if not minutiae1 or not minutiae2:
            return 0.0
        
        matches = 0
        total_points = min(len(minutiae1), len(minutiae2))
        
        # Simple matching based on distance and orientation
        for point1 in minutiae1:
            for point2 in minutiae2:
                # Calculate distance
                distance = np.sqrt((point1['x'] - point2['x'])**2 + (point1['y'] - point2['y'])**2)
                
                # Calculate orientation difference
                angle_diff = abs(point1['theta'] - point2['theta'])
                angle_diff = min(angle_diff, 2*np.pi - angle_diff)
                
                # Check if points match
                if distance < 10 and angle_diff < np.pi/6:  # 30 degrees
                    matches += 1
                    break
        
        return matches / total_points if total_points > 0 else 0.0 