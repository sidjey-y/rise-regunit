#!/usr/bin/env python3

import numpy as np
import math
from typing import List, Dict, Any, Tuple, Optional
import logging
from collections import defaultdict
import json

class BoyerMooreMatcher:
    """
    Boyer-Moore string matching algorithm implementation for fingerprint pattern matching.
    Used for efficient substring search in minutiae patterns.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def build_bad_char_table(self, pattern: str) -> Dict[str, int]:
        """Build bad character table for Boyer-Moore algorithm"""
        bad_char = {}
        pattern_length = len(pattern)
        
        for i in range(pattern_length - 1):
            bad_char[pattern[i]] = pattern_length - 1 - i
            
        return bad_char
    
    def build_good_suffix_table(self, pattern: str) -> List[int]:
        """Build good suffix table for Boyer-Moore algorithm"""
        pattern_length = len(pattern)
        good_suffix = [0] * pattern_length
        
        # Case 1: Exact match
        good_suffix[pattern_length - 1] = 1
        
        # Case 2: Good suffix exists
        for i in range(pattern_length - 2, -1, -1):
            if pattern[i] == pattern[pattern_length - 1]:
                good_suffix[i] = pattern_length - 1 - i
            else:
                good_suffix[i] = good_suffix[i + 1]
        
        return good_suffix
    
    def search(self, text: str, pattern: str) -> List[int]:
        """
        Boyer-Moore string search algorithm
        
        Args:
            text: Text to search in
            pattern: Pattern to search for
            
        Returns:
            List of starting positions where pattern is found
        """
        if not pattern or not text:
            return []
        
        pattern_length = len(pattern)
        text_length = len(text)
        
        if pattern_length > text_length:
            return []
        
        bad_char = self.build_bad_char_table(pattern)
        good_suffix = self.build_good_suffix_table(pattern)
        
        positions = []
        i = pattern_length - 1
        
        while i < text_length:
            j = pattern_length - 1
            k = i
            
            while j >= 0 and text[k] == pattern[j]:
                j -= 1
                k -= 1
            
            if j == -1:
                positions.append(k + 1)
                i += good_suffix[0] if pattern_length > 1 else 1
            else:
                bad_char_shift = bad_char.get(text[k], pattern_length)
                good_suffix_shift = good_suffix[j] if j < pattern_length - 1 else 1
                i += max(bad_char_shift, good_suffix_shift)
        
        return positions

class Bozorth3Matcher:
    """
    Bozorth3 fingerprint matching algorithm implementation.
    Uses xytheta format for minutiae points and Boyer-Moore for pattern matching.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.boyer_moore = BoyerMooreMatcher()
        
        # Bozorth3 parameters
        self.max_distance = self.config.get('max_distance', 20.0)
        self.max_angle_diff = self.config.get('max_angle_diff', 30.0)  # degrees
        self.min_matches = self.config.get('min_matches', 12)
        self.max_score = self.config.get('max_score', 100.0)
        
    def convert_to_xytheta(self, minutiae: List[Dict[str, Any]]) -> List[Tuple[float, float, float]]:
        """
        Convert minutiae points to xytheta format
        
        Args:
            minutiae: List of minutiae dictionaries with x, y, theta keys
            
        Returns:
            List of (x, y, theta) tuples
        """
        xytheta_points = []
        
        for point in minutiae:
            x = float(point.get('x', 0))
            y = float(point.get('y', 0))
            theta = float(point.get('theta', 0))
            
            # Normalize theta to 0-360 degrees
            theta = theta % 360.0
            
            xytheta_points.append((x, y, theta))
        
        return xytheta_points
    
    def calculate_distance(self, point1: Tuple[float, float, float], 
                          point2: Tuple[float, float, float]) -> float:
        """Calculate Euclidean distance between two points"""
        x1, y1, _ = point1
        x2, y2, _ = point2
        return math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    
    def calculate_angle_difference(self, theta1: float, theta2: float) -> float:
        """Calculate the difference between two angles (0-180 degrees)"""
        diff = abs(theta1 - theta2)
        return min(diff, 360.0 - diff)
    
    def find_corresponding_points(self, template_points: List[Tuple[float, float, float]], 
                                 query_points: List[Tuple[float, float, float]]) -> List[Tuple[int, int]]:
        """
        Find corresponding minutiae points between template and query fingerprints
        
        Args:
            template_points: Template fingerprint minutiae in xytheta format
            query_points: Query fingerprint minutiae in xytheta format
            
        Returns:
            List of (template_index, query_index) pairs
        """
        correspondences = []
        
        for i, template_point in enumerate(template_points):
            best_match = None
            best_score = float('inf')
            
            for j, query_point in enumerate(query_points):
                # Calculate distance
                distance = self.calculate_distance(template_point, query_point)
                
                if distance > self.max_distance:
                    continue
                
                # Calculate angle difference
                angle_diff = self.calculate_angle_difference(template_point[2], query_point[2])
                
                if angle_diff > self.max_angle_diff:
                    continue
                
                # Calculate combined score (lower is better)
                score = distance + (angle_diff / 10.0)  # Weight angle difference less
                
                if score < best_score:
                    best_score = score
                    best_match = j
            
            if best_match is not None:
                correspondences.append((i, best_match))
        
        return correspondences
    
    def create_minutiae_pattern(self, points: List[Tuple[float, float, float]]) -> str:
        """
        Create a string pattern representation of minutiae for Boyer-Moore matching
        
        Args:
            points: List of (x, y, theta) minutiae points
            
        Returns:
            String pattern representation
        """
        if not points:
            return ""
        
        # Sort points by x coordinate for consistent ordering
        sorted_points = sorted(points, key=lambda p: p[0])
        
        # Create pattern string with discretized values
        pattern_parts = []
        for x, y, theta in sorted_points:
            # Discretize coordinates and angles
            x_disc = int(x / 10.0)  # 10-pixel bins
            y_disc = int(y / 10.0)
            theta_disc = int(theta / 15.0)  # 15-degree bins
            
            pattern_parts.append(f"{x_disc:03d}{y_disc:03d}{theta_disc:02d}")
        
        return "".join(pattern_parts)
    
    def match_fingerprints(self, template_minutiae: List[Dict[str, Any]], 
                          query_minutiae: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Match two fingerprints using Bozorth3 algorithm
        
        Args:
            template_minutiae: Template fingerprint minutiae
            query_minutiae: Query fingerprint minutiae
            
        Returns:
            Dictionary with match results
        """
        try:
            # Convert to xytheta format
            template_points = self.convert_to_xytheta(template_minutiae)
            query_points = self.convert_to_xytheta(query_minutiae)
            
            if not template_points or not query_points:
                return {
                    'match_score': 0.0,
                    'is_match': False,
                    'correspondences': [],
                    'pattern_match': False,
                    'error': 'No minutiae points found'
                }
            
            # Find corresponding points
            correspondences = self.find_corresponding_points(template_points, query_points)
            
            # Calculate match score using Bozorth3 scoring
            match_score = self._calculate_bozorth3_score(correspondences, template_points, query_points)
            
            # Create pattern strings for Boyer-Moore matching
            template_pattern = self.create_minutiae_pattern(template_points)
            query_pattern = self.create_minutiae_pattern(query_points)
            
            # Use Boyer-Moore for pattern matching
            pattern_matches = self.boyer_moore.search(template_pattern, query_pattern[:20])  # Use first 20 chars
            pattern_match = len(pattern_matches) > 0
            
            # Determine if it's a match
            is_match = match_score >= self.min_matches and pattern_match
            
            return {
                'match_score': match_score,
                'is_match': is_match,
                'correspondences': correspondences,
                'pattern_match': pattern_match,
                'template_points': len(template_points),
                'query_points': len(query_points),
                'correspondence_count': len(correspondences)
            }
            
        except Exception as e:
            self.logger.error(f"Error in fingerprint matching: {e}")
            return {
                'match_score': 0.0,
                'is_match': False,
                'correspondences': [],
                'pattern_match': False,
                'error': str(e)
            }
    
    def _calculate_bozorth3_score(self, correspondences: List[Tuple[int, int]], 
                                 template_points: List[Tuple[float, float, float]], 
                                 query_points: List[Tuple[float, float, float]]) -> float:
        """
        Calculate Bozorth3 match score based on correspondences
        
        Args:
            correspondences: List of (template_index, query_index) pairs
            template_points: Template minutiae points
            query_points: Query minutiae points
            
        Returns:
            Bozorth3 match score
        """
        if len(correspondences) < 2:
            return 0.0
        
        # Calculate pairwise distances for template and query
        template_distances = self._calculate_pairwise_distances(template_points, correspondences, 0)
        query_distances = self._calculate_pairwise_distances(query_points, correspondences, 1)
        
        # Count matching distance pairs
        matching_pairs = 0
        total_pairs = len(template_distances)
        
        for template_dist, query_dist in zip(template_distances, query_distances):
            if abs(template_dist - query_dist) <= 5.0:  # 5-pixel tolerance
                matching_pairs += 1
        
        # Calculate score based on matching pairs
        if total_pairs == 0:
            return 0.0
        
        score = (matching_pairs / total_pairs) * len(correspondences)
        return min(score, self.max_score)
    
    def _calculate_pairwise_distances(self, points: List[Tuple[float, float, float]], 
                                    correspondences: List[Tuple[int, int]], 
                                    index: int) -> List[float]:
        """Calculate pairwise distances between corresponding points"""
        distances = []
        correspondence_indices = [corr[index] for corr in correspondences]
        
        for i in range(len(correspondence_indices)):
            for j in range(i + 1, len(correspondence_indices)):
                idx1 = correspondence_indices[i]
                idx2 = correspondence_indices[j]
                distance = self.calculate_distance(points[idx1], points[idx2])
                distances.append(distance)
        
        return distances
    
    def save_xytheta_format(self, minutiae: List[Dict[str, Any]], filename: str) -> bool:
        """
        Save minutiae points in xytheta format to file
        
        Args:
            minutiae: List of minutiae dictionaries
            filename: Output filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            xytheta_points = self.convert_to_xytheta(minutiae)
            
            with open(filename, 'w') as f:
                f.write("# x y theta\n")
                for x, y, theta in xytheta_points:
                    f.write(f"{x:.2f} {y:.2f} {theta:.2f}\n")
            
            self.logger.info(f"Saved {len(xytheta_points)} points to {filename}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving xytheta format: {e}")
            return False
    
    def load_xytheta_format(self, filename: str) -> List[Tuple[float, float, float]]:
        """
        Load minutiae points from xytheta format file
        
        Args:
            filename: Input filename
            
        Returns:
            List of (x, y, theta) tuples
        """
        try:
            points = []
            
            with open(filename, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('#') or not line:
                        continue
                    
                    parts = line.split()
                    if len(parts) >= 3:
                        x = float(parts[0])
                        y = float(parts[1])
                        theta = float(parts[2])
                        points.append((x, y, theta))
            
            self.logger.info(f"Loaded {len(points)} points from {filename}")
            return points
            
        except Exception as e:
            self.logger.error(f"Error loading xytheta format: {e}")
            return [] 