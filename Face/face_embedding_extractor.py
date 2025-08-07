#!/usr/bin/env python3

import cv2
import numpy as np
import os
import json
from typing import List, Dict, Any, Optional, Tuple
from deepface import DeepFace
import logging
from datetime import datetime

class FaceEmbeddingExtractor:
    """
    Face embedding extractor using DeepFace for face recognition and comparison.
    Extracts face embeddings from images and compares them with approved images.
    """
    
    def __init__(self, model_name: str = "VGG-Face", distance_metric: str = "cosine"):
        """
        Initialize the face embedding extractor.
        
        Args:
            model_name: DeepFace model to use (VGG-Face, Facenet, OpenFace, etc.)
            distance_metric: Distance metric for comparison (cosine, euclidean, manhattan)
        """
        self.model_name = model_name
        self.distance_metric = distance_metric
        self.logger = logging.getLogger(self.__class__.__name__)
        
        # Set similarity threshold
        self.similarity_threshold = 0.6  # Lower threshold = stricter matching
        
        self.logger.info(f"Face embedding extractor initialized with {model_name} model")
    
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:
        """
        Extract face embedding from an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            np.ndarray: Face embedding vector or None if no face detected
        """
        try:
            # Detect face and extract embedding
            embedding = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                enforce_detection=True
            )
            
            if embedding:
                embedding_array = np.array(embedding)
                self.logger.info(f"Successfully extracted embedding from {image_path}")
                return embedding_array
            else:
                self.logger.warning(f"No face detected in {image_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting embedding from {image_path}: {e}")
            return None
    
    def extract_embeddings_from_directory(self, directory_path: str) -> Dict[str, np.ndarray]:
        """
        Extract embeddings from all images in a directory.
        
        Args:
            directory_path: Path to directory containing images
            
        Returns:
            dict: Dictionary mapping filename to embedding
        """
        embeddings = {}
        
        if not os.path.exists(directory_path):
            self.logger.error(f"Directory not found: {directory_path}")
            return embeddings
        
        # Supported image formats
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
        
        for filename in os.listdir(directory_path):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                image_path = os.path.join(directory_path, filename)
                embedding = self.extract_embedding(image_path)
                
                if embedding is not None:
                    embeddings[filename] = embedding
                    self.logger.info(f"Extracted embedding from {filename}")
                else:
                    self.logger.warning(f"Failed to extract embedding from {filename}")
        
        self.logger.info(f"Extracted {len(embeddings)} embeddings from {directory_path}")
        return embeddings
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two face embeddings.
        
        Args:
            embedding1: First face embedding
            embedding2: Second face embedding
            
        Returns:
            float: Similarity score (0-1, higher = more similar)
        """
        try:
            # Calculate distance using specified metric
            if self.distance_metric == "cosine":
                # Calculate cosine similarity
                dot_product = np.dot(embedding1, embedding2)
                norm1 = np.linalg.norm(embedding1)
                norm2 = np.linalg.norm(embedding2)
                similarity = dot_product / (norm1 * norm2)
            elif self.distance_metric == "euclidean":
                distance = np.linalg.norm(embedding1 - embedding2)
                similarity = 1 / (1 + distance)  # Convert distance to similarity
            elif self.distance_metric == "manhattan":
                distance = np.sum(np.abs(embedding1 - embedding2))
                similarity = 1 / (1 + distance)  # Convert distance to similarity
            else:
                raise ValueError(f"Unsupported distance metric: {self.distance_metric}")
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def compare_with_approved_images(self, test_image_path: str, 
                                   approved_dir: str = "aproved_img") -> Dict[str, Any]:
        """
        Compare a test image with all approved images.
        
        Args:
            test_image_path: Path to test image
            approved_dir: Directory containing approved images
            
        Returns:
            dict: Comparison results
        """
        results = {
            'test_image': test_image_path,
            'approved_dir': approved_dir,
            'matches': [],
            'best_match': None,
            'is_approved': False,
            'timestamp': datetime.now().isoformat()
        }
        
        try:
            # Extract embedding from test image
            test_embedding = self.extract_embedding(test_image_path)
            if test_embedding is None:
                results['error'] = "No face detected in test image"
                return results
            
            # Extract embeddings from approved images
            approved_embeddings = self.extract_embeddings_from_directory(approved_dir)
            
            if not approved_embeddings:
                results['error'] = "No approved images found or no faces detected"
                return results
            
            # Compare with each approved image
            best_similarity = 0.0
            best_match_file = None
            
            for filename, approved_embedding in approved_embeddings.items():
                similarity = self.calculate_similarity(test_embedding, approved_embedding)
                
                match_info = {
                    'filename': filename,
                    'similarity_score': similarity,
                    'is_match': similarity >= self.similarity_threshold
                }
                
                results['matches'].append(match_info)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_file = filename
            
            # Set best match and approval status
            if best_match_file:
                results['best_match'] = {
                    'filename': best_match_file,
                    'similarity_score': best_similarity
                }
                results['is_approved'] = best_similarity >= self.similarity_threshold
            
            self.logger.info(f"Comparison completed. Best match: {best_match_file} "
                           f"(similarity: {best_similarity:.3f})")
            
        except Exception as e:
            self.logger.error(f"Error during comparison: {e}")
            results['error'] = str(e)
        
        return results
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], 
                       output_path: str) -> bool:
        """
        Save embeddings to a JSON file.
        
        Args:
            embeddings: Dictionary of embeddings
            output_path: Path to save the embeddings
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # Convert numpy arrays to lists for JSON serialization
            serializable_embeddings = {}
            for filename, embedding in embeddings.items():
                serializable_embeddings[filename] = embedding.tolist()
            
            # Save to JSON file
            with open(output_path, 'w') as f:
                json.dump(serializable_embeddings, f, indent=2)
            
            self.logger.info(f"Saved {len(embeddings)} embeddings to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving embeddings: {e}")
            return False
    
    def load_embeddings(self, embeddings_path: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from a JSON file.
        
        Args:
            embeddings_path: Path to the embeddings file
            
        Returns:
            dict: Dictionary of embeddings
        """
        try:
            with open(embeddings_path, 'r') as f:
                serializable_embeddings = json.load(f)
            
            # Convert lists back to numpy arrays
            embeddings = {}
            for filename, embedding_list in serializable_embeddings.items():
                embeddings[filename] = np.array(embedding_list)
            
            self.logger.info(f"Loaded {len(embeddings)} embeddings from {embeddings_path}")
            return embeddings
            
        except Exception as e:
            self.logger.error(f"Error loading embeddings: {e}")
            return {}
    
    def verify_face_quality(self, image_path: str) -> Dict[str, Any]:
        """
        Verify face quality for embedding extraction.
        
        Args:
            image_path: Path to the image
            
        Returns:
            dict: Quality assessment results
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'error': 'Could not load image'}
            
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Use OpenCV face detector for quick check
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            quality_assessment = {
                'image_path': image_path,
                'faces_detected': len(faces),
                'image_size': img.shape,
                'is_suitable': len(faces) > 0,
                'recommendations': []
            }
            
            if len(faces) == 0:
                quality_assessment['recommendations'].append("No face detected")
            elif len(faces) > 1:
                quality_assessment['recommendations'].append("Multiple faces detected")
            
            # Check image size
            height, width = img.shape[:2]
            if width < 100 or height < 100:
                quality_assessment['recommendations'].append("Image too small")
                quality_assessment['is_suitable'] = False
            
            return quality_assessment
            
        except Exception as e:
            self.logger.error(f"Error assessing face quality: {e}")
            return {'error': str(e)} 