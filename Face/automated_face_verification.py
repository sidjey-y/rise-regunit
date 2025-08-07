#!/usr/bin/env python3

import cv2
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional, Tuple
from deepface import DeepFace
from datetime import datetime

class AutomatedFaceVerification:
    def __init__(self, 
                 approved_dir: str = "aproved_img",
                 raw_pic_dir: str = "raw_pic",
                 model_name: str = "VGG-Face",
                 similarity_threshold: float = 0.6):

        self.approved_dir = approved_dir
        self.raw_pic_dir = raw_pic_dir
        self.model_name = model_name
        self.similarity_threshold = similarity_threshold
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.embedding_cache = {}
        
        self.logger.info("Automated face verification system initialized")
    
    def extract_embedding(self, image_path: str) -> Optional[np.ndarray]:

        try:
            if image_path in self.embedding_cache:
                self.logger.info(f"Using cached embedding for {image_path}")
                return self.embedding_cache[image_path]
            
            embedding_result = DeepFace.represent(
                img_path=image_path,
                model_name=self.model_name,
                enforce_detection=True
            )
            
            if embedding_result and len(embedding_result) > 0:
                embedding_array = np.array(embedding_result[0]['embedding'])
                
                self.embedding_cache[image_path] = embedding_array
                
                self.logger.info(f"Successfully extracted embedding from {image_path}")
                return embedding_array
            else:
                self.logger.warning(f"No face detected in {image_path}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error extracting embedding from {image_path}: {e}")
            return None
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:

        try:
            dot_product = np.dot(embedding1, embedding2)
            norm1 = np.linalg.norm(embedding1)
            norm2 = np.linalg.norm(embedding2)
            similarity = dot_product / (norm1 * norm2)
            
            return float(similarity)
            
        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0
    
    def load_reference_embeddings(self) -> Dict[str, np.ndarray]:

        reference_embeddings = {}
        
        if os.path.exists(self.approved_dir):
            for filename in os.listdir(self.approved_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(self.approved_dir, filename)
                    embedding = self.extract_embedding(image_path)
                    if embedding is not None:
                        reference_embeddings[f"approved_{filename}"] = embedding
        
        if os.path.exists(self.raw_pic_dir):
            for filename in os.listdir(self.raw_pic_dir):
                if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                    image_path = os.path.join(self.raw_pic_dir, filename)
                    embedding = self.extract_embedding(image_path)
                    if embedding is not None:
                        reference_embeddings[f"raw_{filename}"] = embedding
        
        self.logger.info(f"Loaded {len(reference_embeddings)} reference embeddings")
        return reference_embeddings
    
    def verify_captured_face(self, captured_image_path: str) -> Dict[str, Any]:

        results = {
            'captured_image': captured_image_path,
            'verification_passed': False,
            'best_match': None,
            'similarity_score': 0.0,
            'all_matches': [],
            'timestamp': datetime.now().isoformat(),
            'error': None
        }
        
        try:
            captured_embedding = self.extract_embedding(captured_image_path)
            if captured_embedding is None:
                results['error'] = "No face detected in captured image"
                return results
            
            reference_embeddings = self.load_reference_embeddings()
            if not reference_embeddings:
                results['error'] = "No reference embeddings found"
                return results
            
            best_similarity = 0.0
            best_match_file = None
            
            for ref_name, ref_embedding in reference_embeddings.items():
                similarity = self.calculate_similarity(captured_embedding, ref_embedding)
                
                match_info = {
                    'reference_name': ref_name,
                    'similarity_score': similarity,
                    'is_match': similarity >= self.similarity_threshold
                }
                
                results['all_matches'].append(match_info)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_match_file = ref_name
            
            results['similarity_score'] = best_similarity
            results['best_match'] = best_match_file
            results['verification_passed'] = best_similarity >= self.similarity_threshold
            
            self.logger.info(f"Face verification completed. Best match: {best_match_file} "
                           f"(similarity: {best_similarity:.3f}, passed: {results['verification_passed']})")
            
        except Exception as e:
            self.logger.error(f"Error during face verification: {e}")
            results['error'] = str(e)
        
        return results
    
    def save_verification_results(self, results: Dict[str, Any], output_path: str) -> bool:

        try:
            serializable_results = results.copy()
            if 'all_matches' in serializable_results:
                for match in serializable_results['all_matches']:
                    if 'similarity_score' in match:
                        match['similarity_score'] = float(match['similarity_score'])
            
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            self.logger.info(f"Saved verification results to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving verification results: {e}")
            return False
    
    def get_verification_summary(self, results: Dict[str, Any]) -> str:

        if results.get('error'):
            return f"❌ VERIFICATION FAILED: {results['error']}"
        
        if results['verification_passed']:
            return (f"✅ VERIFICATION PASSED\n"
                   f"   Best Match: {results['best_match']}\n"
                   f"   Similarity Score: {results['similarity_score']:.3f}\n"
                   f"   Threshold: {self.similarity_threshold}")
        else:
            return (f"❌ VERIFICATION FAILED\n"
                   f"   Best Match: {results['best_match']}\n"
                   f"   Similarity Score: {results['similarity_score']:.3f}\n"
                   f"   Threshold: {self.similarity_threshold}")

def verify_captured_face_automated(captured_image_path: str, 
                                 approved_dir: str = "aproved_img",
                                 raw_pic_dir: str = "raw_pic",
                                 similarity_threshold: float = 0.6) -> Dict[str, Any]:

    verifier = AutomatedFaceVerification(
        approved_dir=approved_dir,
        raw_pic_dir=raw_pic_dir,
        similarity_threshold=similarity_threshold
    )
    
    return verifier.verify_captured_face(captured_image_path) 