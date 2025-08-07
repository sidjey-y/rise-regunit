import cv2
import numpy as np
from typing import Tuple, Optional, Dict, Any
from base_processor import BaseProcessor
from config_manager import ConfigManager
import logging

class FingerprintPreprocessor(BaseProcessor):
    """
    Fingerprint image preprocessor that handles enhancement, normalization,
    and preparation for feature extraction and AI model input.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        super().__init__(config_manager)
        
        # Initialize attributes
        self.target_size = None
        self.normalization_enabled = False
        self.augmentation_enabled = False
        self.augmentation_params = {}
        
    def _initialize_components(self) -> None:
        """Initialize preprocessing components"""
        preprocess_config = self.config_manager.get_preprocessing_config()
        
        # Set target size
        self.target_size = tuple(preprocess_config.get('target_size', [224, 224]))
        
        # Set normalization flag
        self.normalization_enabled = preprocess_config.get('normalization', True)
        
        # Set augmentation parameters
        self.augmentation_enabled = preprocess_config.get('augmentation', True)
        self.augmentation_params = {
            'rotation_range': preprocess_config.get('rotation_range', 15),
            'zoom_range': preprocess_config.get('zoom_range', 0.1),
            'brightness_range': preprocess_config.get('brightness_range', 0.2),
            'contrast_range': preprocess_config.get('contrast_range', 0.2)
        }
        
        self.logger.info("Fingerprint preprocessor components initialized")
    
    def process(self, image: np.ndarray) -> Dict[str, Any]:
        """
        Process fingerprint image through complete preprocessing pipeline.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            dict: Processed image and metadata
        """
        if not self.validate_image(image):
            return {}
        
        # Preprocess image
        processed_image = self.preprocess_image(image)
        
        # Apply enhancement
        enhanced_image = self._enhance_fingerprint(processed_image)
        
        # Resize to target size
        resized_image = self._resize_image(enhanced_image)
        
        # Normalize if enabled
        if self.normalization_enabled:
            normalized_image = self._normalize_image(resized_image)
        else:
            normalized_image = resized_image
        
        # Prepare for AI model input
        model_input = self._prepare_model_input(normalized_image)
        
        results = {
            'original_image': image,
            'processed_image': processed_image,
            'enhanced_image': enhanced_image,
            'resized_image': resized_image,
            'normalized_image': normalized_image,
            'model_input': model_input,
            'metadata': {
                'original_size': image.shape,
                'target_size': self.target_size,
                'normalization_applied': self.normalization_enabled
            }
        }
        
        return self.postprocess_results(results)
    
    def _enhance_fingerprint(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance fingerprint image quality for better feature extraction.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            np.ndarray: Enhanced fingerprint image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply histogram equalization for better contrast
        enhanced = cv2.equalizeHist(gray)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(enhanced, (3, 3), 0)
        
        # Apply morphological operations to enhance ridge structure
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        enhanced = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
        
        # Apply unsharp masking for edge enhancement
        gaussian = cv2.GaussianBlur(enhanced, (0, 0), 2.0)
        enhanced = cv2.addWeighted(enhanced, 1.5, gaussian, -0.5, 0)
        
        return enhanced
    
    def _resize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Resize image to target size while maintaining aspect ratio.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Resized image
        """
        # Calculate aspect ratio
        h, w = image.shape[:2]
        target_h, target_w = self.target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Create padded image with target size
        padded = np.zeros(self.target_size, dtype=np.uint8)
        
        # Calculate padding
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        
        # Place resized image in center
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized
        
        return padded
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image pixel values to [0, 1] range.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Normalized image
        """
        # Convert to float and normalize
        normalized = image.astype(np.float32) / 255.0
        
        return normalized
    
    def _prepare_model_input(self, image: np.ndarray) -> np.ndarray:
        """
        Prepare image for AI model input.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Model-ready input
        """
        # Ensure 3-channel input for CNN
        if len(image.shape) == 2:
            # Convert grayscale to 3-channel
            model_input = np.stack([image] * 3, axis=-1)
        else:
            model_input = image.copy()
        
        # Add batch dimension
        model_input = np.expand_dims(model_input, axis=0)
        
        return model_input
    
    def augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply data augmentation to image.
        
        Args:
            image: Input image
            
        Returns:
            np.ndarray: Augmented image
        """
        if not self.augmentation_enabled:
            return image
        
        augmented = image.copy()
        
        # Random rotation
        if self.augmentation_params['rotation_range'] > 0:
            angle = np.random.uniform(-self.augmentation_params['rotation_range'], 
                                    self.augmentation_params['rotation_range'])
            h, w = augmented.shape[:2]
            center = (w // 2, h // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            augmented = cv2.warpAffine(augmented, rotation_matrix, (w, h))
        
        # Random zoom
        if self.augmentation_params['zoom_range'] > 0:
            scale = np.random.uniform(1 - self.augmentation_params['zoom_range'], 
                                    1 + self.augmentation_params['zoom_range'])
            h, w = augmented.shape[:2]
            new_h, new_w = int(h * scale), int(w * scale)
            augmented = cv2.resize(augmented, (new_w, new_h))
            augmented = cv2.resize(augmented, (w, h))
        
        # Random brightness adjustment
        if self.augmentation_params['brightness_range'] > 0:
            brightness_factor = np.random.uniform(1 - self.augmentation_params['brightness_range'],
                                                1 + self.augmentation_params['brightness_range'])
            augmented = cv2.convertScaleAbs(augmented, alpha=brightness_factor, beta=0)
        
        # Random contrast adjustment
        if self.augmentation_params['contrast_range'] > 0:
            contrast_factor = np.random.uniform(1 - self.augmentation_params['contrast_range'],
                                              1 + self.augmentation_params['contrast_range'])
            augmented = cv2.convertScaleAbs(augmented, alpha=contrast_factor, beta=0)
        
        return augmented
    
    def extract_roi(self, image: np.ndarray) -> np.ndarray:
        """
        Extract region of interest (ROI) from fingerprint image.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            np.ndarray: ROI image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply threshold to find fingerprint region
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            # Find largest contour (assumed to be the fingerprint)
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(largest_contour)
            
            # Extract ROI with some padding
            padding = 10
            x_start = max(0, x - padding)
            y_start = max(0, y - padding)
            x_end = min(image.shape[1], x + w + padding)
            y_end = min(image.shape[0], y + h + padding)
            
            roi = image[y_start:y_end, x_start:x_end]
            return roi
        
        return image
    
    def get_quality_score(self, image: np.ndarray) -> float:
        """
        Calculate quality score for fingerprint image.
        
        Args:
            image: Input fingerprint image
            
        Returns:
            float: Quality score between 0 and 1
        """
        if not self.validate_image(image):
            return 0.0
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calculate various quality metrics
        
        # 1. Contrast score
        contrast = np.std(gray)
        contrast_score = min(contrast / 50.0, 1.0)
        
        # 2. Sharpness score (using Laplacian variance)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        sharpness = np.var(laplacian)
        sharpness_score = min(sharpness / 1000.0, 1.0)
        
        # 3. Ridge clarity score (using gradient magnitude)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        ridge_score = np.mean(gradient_magnitude) / 255.0
        
        # 4. Noise score (using high-frequency content)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
        noise = cv2.filter2D(gray, cv2.CV_64F, kernel)
        noise_score = 1.0 - min(np.std(noise) / 50.0, 1.0)
        
        # Combine scores
        quality_score = (contrast_score + sharpness_score + ridge_score + noise_score) / 4.0
        
        return max(0.0, min(1.0, quality_score)) 