import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np
from typing import Tuple, List, Dict, Any, Optional
from base_processor import BaseProcessor
from config_manager import ConfigManager
import logging
import os

class SiameseNetwork(BaseProcessor):
    """
    Siamese Neural Network for fingerprint uniqueness checking.
    Compares two fingerprints and determines if they belong to the same person.
    """
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        super().__init__(config_manager)
        
        # Initialize attributes
        self.model = None
        self.embedding_model = None
        self.embedding_dim = None
        self.margin = None
        self.learning_rate = None
        self.batch_size = None
        self.epochs = None
        self.model_path = None
        
    def _initialize_components(self) -> None:
        """Initialize Siamese network components"""
        siamese_config = self.config_manager.get_siamese_config()
        
        # Set configuration parameters
        self.embedding_dim = siamese_config.get('embedding_dim', 128)
        self.margin = siamese_config.get('margin', 1.0)
        self.learning_rate = siamese_config.get('learning_rate', 0.001)
        self.batch_size = siamese_config.get('batch_size', 32)
        self.epochs = siamese_config.get('epochs', 100)
        self.model_path = siamese_config.get('model_save_path', 'models/siamese_fingerprint_model.h5')
        
        # Create model directory if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
        self.logger.info("Siamese network components initialized")
    
    def build_model(self) -> None:
        """Build the Siamese neural network architecture"""
        
        # Base CNN for feature extraction
        def create_base_network():
            """Create the base CNN network for feature extraction"""
            model = keras.Sequential([
                # First convolutional block
                layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Second convolutional block
                layers.Conv2D(64, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Third convolutional block
                layers.Conv2D(128, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Fourth convolutional block
                layers.Conv2D(256, (3, 3), activation='relu'),
                layers.BatchNormalization(),
                layers.MaxPooling2D((2, 2)),
                layers.Dropout(0.25),
                
                # Flatten and dense layers
                layers.Flatten(),
                layers.Dense(512, activation='relu'),
                layers.BatchNormalization(),
                layers.Dropout(0.5),
                layers.Dense(self.embedding_dim, activation=None)  # No activation for embedding
            ])
            return model
        
        # Create base network
        base_network = create_base_network()
        
        # Create Siamese network
        input_a = layers.Input(shape=(224, 224, 3))
        input_b = layers.Input(shape=(224, 224, 3))
        
        # Get embeddings
        embedding_a = base_network(input_a)
        embedding_b = base_network(input_b)
        
        # Normalize embeddings
        embedding_a = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding_a)
        embedding_b = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(embedding_b)
        
        # Calculate distance
        distance = layers.Lambda(lambda x: tf.reduce_sum(tf.square(x[0] - x[1]), axis=1, keepdims=True))([embedding_a, embedding_b])
        
        # Create Siamese model
        self.model = models.Model(inputs=[input_a, input_b], outputs=distance)
        
        # Create embedding model for inference
        self.embedding_model = models.Model(inputs=input_a, outputs=embedding_a)
        
        # Compile model with contrastive loss
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss=self._contrastive_loss,
            metrics=['accuracy']
        )
        
        self.logger.info("Siamese network model built successfully")
    
    def _contrastive_loss(self, y_true: tf.Tensor, y_pred: tf.Tensor) -> tf.Tensor:
        """
        Contrastive loss function for Siamese network.
        
        Args:
            y_true: True labels (0 for same person, 1 for different person)
            y_pred: Predicted distances
            
        Returns:
            tf.Tensor: Contrastive loss
        """
        square_pred = tf.square(y_pred)
        margin_square = tf.square(tf.maximum(self.margin - y_pred, 0))
        return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)
    
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process fingerprint pair for similarity comparison.
        
        Args:
            data: Dictionary containing 'fingerprint1' and 'fingerprint2' images
            
        Returns:
            dict: Similarity score and prediction
        """
        if not self._initialized or self.model is None:
            self.logger.error("Model not initialized")
            return {}
        
        fingerprint1 = data.get('fingerprint1')
        fingerprint2 = data.get('fingerprint2')
        
        if fingerprint1 is None or fingerprint2 is None:
            self.logger.error("Both fingerprints are required")
            return {}
        
        # Ensure correct input shape
        if len(fingerprint1.shape) == 3 and fingerprint1.shape[0] == 1:
            fingerprint1 = fingerprint1[0]
        if len(fingerprint2.shape) == 3 and fingerprint2.shape[0] == 1:
            fingerprint2 = fingerprint2[0]
        
        # Add batch dimension
        fingerprint1 = np.expand_dims(fingerprint1, axis=0)
        fingerprint2 = np.expand_dims(fingerprint2, axis=0)
        
        # Predict distance
        distance = self.model.predict([fingerprint1, fingerprint2], verbose=0)
        
        # Calculate similarity score (1 - normalized distance)
        similarity_score = 1.0 - (distance[0][0] / (self.margin * 2))
        similarity_score = max(0.0, min(1.0, similarity_score))
        
        # Determine if same person (threshold-based)
        threshold = self.config_manager.get('database.similarity_threshold', 0.85)
        is_same_person = similarity_score >= threshold
        
        results = {
            'distance': float(distance[0][0]),
            'similarity_score': float(similarity_score),
            'is_same_person': bool(is_same_person),
            'threshold': threshold,
            'confidence': self._calculate_confidence(similarity_score, threshold)
        }
        
        return self.postprocess_results(results)
    
    def extract_embedding(self, fingerprint: np.ndarray) -> np.ndarray:
        """
        Extract embedding vector from a single fingerprint.
        
        Args:
            fingerprint: Input fingerprint image
            
        Returns:
            np.ndarray: Embedding vector
        """
        if not self._initialized or self.embedding_model is None:
            self.logger.error("Embedding model not initialized")
            return None
        
        # Handle different input shapes
        if len(fingerprint.shape) == 4:
            # Already has batch dimension, check if it's (1, 1, H, W, C)
            if fingerprint.shape[0] == 1 and fingerprint.shape[1] == 1:
                fingerprint = fingerprint[0, 0]  # Remove extra dimensions
            elif fingerprint.shape[0] == 1:
                fingerprint = fingerprint[0]  # Remove batch dimension
        elif len(fingerprint.shape) == 3:
            # Standard (H, W, C) format
            pass
        else:
            self.logger.error(f"Unexpected fingerprint shape: {fingerprint.shape}")
            return None
        
        # Ensure we have (H, W, C) format
        if len(fingerprint.shape) != 3:
            self.logger.error(f"Invalid fingerprint shape after processing: {fingerprint.shape}")
            return None
        
        # Add batch dimension for model input
        fingerprint = np.expand_dims(fingerprint, axis=0)
        
        # Extract embedding
        embedding = self.embedding_model.predict(fingerprint, verbose=0)
        
        return embedding[0]
    
    def calculate_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Calculate similarity between two embedding vectors.
        
        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Normalize embeddings
        embedding1_norm = embedding1 / np.linalg.norm(embedding1)
        embedding2_norm = embedding2 / np.linalg.norm(embedding2)
        
        # Calculate cosine similarity
        similarity = np.dot(embedding1_norm, embedding2_norm)
        
        return max(0.0, min(1.0, similarity))
    
    def _calculate_confidence(self, similarity_score: float, threshold: float) -> float:
        """
        Calculate confidence score based on similarity and threshold.
        
        Args:
            similarity_score: Similarity score
            threshold: Decision threshold
            
        Returns:
            float: Confidence score between 0 and 1
        """
        # Higher confidence when similarity is far from threshold
        distance_from_threshold = abs(similarity_score - threshold)
        confidence = min(1.0, distance_from_threshold / (1.0 - threshold))
        
        return confidence
    
    def train(self, train_data: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> Dict[str, Any]:
        """
        Train the Siamese network.
        
        Args:
            train_data: Tuple of (fingerprint1_batch, fingerprint2_batch, labels)
            
        Returns:
            dict: Training history and metrics
        """
        if not self._initialized or self.model is None:
            self.logger.error("Model not initialized")
            return {}
        
        fingerprint1_batch, fingerprint2_batch, labels = train_data
        
        # Training callbacks
        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=self.config_manager.get('siamese_network.early_stopping_patience', 10),
                restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=5,
                min_lr=1e-7
            ),
            keras.callbacks.ModelCheckpoint(
                self.model_path,
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=False
            )
        ]
        
        # Train the model
        history = self.model.fit(
            [fingerprint1_batch, fingerprint2_batch],
            labels,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_split=0.2,
            callbacks=callbacks,
            verbose=1
        )
        
        # Save final model
        self.model.save(self.model_path)
        
        self.logger.info(f"Training completed. Model saved to {self.model_path}")
        
        return {
            'history': history.history,
            'final_loss': history.history['loss'][-1],
            'final_accuracy': history.history['accuracy'][-1],
            'model_path': self.model_path
        }
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        Load pre-trained model.
        
        Args:
            model_path: Path to model file. If None, uses default path.
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            path = model_path or self.model_path
            
            if not os.path.exists(path):
                self.logger.warning(f"Model file not found: {path}")
                return False
            
            # Load the model
            self.model = keras.models.load_model(
                path,
                custom_objects={'contrastive_loss': self._contrastive_loss}
            )
            
            # Recreate embedding model
            self.embedding_model = models.Model(
                inputs=self.model.input[0],
                outputs=self.model.layers[-2].output
            )
            
            self.logger.info(f"Model loaded successfully from {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            return False
    
    def save_model(self, model_path: Optional[str] = None) -> bool:
        """
        Save the trained model.
        
        Args:
            model_path: Path to save model. If None, uses default path.
            
        Returns:
            bool: True if model saved successfully
        """
        try:
            path = model_path or self.model_path
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Save the model
            self.model.save(path)
            
            self.logger.info(f"Model saved successfully to {path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error saving model: {e}")
            return False
    
    def get_model_summary(self) -> str:
        """
        Get model architecture summary.
        
        Returns:
            str: Model summary
        """
        if self.model is None:
            return "Model not initialized"
        
        # Capture model summary
        summary_list = []
        self.model.summary(print_fn=lambda x: summary_list.append(x))
        
        return '\n'.join(summary_list) 