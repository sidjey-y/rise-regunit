from abc import ABC, abstractmethod
from typing import Tuple, List, Any, Optional
import cv2
import numpy as np
import logging
from config_manager import ConfigManager

class BaseDetector(ABC):
    def __init__(self, config_manager: Optional[ConfigManager] = None):

        self.config_manager = config_manager or ConfigManager()
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False
        
    def initialize(self) -> bool:
        try:
            self._initialize_components()
            self._initialized = True
            self.logger.info(f"{self.__class__.__name__} initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to initialize {self.__class__.__name__}: {e}")
            return False
    
    @abstractmethod
    def _initialize_components(self) -> None:
        pass
    
    @abstractmethod
    def detect(self, frame: np.ndarray) -> Any:
        pass
    
    def is_initialized(self) -> bool:
        return self._initialized
    
    def validate_frame(self, frame: np.ndarray) -> bool:
        if frame is None:
            self.logger.warning("Input frame is None")
            return False
        
        if not isinstance(frame, np.ndarray):
            self.logger.warning("Input frame is not a numpy array")
            return False
        
        if len(frame.shape) != 3:
            self.logger.warning("Input frame must be 3-dimensional (height, width, channels)")
            return False
        
        if frame.shape[2] != 3:
            self.logger.warning("Input frame must have 3 channels (BGR)")
            return False        
        return True
    
    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        return frame.copy()
    
    def postprocess_results(self, results: Any) -> Any:
        return results
    
    def get_status(self) -> dict:
        return {
            'class_name': self.__class__.__name__,
            'initialized': self._initialized,
            'config_loaded': self.config_manager is not None
        }
    
    def cleanup(self) -> None:
        self._initialized = False
        self.logger.info(f"{self.__class__.__name__} cleaned up")
    
    def __enter__(self):
        if not self.initialize():
            raise RuntimeError(f"Failed to initialize {self.__class__.__name__}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup() 


#template method pattern and abstract factory pattern
#common interface for all detectors
