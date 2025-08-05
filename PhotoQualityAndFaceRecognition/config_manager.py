import yaml
import os
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    """
    Configuration Manager class to handle YAML configuration files.
    Follows Singleton pattern to ensure single configuration instance.
    """
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, config_path: str) -> None:
        """Load configuration from YAML file"""
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            
            # Setup logging based on config
            self._setup_logging()
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
    
    def _setup_logging(self) -> None:
        """Setup logging configuration"""
        log_config = self._config.get('logging', {})
        
        logging.basicConfig(
            level=getattr(logging, log_config.get('level', 'INFO')),
            format=log_config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            handlers=[
                logging.FileHandler(log_config.get('file', 'face_recognition.log')),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation (e.g., 'camera.default_index')"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_camera_config(self) -> Dict[str, Any]:
        """Get camera configuration"""
        return self._config.get('camera', {})
    
    def get_face_detection_config(self) -> Dict[str, Any]:
        """Get face detection configuration"""
        return self._config.get('face_detection', {})
    
    def get_eye_detection_config(self) -> Dict[str, Any]:
        """Get eye detection configuration"""
        return self._config.get('eye_detection', {})
    
    def get_liveness_config(self) -> Dict[str, Any]:
        """Get liveness detection configuration"""
        return self._config.get('liveness', {})
    
    def get_display_config(self) -> Dict[str, Any]:
        """Get display configuration"""
        return self._config.get('display', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        """Get paths configuration"""
        return self._config.get('paths', {})
    
    def validate_config(self) -> bool:
        """Validate that all required configuration sections exist"""
        required_sections = [
            'camera', 'face_detection', 'eye_detection', 
            'liveness', 'display', 'paths'
        ]
        
        for section in required_sections:
            if section not in self._config:
                print(f"Missing required configuration section: {section}")
                return False
        
        return True
    
    def reload_config(self, config_path: str = "config.yaml") -> None:
        """Reload configuration from file"""
        self._load_config(config_path)
    
    @property
    def config(self) -> Dict[str, Any]:
        """Get the entire configuration dictionary"""
        return self._config.copy() 