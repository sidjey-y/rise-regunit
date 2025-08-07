import yaml
import os
from typing import Dict, Any, Optional
import logging

class ConfigManager:
    
    _instance = None
    _config = None
    
    def __new__(cls, config_path: str = "config.yaml"):
        if cls._instance is None:
            cls._instance = super(ConfigManager, cls).__new__(cls)
            cls._instance._load_config(config_path)
        return cls._instance
    
    def _load_config(self, config_path: str) -> None:
        try:
            if not os.path.exists(config_path):
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            with open(config_path, 'r', encoding='utf-8') as file:
                self._config = yaml.safe_load(file)
            
            self._setup_logging()
            
        except Exception as e:
            print(f"Error loading configuration: {e}")
            raise
    
    def _setup_logging(self) -> None:
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
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_camera_config(self) -> Dict[str, Any]:
        return self._config.get('camera', {})
    
    def get_face_detection_config(self) -> Dict[str, Any]:
        return self._config.get('face_detection', {})
    
    def get_eye_detection_config(self) -> Dict[str, Any]:
        return self._config.get('eye_detection', {})
    
    def get_liveness_config(self) -> Dict[str, Any]:
        return self._config.get('liveness', {})
    
    def get_head_pose_config(self) -> Dict[str, Any]:
        return self._config.get('head_pose', {})
    
    def get_display_config(self) -> Dict[str, Any]:
        return self._config.get('display', {})
    
    def get_paths_config(self) -> Dict[str, Any]:
        return self._config.get('paths', {})
    
    def validate_config(self) -> bool:
        required_sections = [
            'camera', 'face_detection', 'eye_detection', 
            'liveness', 'head_pose', 'display', 'paths'
        ]
        
        for section in required_sections:
            if section not in self._config:
                print(f"Missing required configuration section: {section}")
                return False
        
        return True
    
    def reload_config(self, config_path: str = "config.yaml") -> None:
        self._load_config(config_path)
    
    @property
    def config(self) -> Dict[str, Any]:
        return self._config.copy() 