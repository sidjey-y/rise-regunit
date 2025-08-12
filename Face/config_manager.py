import yaml
import os
from typing import Dict, Any, Optional, Union
import logging
from functools import lru_cache

class ConfigManager:
    
    _instance = None
    _config = None
    _is_valid = None
    _config_hash = None
    
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
            
            # Reset validation cache when config changes
            self._is_valid = None
            self._config_hash = hash(str(self._config))
            
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
        """Get configuration value with dot notation support and caching."""
        if '.' not in key:
            return self._config.get(key, default)
        
        return self._get_nested_value(key, default)
    
    @lru_cache(maxsize=128)
    def _get_nested_value(self, key: str, default: Any = None) -> Any:
        """Cached method for getting nested configuration values."""
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
    
    def get_compliance_config(self) -> Dict[str, Any]:
        return self._config.get('compliance', {})
    
    def validate_config(self) -> bool:
        """Validate configuration with caching for performance."""
        if self._is_valid is not None:
            return self._is_valid
        
        required_sections = [
            'camera', 'face_detection', 'eye_detection', 
            'liveness', 'head_pose', 'display', 'paths'
        ]
        
        self._is_valid = all(section in self._config for section in required_sections)
        
        if not self._is_valid:
            missing_sections = [section for section in required_sections if section not in self._config]
            print(f"Missing required configuration sections: {missing_sections}")
        
        return self._is_valid
    
    def reload_config(self, config_path: str = "config.yaml") -> None:
        """Reload configuration and clear caches."""
        old_hash = self._config_hash
        self._load_config(config_path)
        
        # Clear LRU cache when config changes
        self._get_nested_value.cache_clear()
        
        if old_hash != self._config_hash:
            print("Configuration reloaded successfully")
        else:
            print("Configuration unchanged")
    
    def has_changed(self) -> bool:
        """Check if configuration has changed since last load."""
        return self._config_hash != hash(str(self._config))
    
    def get_config_info(self) -> Dict[str, Any]:
        """Get information about the current configuration."""
        return {
            'sections': list(self._config.keys()),
            'is_valid': self.validate_config(),
            'config_hash': self._config_hash,
            'file_size': os.path.getsize(self._config_path) if hasattr(self, '_config_path') else None
        }
    
    @property
    def config(self) -> Dict[str, Any]:
        """Return configuration with shallow copy for safety."""
        return self._config.copy()  # Shallow copy is much faster than deep copy
    
    @property
    def config_path(self) -> str:
        """Get the current configuration file path."""
        return getattr(self, '_config_path', 'config.yaml')
    
    def __contains__(self, key: str) -> bool:
        """Support 'in' operator for checking if key exists."""
        return self.get(key) is not None
    
    def __len__(self) -> int:
        """Return number of top-level configuration sections."""
        return len(self._config)
    
    def __str__(self) -> str:
        """String representation of the configuration manager."""
        return f"ConfigManager(config_path='{self.config_path}', sections={list(self._config.keys())})" 