import yaml
import os
from typing import Dict, Any, Optional, List
import logging

class ConfigManager:
    """
    Configuration Manager for Fingerprint Uniqueness Checker.
    Handles YAML configuration loading and validation.
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
                logging.FileHandler(log_config.get('file', 'fingerprint_checker.log')),
                logging.StreamHandler()
            ]
        )
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self._config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_dataset_config(self) -> Dict[str, Any]:
        """Get dataset configuration"""
        return self._config.get('dataset', {})
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration"""
        return self._config.get('preprocessing', {})
    
    def get_siamese_config(self) -> Dict[str, Any]:
        """Get Siamese network configuration"""
        return self._config.get('siamese_network', {})
    
    def get_feature_config(self) -> Dict[str, Any]:
        """Get feature extraction configuration"""
        return self._config.get('feature_extraction', {})
    
    def get_database_config(self) -> Dict[str, Any]:
        """Get database configuration"""
        return self._config.get('database', {})
    
    def get_verification_config(self) -> Dict[str, Any]:
        """Get verification configuration"""
        return self._config.get('verification', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration"""
        return self._config.get('performance', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self._config.get('output', {})
    
    def validate_config(self) -> bool:
        """Validate that all required configuration sections exist"""
        required_sections = [
            'dataset', 'preprocessing', 'siamese_network', 
            'feature_extraction', 'database', 'verification',
            'performance', 'output'
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