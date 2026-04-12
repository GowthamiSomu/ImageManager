"""
Configuration management - loads settings from config.yaml and environment variables.
"""
import yaml
import os
from typing import Any, Dict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class Config:
    """Application configuration manager."""
    
    def __init__(self, config_file: str = "config.yaml"):
        """
        Load configuration from YAML file and environment variables.
        
        Args:
            config_file: Path to configuration YAML file
        """
        self.config_file = config_file
        self.config: Dict[str, Any] = {}
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file."""
        config_path = Path(self.config_file)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        logger.info(f"Configuration loaded from {config_path}")
        
        # Override with environment variables if present
        self._apply_env_overrides()
    
    def _apply_env_overrides(self):
        """Override config values with environment variables."""
        env_mappings = {
            'DB_HOST': ('database', 'host'),
            'DB_PORT': ('database', 'port'),
            'DB_NAME': ('database', 'name'),
            'DB_USER': ('database', 'user'),
            'DB_PASSWORD': ('database', 'password'),
        }
        
        for env_var, (section, key) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                if section not in self.config:
                    self.config[section] = {}
                self.config[section][key] = value
                logger.debug(f"Overriding config from env: {section}.{key}")
    
    def get(self, *keys, default=None, expected_type=None):
        """
        Get configuration value by nested keys with optional type coercion.
        
        Args:
            *keys: Nested keys to traverse (e.g., 'database', 'host')
            default: Default value if key not found
            expected_type: Optional type to coerce to (int, float, bool, str)
            
        Returns:
            Configuration value (coerced to expected_type if provided) or default
        """
        value = self.config
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        # Type coercion
        if expected_type is not None and value is not None:
            try:
                if expected_type == int:
                    return int(value)
                elif expected_type == float:
                    return float(value)
                elif expected_type == bool:
                    if isinstance(value, bool):
                        return value
                    if isinstance(value, str):
                        return value.lower() in ('true', '1', 'yes', 'on')
                    return bool(value)
                elif expected_type == str:
                    return str(value)
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to coerce {value} to {expected_type}: {e}, using default")
                return default
        
        return value
    
    def get_database_url(self) -> str:
        """
        Build PostgreSQL connection string.
        
        Returns:
            SQLAlchemy-compatible connection string
        """
        host = self.get('database', 'host')
        port = self.get('database', 'port')
        name = self.get('database', 'name')
        user = self.get('database', 'user')
        password = self.get('database', 'password')
        
        return f"postgresql://{user}:{password}@{host}:{port}/{name}"
    
    def get_input_directory(self) -> Path:
        """Get input directory as Path object."""
        return Path(self.get('paths', 'input_directory'))
    
    def get_output_directory(self) -> Path:
        """Get output directory as Path object."""
        return Path(self.get('paths', 'output_directory'))
    
    def ensure_directories(self):
        """Create necessary directories if they don't exist."""
        output_dir = self.get_output_directory()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory ensured: {output_dir}")
