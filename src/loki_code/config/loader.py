"""
Configuration loading system for Loki Code.

This module handles loading, merging, and validating configuration from
multiple sources including YAML files, environment variables, and CLI arguments.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any, Union
import yaml
from pydantic import ValidationError
from dotenv import load_dotenv

from .models import LokiCodeConfig


class ConfigurationError(Exception):
    """Raised when configuration loading or validation fails."""
    pass


class ConfigLoader:
    """
    Configuration loader that supports multiple sources and validation.
    
    Loading priority (highest to lowest):
    1. Environment variables (LOKI_*)
    2. CLI-specified config file
    3. Environment-specific config (e.g., development.yaml)
    4. Default configuration file
    5. Built-in defaults (from Pydantic models)
    """
    
    def __init__(self):
        """Initialize the configuration loader."""
        self._config: Optional[LokiCodeConfig] = None
        self._config_path: Optional[Path] = None
        
        # Load environment variables from .env file if it exists
        env_file = Path(".env")
        if env_file.exists():
            load_dotenv(env_file)
    
    def load_config(self, config_path: Optional[Union[str, Path]] = None) -> LokiCodeConfig:
        """
        Load configuration from multiple sources and validate it.
        
        Args:
            config_path: Optional path to a specific config file
            
        Returns:
            Validated LokiCodeConfig instance
            
        Raises:
            ConfigurationError: If configuration loading or validation fails
        """
        try:
            # Start with empty config dict
            config_data = {}
            
            # 1. Load default configuration
            default_config_path = self._find_default_config()
            if default_config_path:
                default_data = self._load_yaml_file(default_config_path)
                config_data = self._deep_merge(config_data, default_data)
            
            # 2. Load environment-specific configuration
            env_config_path = self._find_environment_config()
            if env_config_path and env_config_path != default_config_path:
                env_data = self._load_yaml_file(env_config_path)
                config_data = self._deep_merge(config_data, env_data)
            
            # 3. Load CLI-specified configuration (highest priority)
            if config_path:
                cli_config_path = Path(config_path)
                if not cli_config_path.exists():
                    raise ConfigurationError(f"Specified config file not found: {config_path}")
                
                cli_data = self._load_yaml_file(cli_config_path)
                config_data = self._deep_merge(config_data, cli_data)
                self._config_path = cli_config_path
            
            # 4. Apply environment variable overrides
            config_data = self._apply_env_overrides(config_data)
            
            # 5. Validate the final configuration
            self._config = LokiCodeConfig(**config_data)
            
            return self._config
            
        except ValidationError as e:
            raise ConfigurationError(f"Configuration validation failed: {self._format_validation_error(e)}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration: {str(e)}")
    
    def get_config(self) -> LokiCodeConfig:
        """
        Get the current configuration, loading it if necessary.
        
        Returns:
            Current LokiCodeConfig instance
        """
        if self._config is None:
            self._config = self.load_config()
        return self._config
    
    def reload_config(self, config_path: Optional[Union[str, Path]] = None) -> LokiCodeConfig:
        """
        Reload configuration from sources.
        
        Args:
            config_path: Optional path to a specific config file
            
        Returns:
            Newly loaded LokiCodeConfig instance
        """
        self._config = None
        return self.load_config(config_path)
    
    def _find_default_config(self) -> Optional[Path]:
        """Find the default configuration file."""
        possible_paths = [
            Path("configs/default.yaml"),
            Path("configs/default.yml"),
            Path("config/default.yaml"),
            Path("config/default.yml"),
            Path("default.yaml"),
            Path("default.yml"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _find_environment_config(self) -> Optional[Path]:
        """Find environment-specific configuration file."""
        # Try to detect environment from various sources
        env = (
            os.getenv("LOKI_ENV") or 
            os.getenv("ENVIRONMENT") or
            os.getenv("ENV") or
            "development" if os.getenv("DEBUG") else None
        )
        
        if not env:
            return None
        
        possible_paths = [
            Path(f"configs/{env}.yaml"),
            Path(f"configs/{env}.yml"),
            Path(f"config/{env}.yaml"),
            Path(f"config/{env}.yml"),
            Path(f"{env}.yaml"),
            Path(f"{env}.yml"),
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
        
        return None
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load and parse a YAML configuration file.
        
        Args:
            file_path: Path to the YAML file
            
        Returns:
            Parsed configuration data
            
        Raises:
            ConfigurationError: If file cannot be loaded or parsed
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
                
            if data is None:
                return {}
                
            if not isinstance(data, dict):
                raise ConfigurationError(f"Configuration file {file_path} must contain a YAML object (dictionary)")
                
            return data
            
        except yaml.YAMLError as e:
            raise ConfigurationError(f"Invalid YAML in {file_path}: {str(e)}")
        except IOError as e:
            raise ConfigurationError(f"Cannot read configuration file {file_path}: {str(e)}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Deep merge two configuration dictionaries.
        
        Args:
            base: Base configuration dictionary
            override: Override configuration dictionary
            
        Returns:
            Merged configuration dictionary
        """
        result = base.copy()
        
        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = value
        
        return result
    
    def _apply_env_overrides(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply environment variable overrides to configuration.
        
        Environment variables with LOKI_ prefix override config values.
        Example: LOKI_APP_DEBUG=true overrides app.debug
        
        Args:
            config_data: Configuration data to override
            
        Returns:
            Configuration data with environment overrides applied
        """
        result = config_data.copy()
        
        # Get all environment variables that start with LOKI_
        loki_env_vars = {k: v for k, v in os.environ.items() if k.startswith('LOKI_')}
        
        for env_key, env_value in loki_env_vars.items():
            # Convert LOKI_APP_DEBUG to ['app', 'debug']
            config_path = env_key[5:].lower().split('_')  # Remove 'LOKI_' prefix
            
            # Convert string values to appropriate types
            typed_value = self._convert_env_value(env_value)
            
            # Set the value in the config dict
            self._set_nested_value(result, config_path, typed_value)
        
        return result
    
    def _convert_env_value(self, value: str) -> Any:
        """
        Convert environment variable string to appropriate Python type.
        
        Args:
            value: String value from environment variable
            
        Returns:
            Converted value (bool, int, float, or string)
        """
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False
        
        # Handle numeric values
        try:
            # Try integer first
            if '.' not in value:
                return int(value)
            else:
                return float(value)
        except ValueError:
            pass
        
        # Handle lists (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]
        
        # Return as string
        return value
    
    def _set_nested_value(self, data: Dict[str, Any], path: list, value: Any) -> None:
        """
        Set a value in a nested dictionary using a path.
        
        Args:
            data: Dictionary to modify
            path: List of keys representing the path
            value: Value to set
        """
        current = data
        
        # Navigate to the parent of the target key
        for key in path[:-1]:
            if key not in current:
                current[key] = {}
            elif not isinstance(current[key], dict):
                # Can't navigate further, skip this override
                return
            current = current[key]
        
        # Set the final value
        if path:
            current[path[-1]] = value
    
    def _format_validation_error(self, error: ValidationError) -> str:
        """
        Format a Pydantic validation error for user-friendly display.
        
        Args:
            error: Pydantic ValidationError
            
        Returns:
            Formatted error message
        """
        messages = []
        for err in error.errors():
            location = " -> ".join(str(loc) for loc in err['loc'])
            message = err['msg']
            value = err.get('input', 'N/A')
            messages.append(f"  {location}: {message} (got: {value})")
        
        return "Validation errors:\n" + "\n".join(messages)


# Global configuration loader instance
_config_loader = ConfigLoader()


def load_config(config_path: Optional[Union[str, Path]] = None) -> LokiCodeConfig:
    """
    Load configuration from multiple sources.
    
    Args:
        config_path: Optional path to a specific config file
        
    Returns:
        Validated LokiCodeConfig instance
        
    Raises:
        ConfigurationError: If configuration loading fails
    """
    return _config_loader.load_config(config_path)


def get_config() -> LokiCodeConfig:
    """
    Get the current configuration, loading it if necessary.
    
    Returns:
        Current LokiCodeConfig instance
    """
    return _config_loader.get_config()


def reload_config(config_path: Optional[Union[str, Path]] = None) -> LokiCodeConfig:
    """
    Reload configuration from sources.
    
    Args:
        config_path: Optional path to a specific config file
        
    Returns:
        Newly loaded LokiCodeConfig instance
    """
    return _config_loader.reload_config(config_path)


def validate_config_file(config_path: Union[str, Path]) -> tuple[bool, Optional[str]]:
    """
    Validate a configuration file without loading it globally.
    
    Args:
        config_path: Path to configuration file to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        temp_loader = ConfigLoader()
        temp_loader.load_config(config_path)
        return True, None
    except ConfigurationError as e:
        return False, str(e)