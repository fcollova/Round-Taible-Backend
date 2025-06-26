"""
Configuration Manager Module

Handles environment detection and configuration loading for the backend.
Supports automatic dev/prod environment detection based on Render platform variables.
"""

import os
import configparser
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing."""
    pass


def is_render_environment() -> bool:
    """
    Detect if running in Render production environment.
    
    Returns:
        bool: True if running on Render platform, False otherwise
    """
    return (
        os.getenv('RENDER') == 'true' or
        os.getenv('RENDER_SERVICE_ID') is not None or
        os.getenv('RENDER_SERVICE_NAME') is not None
    )


def detect_environment() -> str:
    """
    Automatically detect the current environment.
    
    Returns:
        str: 'prod' if on Render, 'dev' otherwise
    """
    if is_render_environment():
        return 'prod'
    else:
        return 'dev'


class ConfigManager:
    """
    Centralized configuration manager that handles environment detection
    and configuration loading from appropriate config files.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize ConfigManager.
        
        Args:
            config_dir: Directory containing config files. Defaults to current directory.
        """
        self.config_dir = Path(config_dir or '.')
        self.environment = detect_environment()
        self.config = configparser.ConfigParser()
        self._load_config()
    
    def _load_config(self):
        """Load configuration from appropriate config file based on environment."""
        # Try environment-specific config first
        config_file = self.config_dir / f'config.{self.environment}.conf'
        
        if not config_file.exists():
            # Fallback to default config.conf
            config_file = self.config_dir / 'config.conf'
            
        if not config_file.exists():
            raise ConfigurationError(f"No configuration file found. Looked for: config.{self.environment}.conf and config.conf")
        
        try:
            self.config.read(config_file)
            self.config_file_path = config_file
            # Note: Cannot use logging here as it may not be set up yet
            print(f"ConfigManager: Loaded configuration from: {config_file}")
        except Exception as e:
            raise ConfigurationError(f"Failed to load configuration from {config_file}: {e}")
    
    def get_environment(self) -> str:
        """Get the current environment."""
        return self.environment
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.environment == 'prod'
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.environment == 'dev'
    
    # Server Configuration
    def get_server_host(self) -> str:
        """Get server host."""
        return self.config.get('server', 'host', fallback='0.0.0.0')
    
    def get_server_port(self) -> int:
        """Get server port."""
        return self.config.getint('server', 'port', fallback=8000)
    
    def get_server_reload(self) -> bool:
        """Get server reload setting."""
        return self.config.getboolean('server', 'reload', fallback=False)
    
    # Logging Configuration
    def get_logging_debug(self) -> bool:
        """Get logging debug setting."""
        return self.config.getboolean('logging', 'debug', fallback=False)
    
    def get_logging_level(self) -> str:
        """Get logging level."""
        return self.config.get('logging', 'level', fallback='INFO')
    
    def get_logging_console_output(self) -> bool:
        """Get console output setting."""
        return self.config.getboolean('logging', 'console_output', fallback=True)
    
    def get_logging_file_output(self) -> bool:
        """Get file output setting."""
        return self.config.getboolean('logging', 'file_output', fallback=True)
    
    def get_logging_file_path(self) -> str:
        """Get log file path."""
        return self.config.get('logging', 'log_file', fallback='./logs/backend.log')
    
    # OpenRouter Configuration
    def get_openrouter_api_key(self) -> str:
        """Get OpenRouter API key."""
        api_key = self.config.get('openrouter', 'api_key', fallback='')
        if not api_key:
            raise ConfigurationError("OpenRouter API key not configured")
        return api_key
    
    def get_openrouter_base_url(self) -> str:
        """Get OpenRouter base URL."""
        return self.config.get('openrouter', 'base_url', fallback='https://openrouter.ai/api/v1')
    
    def get_openrouter_timeout(self) -> int:
        """Get OpenRouter timeout."""
        return self.config.getint('openrouter', 'timeout', fallback=60)
    
    def get_openrouter_min_delay(self) -> float:
        """Get minimum delay between OpenRouter requests."""
        return self.config.getfloat('openrouter', 'min_delay', fallback=0.5)
    
    def get_openrouter_max_delay(self) -> float:
        """Get maximum delay between OpenRouter requests."""
        return self.config.getfloat('openrouter', 'max_delay', fallback=1.5)
    
    def get_openrouter_max_concurrent(self) -> int:
        """Get maximum concurrent OpenRouter requests."""
        return self.config.getint('openrouter', 'max_concurrent', fallback=2)
    
    def get_openrouter_free_model_delay(self) -> float:
        """Get delay for free models to prevent key bans."""
        return self.config.getfloat('openrouter', 'free_model_delay', fallback=5.0)
    
    def get_openrouter_free_model_max_errors(self) -> int:
        """Get max consecutive errors before circuit breaker activation."""
        return self.config.getint('openrouter', 'free_model_max_errors', fallback=3)
    
    def get_openrouter_free_model_cooldown(self) -> int:
        """Get cooldown period in seconds after circuit breaker activation."""
        return self.config.getint('openrouter', 'free_model_cooldown', fallback=30)
    
    # Models Configuration
    # Models are now retrieved from frontend API /api/models - no longer in config files
    
    # Frontend Configuration
    def get_frontend_url(self) -> str:
        """Get frontend URL."""
        return self.config.get('frontend', 'url', fallback='http://localhost:3000')
    
    def get_frontend_timeout(self) -> int:
        """Get frontend timeout."""
        return self.config.getint('frontend', 'timeout', fallback=10)
    
    # CORS Configuration
    def get_cors_allowed_origins(self) -> list:
        """Get CORS allowed origins."""
        origins = self.config.get('cors', 'allowed_origins', fallback='')
        return [origin.strip() for origin in origins.split(',') if origin.strip()]
    
    # Database Configuration
    def get_database_url(self) -> str:
        """Get database URL."""
        return self.config.get('database', 'database_url', fallback='')
    
    def get_all_config(self) -> Dict[str, Dict[str, Any]]:
        """Get all configuration as dictionary."""
        result = {}
        for section_name in self.config.sections():
            result[section_name] = dict(self.config.items(section_name))
        return result
    
    def validate_required_config(self):
        """Validate that all required configuration is present."""
        required_sections = ['server', 'openrouter', 'logging']
        missing_sections = []
        
        for section in required_sections:
            if not self.config.has_section(section):
                missing_sections.append(section)
        
        if missing_sections:
            raise ConfigurationError(f"Missing required configuration sections: {missing_sections}")
        
        # Validate specific required keys
        required_keys = {
            'openrouter': ['api_key']
        }
        
        missing_keys = []
        for section, keys in required_keys.items():
            for key in keys:
                if not self.config.has_option(section, key):
                    missing_keys.append(f"{section}.{key}")
        
        if missing_keys:
            raise ConfigurationError(f"Missing required configuration keys: {missing_keys}")


# Global config manager instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """
    Get the global configuration manager instance.
    
    Returns:
        ConfigManager: The global configuration manager
    """
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
        _config_manager.validate_required_config()
    return _config_manager


def reload_config():
    """Reload the configuration from files."""
    global _config_manager
    _config_manager = None
    return get_config()


# Convenience functions for common config access
def get_environment() -> str:
    """Get current environment."""
    return get_config().get_environment()


def is_production() -> bool:
    """Check if running in production."""
    return get_config().is_production()


def is_development() -> bool:
    """Check if running in development."""
    return get_config().is_development()