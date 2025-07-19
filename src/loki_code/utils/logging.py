"""
Comprehensive logging system for Loki Code.

This module provides a fully-featured logging system that integrates with the
configuration system, supports multiple output formats, and includes advanced
features like performance monitoring and sensitive data filtering.
"""

import os
import sys
import json
import time
import logging
import logging.handlers
import re
from pathlib import Path
from typing import Dict, Any, Optional, Callable, Union
from functools import wraps
from contextlib import contextmanager
from datetime import datetime

# Color codes for console output
class Colors:
    """ANSI color codes for console output."""
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    # Standard colors
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    
    # Bright colors
    BRIGHT_BLACK = '\033[90m'
    BRIGHT_RED = '\033[91m'
    BRIGHT_GREEN = '\033[92m'
    BRIGHT_YELLOW = '\033[93m'
    BRIGHT_BLUE = '\033[94m'
    BRIGHT_MAGENTA = '\033[95m'
    BRIGHT_CYAN = '\033[96m'
    BRIGHT_WHITE = '\033[97m'


class SensitiveDataFilter(logging.Filter):
    """Filter to automatically redact sensitive information from logs."""
    
    def __init__(self):
        super().__init__()
        # Patterns for sensitive data
        self.patterns = [
            # API keys and tokens
            (re.compile(r'(api[_-]?key|token|secret|password|pwd)["\s]*[:=]["\s]*([^\s"]{8,})', re.IGNORECASE), r'\1=***REDACTED***'),
            (re.compile(r'(bearer\s+)([a-zA-Z0-9._-]{20,})', re.IGNORECASE), r'\1***REDACTED***'),
            
            # Common secret patterns
            (re.compile(r'([a-zA-Z0-9._-]+_secret["\s]*[:=]["\s]*)([^\s"]{8,})', re.IGNORECASE), r'\1***REDACTED***'),
            (re.compile(r'(sk-[a-zA-Z0-9]{32,})', re.IGNORECASE), r'sk-***REDACTED***'),
            
            # Database URLs with passwords
            (re.compile(r'(://[^:]+:)([^@]+)(@)', re.IGNORECASE), r'\1***REDACTED***\3'),
            
            # Email addresses (optional - might be too aggressive)
            # (re.compile(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', re.IGNORECASE), r'***EMAIL***'),
        ]
    
    def filter(self, record):
        """Filter log record to redact sensitive information."""
        if hasattr(record, 'msg') and isinstance(record.msg, str):
            message = record.msg
            for pattern, replacement in self.patterns:
                message = pattern.sub(replacement, message)
            record.msg = message
        
        # Also filter args if present
        if hasattr(record, 'args') and record.args:
            filtered_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    for pattern, replacement in self.patterns:
                        arg = pattern.sub(replacement, arg)
                filtered_args.append(arg)
            record.args = tuple(filtered_args)
        
        return True


class ColoredConsoleFormatter(logging.Formatter):
    """Formatter that adds colors to console output based on log level."""
    
    # Color mapping for different log levels
    LEVEL_COLORS = {
        'DEBUG': Colors.BRIGHT_BLACK,
        'INFO': Colors.BRIGHT_BLUE,
        'WARNING': Colors.BRIGHT_YELLOW,
        'ERROR': Colors.BRIGHT_RED,
        'CRITICAL': Colors.BRIGHT_MAGENTA + Colors.BOLD,
    }
    
    def __init__(self, use_colors=True):
        """Initialize the formatter.
        
        Args:
            use_colors: Whether to use colors in output
        """
        self.use_colors = use_colors and self._supports_color()
        
        # Console format: timestamp | level | module | message
        fmt = '%(asctime)s | %(levelname)-8s | %(name)s | %(message)s'
        super().__init__(fmt, datefmt='%Y-%m-%d %H:%M:%S')
    
    def _supports_color(self):
        """Check if the terminal supports color output."""
        # Check if we're in a terminal and not being piped
        if not hasattr(sys.stdout, 'isatty') or not sys.stdout.isatty():
            return False
        
        # Check environment variables
        if os.getenv('NO_COLOR'):
            return False
        
        if os.getenv('FORCE_COLOR'):
            return True
            
        # Check if we're in a known terminal that supports colors
        term = os.getenv('TERM', '').lower()
        return 'color' in term or term in ('xterm', 'xterm-256color', 'screen', 'linux')
    
    def format(self, record):
        """Format the log record with colors if enabled."""
        # Get the basic formatted message
        formatted = super().format(record)
        
        if not self.use_colors:
            return formatted
        
        # Apply colors based on log level
        level_name = record.levelname
        color = self.LEVEL_COLORS.get(level_name, '')
        
        if color:
            # Color the entire line
            formatted = f"{color}{formatted}{Colors.RESET}"
        
        return formatted


class JSONFileFormatter(logging.Formatter):
    """Formatter that outputs structured JSON logs for file storage."""
    
    def format(self, record):
        """Format the log record as JSON."""
        # Build the log entry
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'module': record.name,
            'message': record.getMessage(),
            'extra': {}
        }
        
        # Add extra fields if present
        extra_fields = ['filename', 'lineno', 'funcName', 'process', 'thread']
        for field in extra_fields:
            if hasattr(record, field):
                log_entry['extra'][field] = getattr(record, field)
        
        # Add exception information if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add any custom fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'exc_info', 'exc_text', 'stack_info',
                          'lineno', 'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process', 'getMessage',
                          'message']:
                log_entry['extra'][key] = value
        
        return json.dumps(log_entry, default=str)


class PerformanceTimer:
    """Context manager and decorator for performance timing."""
    
    def __init__(self, logger: logging.Logger, operation: str, level: int = logging.DEBUG):
        """Initialize the performance timer.
        
        Args:
            logger: Logger instance to use
            operation: Description of the operation being timed
            level: Log level to use for timing messages
        """
        self.logger = logger
        self.operation = operation
        self.level = level
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        """Start timing."""
        self.start_time = time.perf_counter()
        self.logger.log(self.level, f"Starting {self.operation}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """End timing and log results."""
        self.end_time = time.perf_counter()
        duration = self.end_time - self.start_time
        
        if exc_type is None:
            self.logger.log(self.level, f"Completed {self.operation} in {duration:.3f}s")
        else:
            self.logger.log(self.level, f"Failed {self.operation} after {duration:.3f}s")
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration if timing is complete."""
        if self.start_time is not None and self.end_time is not None:
            return self.end_time - self.start_time
        return None


def performance_timer(operation: str = None, level: int = logging.DEBUG):
    """Decorator for timing function execution.
    
    Args:
        operation: Description of the operation (defaults to function name)
        level: Log level to use for timing messages
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            logger = get_logger(func.__module__)
            op_name = operation or f"{func.__name__}()"
            
            with PerformanceTimer(logger, op_name, level):
                return func(*args, **kwargs)
        
        return wrapper
    return decorator


class LoggingManager:
    """Central logging manager for Loki Code."""
    
    def __init__(self):
        """Initialize the logging manager."""
        self._initialized = False
        self._loggers: Dict[str, logging.Logger] = {}
        self._config = None
        self._log_dir = None
    
    def setup_logging(self, config, verbose: bool = False, force_reinit: bool = False):
        """Setup logging based on configuration.
        
        Args:
            config: LokiCodeConfig instance
            verbose: Enable verbose logging (overrides config)
            force_reinit: Force reinitialization even if already setup
        """
        if self._initialized and not force_reinit:
            return
        
        self._config = config
        
        # Determine log level
        if verbose or config.app.verbose_logging:
            log_level = logging.DEBUG
        else:
            log_level = getattr(logging, config.app.log_level.value.upper(), logging.INFO)
        
        # Setup log directory
        self._setup_log_directory(config)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(log_level)
        
        # Remove existing handlers to avoid duplicates
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        # Setup console handler
        self._setup_console_handler(root_logger, log_level)
        
        # Setup file handler if configured
        if hasattr(config.app, 'log_file') and config.app.log_file:
            self._setup_file_handler(root_logger, config, log_level)
        
        # Configure module-specific loggers
        self._setup_module_loggers(config)
        
        # Add sensitive data filter to all handlers
        sensitive_filter = SensitiveDataFilter()
        for handler in root_logger.handlers:
            handler.addFilter(sensitive_filter)
        
        self._initialized = True
        
        # Log initialization
        logger = self.get_logger('loki_code.logging')
        logger.info("Logging system initialized")
        logger.debug(f"Log level: {logging.getLevelName(log_level)}")
        logger.debug(f"Log directory: {self._log_dir}")
    
    def _setup_log_directory(self, config):
        """Setup the log directory."""
        try:
            if hasattr(config.app, 'log_file'):
                log_file_path = Path(config.app.log_file)
                self._log_dir = log_file_path.parent
            else:
                self._log_dir = Path("logs")
            
            # Create directory if it doesn't exist
            self._log_dir.mkdir(parents=True, exist_ok=True)
            
        except Exception as e:
            # Fallback to current directory
            self._log_dir = Path(".")
            print(f"Warning: Could not create log directory: {e}", file=sys.stderr)
    
    def _setup_console_handler(self, root_logger: logging.Logger, log_level: int):
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(log_level)
        
        # Use colored formatter for console
        console_formatter = ColoredConsoleFormatter(use_colors=True)
        console_handler.setFormatter(console_formatter)
        
        root_logger.addHandler(console_handler)
    
    def _setup_file_handler(self, root_logger: logging.Logger, config, log_level: int):
        """Setup file logging handler with rotation."""
        try:
            log_file = self._log_dir / "loki-code.log"
            
            # Use rotating file handler
            max_bytes = getattr(config.app, 'max_log_size_mb', 10) * 1024 * 1024
            backup_count = getattr(config.app, 'backup_count', 5)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=max_bytes,
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(log_level)
            
            # Use JSON formatter for file output
            file_formatter = JSONFileFormatter()
            file_handler.setFormatter(file_formatter)
            
            root_logger.addHandler(file_handler)
            
        except Exception as e:
            print(f"Warning: Could not setup file logging: {e}", file=sys.stderr)
    
    def _setup_module_loggers(self, config):
        """Setup module-specific loggers based on configuration."""
        # Use default module logger configuration since it was moved to app config
        default_loggers = {
            "loki_code.core": "INFO",
            "loki_code.llm": "INFO",
            "loki_code.tools": "INFO",
            "loki_code.ui": "WARNING",
            "loki_code.config": "INFO",
        }
        
        # Always enable LangChain loggers for agent reasoning visibility
        langchain_loggers = {
            "langchain": "INFO",
            "langchain_core": "INFO", 
            "langchain_community": "INFO",
            "langchain.agents": "INFO",
            "langchain.agents.agent": "INFO",
            "langchain_core.agents": "INFO"
        }
        
        # Combine all loggers
        all_loggers = {**default_loggers, **langchain_loggers}
        
        for module_name, level_name in all_loggers.items():
                logger = logging.getLogger(module_name)
                level = getattr(logging, level_name.upper(), logging.INFO)
                logger.setLevel(level)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger instance for the given name.
        
        Args:
            name: Logger name (typically __name__)
            
        Returns:
            Logger instance
        """
        if name not in self._loggers:
            self._loggers[name] = logging.getLogger(name)
        
        return self._loggers[name]
    
    def create_performance_timer(self, operation: str, level: int = logging.DEBUG) -> PerformanceTimer:
        """Create a performance timer context manager.
        
        Args:
            operation: Description of the operation being timed
            level: Log level to use for timing messages
            
        Returns:
            PerformanceTimer instance
        """
        logger = self.get_logger('loki_code.performance')
        return PerformanceTimer(logger, operation, level)
    
    def is_initialized(self) -> bool:
        """Check if logging has been initialized."""
        return self._initialized


# Global logging manager instance
_logging_manager = LoggingManager()


def setup_logging(config, verbose: bool = False, force_reinit: bool = False):
    """Setup logging based on configuration.
    
    Args:
        config: LokiCodeConfig instance
        verbose: Enable verbose logging
        force_reinit: Force reinitialization
    """
    _logging_manager.setup_logging(config, verbose, force_reinit)


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for the given name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Logger instance
    """
    return _logging_manager.get_logger(name)


@contextmanager
def log_performance(operation: str, level: int = logging.DEBUG):
    """Context manager for performance timing.
    
    Args:
        operation: Description of the operation being timed
        level: Log level to use for timing messages
        
    Yields:
        PerformanceTimer instance
    """
    timer = _logging_manager.create_performance_timer(operation, level)
    with timer:
        yield timer


def is_logging_initialized() -> bool:
    """Check if logging has been initialized."""
    return _logging_manager.is_initialized()


# Convenience functions for common logging operations
def log_startup(config_path: Optional[str] = None):
    """Log application startup information."""
    logger = get_logger('loki_code.startup')
    logger.info("🎭 Loki Code starting up...")
    
    if config_path:
        logger.info(f"📁 Configuration loaded from: {config_path}")
    else:
        logger.info("📁 Using default configuration")


def log_config_info(config):
    """Log configuration information at startup."""
    logger = get_logger('loki_code.config')
    
    logger.debug(f"App debug mode: {config.app.debug}")
    logger.debug(f"Log level: {config.app.log_level}")
    logger.debug(f"LLM provider: {config.llm.provider}")
    logger.debug(f"LLM model: {config.llm.model}")
    logger.debug(f"Tools auto-discover: {config.tools.auto_discover_builtin}")
    
    if config.app.verbose_logging:
        logger.debug("Verbose logging enabled")


def log_shutdown():
    """Log application shutdown."""
    logger = get_logger('loki_code.shutdown')
    logger.info("👋 Loki Code shutting down...")