"""
Exceptions for LLM client communication.
"""


class LLMClientError(Exception):
    """Base exception for LLM client errors."""
    pass


class LLMConnectionError(LLMClientError):
    """Exception raised when unable to connect to LLM server."""
    pass


class LLMServerError(LLMClientError):
    """Exception raised when LLM server returns an error."""
    
    def __init__(self, message: str, status_code: int = None):
        super().__init__(message)
        self.status_code = status_code


class LLMTimeoutError(LLMClientError):
    """Exception raised when LLM server request times out."""
    pass


class LLMValidationError(LLMClientError):
    """Exception raised when request validation fails."""
    pass