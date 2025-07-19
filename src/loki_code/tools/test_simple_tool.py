"""
Simple test tool with only one parameter to verify LangChain parsing works.
"""

from typing import Type
from pydantic import BaseModel, Field
from .langchain_tools import LangChainToolBase


class SimpleTestArgsSchema(BaseModel):
    """Input schema for SimpleTestTool with only one parameter."""
    message: str = Field(description="A simple message to echo back")


class SimpleTestTool(LangChainToolBase):
    """Simple test tool with one parameter to verify LangChain parsing."""
    
    name: str = "simple_test"
    description: str = "Echo back a simple message for testing"
    args_schema: Type[BaseModel] = SimpleTestArgsSchema
    
    
    def _run(
        self,
        message: str,
        run_manager=None,
    ) -> str:
        """Echo back the message."""
        try:
            return f"Echo: {message}"
        except Exception as e:
            return f"Error: {str(e)}"