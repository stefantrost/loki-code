"""
Simple LLM-based intent classifier.

This replaces the complex model strategies system with a straightforward
LLM-based intent classification that integrates cleanly with the agent system.
"""

import json
import asyncio
from typing import Dict, Any, Optional
from enum import Enum

from .types import IntentType
from ..providers.base import GenerationRequest
from ...utils.logging import get_logger


class SimpleLLMClassifier:
    """
    Simple LLM-based intent classifier that determines user intent 
    without complex strategy patterns or adapters.
    """
    
    def __init__(self, llm_provider):
        """
        Initialize with an LLM provider.
        
        Args:
            llm_provider: BaseLLMProvider instance
        """
        self.provider = llm_provider
        self.logger = get_logger(__name__)
    
    async def classify_intent(self, user_input: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Classify user intent using LLM.
        
        Args:
            user_input: User's input text
            context: Optional context information
            
        Returns:
            Dict with intent_type, confidence, entities, etc.
        """
        try:
            prompt = self._build_classification_prompt(user_input, context or {})
            
            request = GenerationRequest(
                prompt=prompt,
                model="codellama:7b",
                max_tokens=500,
                temperature=0.1
            )
            
            response = await self.provider.generate(request)
            return self._parse_classification_response(response.content, user_input)
            
        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}, using fallback")
            return self._fallback_classification(user_input)
    
    def _build_classification_prompt(self, user_input: str, context: Dict[str, Any]) -> str:
        """Build a simple, direct prompt for intent classification."""
        
        # Generate valid intent types from enum
        intent_types = "|".join([intent.value for intent in IntentType])
        
        # Build context if available
        context_str = ""
        if context.get("current_file"):
            context_str += f"Current file: {context['current_file']}\n"
        if context.get("project_path"):
            context_str += f"Project: {context['project_path']}\n"
        
        prompt = f"""Classify this user request into one of these intent types:

INTENT TYPES: {intent_types}

INTENT DEFINITIONS:
- code_generation: Creating, writing, implementing, or saving code to files (includes "write it in a file", "save to file")
- file_analysis: Reading, examining, or analyzing existing files that already exist
- search: Finding or locating files/functions
- help: Asking for explanations or assistance
- conversation: General chat or unclear requests
- system_command: Commands like help, quit, status
- debugging: Fixing errors or troubleshooting
- follow_up: References to repeating previous actions ("do that again", "same thing")

IMPORTANT: "write it in a file" or "save it to a file" = code_generation (creating/saving code)

{context_str}USER REQUEST: {user_input}

Respond with JSON:
{{
    "intent_type": "one of the intent types above",
    "confidence": 0.0-1.0,
    "entities": {{
        "files": ["any file names mentioned"],
        "languages": ["any programming languages mentioned"],
        "actions": ["key actions like create, read, write, analyze"]
    }},
    "reasoning": "brief explanation"
}}

JSON response:"""
        
        return prompt
    
    def _parse_classification_response(self, response: str, user_input: str) -> Dict[str, Any]:
        """Parse LLM response into structured classification."""
        try:
            # Try to parse JSON
            data = json.loads(response.strip())
            
            # Validate and normalize intent_type
            intent_type = data.get("intent_type", "conversation").lower()
            try:
                IntentType(intent_type)  # Validate it's a valid enum value
            except ValueError:
                intent_type = "conversation"
            
            return {
                "intent_type": intent_type,
                "confidence": float(data.get("confidence", 0.7)),
                "entities": data.get("entities", {}),
                "reasoning": data.get("reasoning", ""),
                "classification_method": "llm"
            }
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            self.logger.warning(f"Failed to parse LLM response: {e}")
            return self._fallback_classification(user_input)
    
    def _fallback_classification(self, user_input: str) -> Dict[str, Any]:
        """Simple keyword-based fallback classification."""
        text = user_input.lower()
        
        # Check for follow-up requests first
        if any(phrase in text for phrase in ["write it", "save it", "put it in", "do that", "that again"]):
            intent_type = "follow_up"
        # Simple but effective keyword detection
        elif any(word in text for word in ["create", "write", "make", "generate", "implement"]):
            intent_type = "code_generation"
        elif any(word in text for word in ["read", "show", "analyze", "examine", "look at"]):
            intent_type = "file_analysis"
        elif any(word in text for word in ["find", "search", "locate"]):
            intent_type = "search"
        elif any(word in text for word in ["help", "how", "explain", "what"]):
            intent_type = "help"
        else:
            intent_type = "conversation"
        
        # Extract basic entities
        entities = {}
        
        # Look for file extensions or file patterns
        import re
        file_patterns = re.findall(r'\b\w+\.\w+\b', user_input)
        if file_patterns:
            entities["files"] = file_patterns
        
        # Look for programming languages
        languages = []
        for lang in ["python", "javascript", "typescript", "java", "go", "rust", "c", "cpp"]:
            if lang in text:
                languages.append(lang)
        if languages:
            entities["languages"] = languages
        
        return {
            "intent_type": intent_type,
            "confidence": 0.6,  # Lower confidence for fallback
            "entities": entities,
            "reasoning": "keyword-based fallback classification",
            "classification_method": "fallback"
        }