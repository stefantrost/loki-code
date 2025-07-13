"""
Request analysis component for the Loki Code agent.

Handles understanding and analyzing user requests.
"""

import json
from typing import Dict, Any

from .types import RequestUnderstanding, RequestContext
from ...utils.logging import get_logger


class RequestAnalyzer:
    """Analyzes and understands user requests."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def analyze_request(self, user_message: str, context: RequestContext) -> RequestUnderstanding:
        """
        Analyze a user request to understand intent and requirements.
        
        This is a simplified implementation that would be enhanced with
        actual NLP/LLM analysis in a full implementation.
        """
        self.logger.debug(f"Analyzing request: {user_message[:100]}...")
        
        # Simple intent detection based on keywords
        intent = self._detect_intent(user_message)
        confidence = self._calculate_confidence(user_message, intent)
        
        # Extract entities and requirements
        entities = self._extract_entities(user_message, context)
        required_tools = self._identify_required_tools(user_message, intent)
        
        # Assess risk level
        risk_assessment = self._assess_risk(user_message, intent, entities)
        
        # Generate suggested approach
        suggested_approach = self._suggest_approach(intent, entities)
        
        understanding = RequestUnderstanding(
            user_intent=intent,
            confidence=confidence,
            extracted_entities=entities,
            required_tools=required_tools,
            risk_assessment=risk_assessment,
            suggested_approach=suggested_approach
        )
        
        self.logger.info(f"Request analysis complete. Intent: {intent}, Confidence: {confidence:.2f}")
        return understanding
    
    def _detect_intent(self, user_message: str) -> str:
        """Detect the primary intent of the user message."""
        message_lower = user_message.lower()
        
        # File operations
        if any(word in message_lower for word in ["read", "open", "show", "display"]):
            return "read_file"
        elif any(word in message_lower for word in ["write", "create", "save", "edit"]):
            return "write_file"
        elif any(word in message_lower for word in ["analyze", "review", "explain", "understand"]):
            return "analyze_code"
        elif any(word in message_lower for word in ["find", "search", "locate"]):
            return "search"
        elif any(word in message_lower for word in ["help", "how", "what", "explain"]):
            return "question"
        else:
            return "general_assistance"
    
    def _calculate_confidence(self, user_message: str, intent: str) -> float:
        """Calculate confidence level for the detected intent."""
        # Simple confidence calculation based on keyword matches
        message_lower = user_message.lower()
        
        confidence_keywords = {
            "read_file": ["read", "open", "show", "display", "file"],
            "write_file": ["write", "create", "save", "edit", "file"],
            "analyze_code": ["analyze", "review", "explain", "code", "function"],
            "search": ["find", "search", "locate", "look"],
            "question": ["help", "how", "what", "why", "explain"],
        }
        
        if intent in confidence_keywords:
            matches = sum(1 for keyword in confidence_keywords[intent] 
                         if keyword in message_lower)
            confidence = min(0.3 + (matches * 0.2), 1.0)
        else:
            confidence = 0.5
        
        return confidence
    
    def _extract_entities(self, user_message: str, context: RequestContext) -> Dict[str, Any]:
        """Extract relevant entities from the user message."""
        entities = {}
        
        # Extract file paths (simple pattern matching)
        import re
        file_patterns = [
            r'[\w\-\_\.\/]+\.py',
            r'[\w\-\_\.\/]+\.js',
            r'[\w\-\_\.\/]+\.ts',
            r'[\w\-\_\.\/]+\.rs',
            r'[\w\-\_\.\/]+\.txt',
            r'[\w\-\_\.\/]+\.md'
        ]
        
        files = []
        for pattern in file_patterns:
            files.extend(re.findall(pattern, user_message))
        
        if files:
            entities["files"] = files
        
        # Extract function/class names (simple patterns)
        func_pattern = r'\b[a-zA-Z_][a-zA-Z0-9_]*\(\)'
        class_pattern = r'\bclass\s+([a-zA-Z_][a-zA-Z0-9_]*)'
        
        functions = re.findall(func_pattern, user_message)
        classes = re.findall(class_pattern, user_message)
        
        if functions:
            entities["functions"] = functions
        if classes:
            entities["classes"] = classes
        
        # Add context information
        if context.project_path:
            entities["project_path"] = context.project_path
        if context.current_file:
            entities["current_file"] = context.current_file
        
        return entities
    
    def _identify_required_tools(self, user_message: str, intent: str) -> list[str]:
        """Identify which tools will be needed to fulfill the request."""
        tools = []
        
        intent_tool_mapping = {
            "read_file": ["file_reader"],
            "write_file": ["file_writer"],
            "analyze_code": ["file_reader", "code_analyzer"],
            "search": ["file_finder", "code_search"],
            "question": [],  # Usually just LLM generation
            "general_assistance": []
        }
        
        tools.extend(intent_tool_mapping.get(intent, []))
        
        # Add additional tools based on content
        message_lower = user_message.lower()
        if "git" in message_lower:
            tools.append("git_tools")
        if "test" in message_lower:
            tools.append("test_runner")
        if "format" in message_lower:
            tools.append("code_formatter")
        
        return list(set(tools))  # Remove duplicates
    
    def _assess_risk(self, user_message: str, intent: str, entities: Dict[str, Any]) -> str:
        """Assess the risk level of the request."""
        message_lower = user_message.lower()
        
        # High risk patterns
        high_risk_patterns = [
            "delete", "remove", "rm ", "sudo", "admin", "system",
            "password", "secret", "key", "token"
        ]
        
        if any(pattern in message_lower for pattern in high_risk_patterns):
            return "high"
        
        # Medium risk patterns
        medium_risk_patterns = [
            "write", "create", "modify", "change", "update", "install"
        ]
        
        if any(pattern in message_lower for pattern in medium_risk_patterns):
            return "medium"
        
        # Write operations on multiple files
        if intent == "write_file" and len(entities.get("files", [])) > 1:
            return "medium"
        
        return "low"
    
    def _suggest_approach(self, intent: str, entities: Dict[str, Any]) -> str:
        """Suggest an approach for handling the request."""
        approach_templates = {
            "read_file": "I'll read the specified file(s) and provide you with the content.",
            "write_file": "I'll create/modify the file with the requested changes.",
            "analyze_code": "I'll analyze the code structure and provide insights.",
            "search": "I'll search through the codebase to find what you're looking for.",
            "question": "I'll provide an explanation based on my knowledge.",
            "general_assistance": "I'll help you with your request step by step."
        }
        
        base_approach = approach_templates.get(intent, "I'll assist you with your request.")
        
        # Add specific details based on entities
        if "files" in entities:
            file_count = len(entities["files"])
            if file_count == 1:
                base_approach += f" File: {entities['files'][0]}"
            elif file_count > 1:
                base_approach += f" Files: {', '.join(entities['files'][:3])}"
                if file_count > 3:
                    base_approach += f" and {file_count - 3} more"
        
        return base_approach