"""
Tool call parsing and validation for LLM responses.

This module provides comprehensive parsing of LLM responses to extract
tool calls, validate them against schemas, and prepare them for execution.
"""

import re
import json
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Pattern, Union
from enum import Enum
import logging

from ..tool_registry import ToolRegistry
from ...tools.types import ToolSchema, ToolResult
from ...utils.logging import get_logger


class ToolCallPattern(Enum):
    """Different patterns for tool call extraction."""
    CODE_BLOCK = "code_block"          # ```tool format
    FUNCTION_CALL = "function_call"    # function_name(args)
    JSON_CALL = "json_call"           # {"tool": "name", "input": {...}}
    MARKDOWN_CALL = "markdown_call"   # **tool_name**: input
    NATURAL_CALL = "natural_call"     # Use the file_reader tool with...


@dataclass
class RawToolCall:
    """Raw tool call extracted from LLM response."""
    tool_name: str
    input_data: Dict[str, Any]
    raw_text: str
    pattern_type: ToolCallPattern
    start_position: int = 0
    end_position: int = 0
    confidence: float = 1.0


@dataclass
class ValidatedToolCall:
    """Validated tool call ready for execution."""
    raw_call: RawToolCall
    is_valid: bool
    error: Optional[str] = None
    tool_schema: Optional[ToolSchema] = None
    normalized_input: Optional[Dict[str, Any]] = None
    validation_warnings: List[str] = field(default_factory=list)


@dataclass
class ParsedResponse:
    """Complete parsed LLM response."""
    text_content: str
    tool_calls: List[ValidatedToolCall] = field(default_factory=list)
    reasoning: Optional[str] = None
    confidence_score: float = 0.8
    needs_clarification: bool = False
    clarification_questions: List[str] = field(default_factory=list)
    parsing_errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def has_valid_tool_calls(self) -> bool:
        """Check if response has any valid tool calls."""
        return any(call.is_valid for call in self.tool_calls)
    
    @property
    def has_errors(self) -> bool:
        """Check if response has any parsing errors."""
        return bool(self.parsing_errors) or any(not call.is_valid for call in self.tool_calls)


@dataclass
class ParsingConfig:
    """Configuration for response parsing."""
    confidence_threshold: float = 0.7
    strict_validation: bool = True
    enable_fuzzy_matching: bool = True
    max_tool_calls: int = 10
    extract_reasoning: bool = True
    detect_clarification: bool = True
    normalize_input: bool = True
    timeout_seconds: float = 5.0


class ResponseParser:
    """
    Parser for LLM responses that extracts and validates tool calls.
    
    Supports multiple tool call formats and provides comprehensive
    validation against tool schemas.
    """
    
    def __init__(self, tool_registry: ToolRegistry, config: Optional[ParsingConfig] = None):
        self.tool_registry = tool_registry
        self.config = config or ParsingConfig()
        self.logger = get_logger(__name__)
        
        # Compile regex patterns for efficiency
        self._patterns = self._compile_patterns()
        
        # Reasoning extraction patterns
        self._reasoning_patterns = [
            r"(?:thinking|reasoning|analysis):\s*(.+?)(?:\n\n|\n(?=\w+:))",
            r"let me think about this:\s*(.+?)(?:\n\n|\n(?=\w+:))",
            r"my approach is:\s*(.+?)(?:\n\n|\n(?=\w+:))"
        ]
        
        # Clarification detection patterns
        self._clarification_patterns = [
            r"(?:could you|can you|would you).*(?:clarify|specify|explain)",
            r"(?:i'm not sure|unclear|ambiguous).*(?:about|what|which|how)",
            r"(?:do you mean|did you mean|are you asking)",
            r"(?:more information|additional details|specific)"
        ]
    
    def parse_llm_response(self, response: str, context: Optional[Dict[str, Any]] = None) -> ParsedResponse:
        """Parse LLM response into structured components.
        
        Args:
            response: Raw LLM response text
            context: Optional conversation context
            
        Returns:
            ParsedResponse with extracted and validated components
        """
        start_time = time.perf_counter()
        
        try:
            # Extract tool calls using multiple patterns
            raw_tool_calls = self._extract_tool_calls(response)
            
            # Validate tool calls against registry
            validated_calls = self._validate_tool_calls(raw_tool_calls)
            
            # Extract reasoning and confidence
            reasoning = self._extract_reasoning(response) if self.config.extract_reasoning else None
            confidence = self._calculate_confidence(response, validated_calls, context)
            
            # Detect clarification needs
            needs_clarification, questions = self._detect_clarification_needs(response) if self.config.detect_clarification else (False, [])
            
            # Clean text content (remove tool call markup)
            text_content = self._clean_response_text(response, raw_tool_calls)
            
            # Calculate parsing time
            parsing_time = time.perf_counter() - start_time
            
            return ParsedResponse(
                text_content=text_content,
                tool_calls=validated_calls,
                reasoning=reasoning,
                confidence_score=confidence,
                needs_clarification=needs_clarification,
                clarification_questions=questions,
                metadata={
                    "parsing_time_ms": parsing_time * 1000,
                    "raw_tool_calls_found": len(raw_tool_calls),
                    "valid_tool_calls": len([c for c in validated_calls if c.is_valid])
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error parsing LLM response: {e}")
            return ParsedResponse(
                text_content=response,
                parsing_errors=[f"Parsing failed: {str(e)}"],
                confidence_score=0.1
            )
    
    def _extract_tool_calls(self, response: str) -> List[RawToolCall]:
        """Extract tool calls from different formats."""
        tool_calls = []
        
        # Try each pattern type
        for pattern_type, pattern_func in self._patterns.items():
            try:
                calls = pattern_func(response)
                tool_calls.extend(calls)
            except Exception as e:
                self.logger.warning(f"Error extracting {pattern_type} pattern: {e}")
        
        # Remove duplicates and sort by position
        unique_calls = self._deduplicate_tool_calls(tool_calls)
        
        # Limit number of tool calls
        if len(unique_calls) > self.config.max_tool_calls:
            self.logger.warning(f"Found {len(unique_calls)} tool calls, limiting to {self.config.max_tool_calls}")
            unique_calls = unique_calls[:self.config.max_tool_calls]
        
        return unique_calls
    
    def _extract_code_block_calls(self, response: str) -> List[RawToolCall]:
        """Extract ```tool format calls."""
        pattern = r'```tool\s*\n(?:tool_name:\s*)?(\w+)\s*\n(?:input:\s*)?({.*?})\s*\n```'
        matches = re.finditer(pattern, response, re.DOTALL | re.IGNORECASE)
        
        calls = []
        for match in matches:
            tool_name, input_json = match.groups()
            try:
                input_data = json.loads(input_json)
                calls.append(RawToolCall(
                    tool_name=tool_name,
                    input_data=input_data,
                    raw_text=match.group(0),
                    pattern_type=ToolCallPattern.CODE_BLOCK,
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=0.95
                ))
            except json.JSONDecodeError:
                self.logger.warning(f"Invalid JSON in code block tool call: {input_json}")
        
        return calls
    
    def _extract_function_calls(self, response: str) -> List[RawToolCall]:
        """Extract function_name(args) format calls."""
        # Pattern for function calls with JSON args
        pattern = r'(\w+)\s*\(\s*({.*?})\s*\)'
        matches = re.finditer(pattern, response, re.DOTALL)
        
        calls = []
        for match in matches:
            tool_name, input_json = match.groups()
            
            # Check if this looks like a tool name (not a programming function)
            if not self._looks_like_tool_name(tool_name):
                continue
            
            try:
                input_data = json.loads(input_json)
                calls.append(RawToolCall(
                    tool_name=tool_name,
                    input_data=input_data,
                    raw_text=match.group(0),
                    pattern_type=ToolCallPattern.FUNCTION_CALL,
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=0.8
                ))
            except json.JSONDecodeError:
                # Try to parse as simple key=value pairs
                input_data = self._parse_simple_args(input_json)
                if input_data:
                    calls.append(RawToolCall(
                        tool_name=tool_name,
                        input_data=input_data,
                        raw_text=match.group(0),
                        pattern_type=ToolCallPattern.FUNCTION_CALL,
                        start_position=match.start(),
                        end_position=match.end(),
                        confidence=0.7
                    ))
        
        return calls
    
    def _extract_json_calls(self, response: str) -> List[RawToolCall]:
        """Extract JSON format tool calls."""
        pattern = r'{"tool":\s*"(\w+)",\s*"input":\s*({.*?})}'
        matches = re.finditer(pattern, response, re.DOTALL)
        
        calls = []
        for match in matches:
            tool_name, input_json = match.groups()
            try:
                input_data = json.loads(input_json)
                calls.append(RawToolCall(
                    tool_name=tool_name,
                    input_data=input_data,
                    raw_text=match.group(0),
                    pattern_type=ToolCallPattern.JSON_CALL,
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=0.9
                ))
            except json.JSONDecodeError:
                self.logger.warning(f"Invalid JSON in tool call: {input_json}")
        
        return calls
    
    def _extract_markdown_calls(self, response: str) -> List[RawToolCall]:
        """Extract **tool_name**: input format calls."""
        pattern = r'\*\*(\w+)\*\*:\s*({.*?})(?=\n|$)'
        matches = re.finditer(pattern, response, re.DOTALL)
        
        calls = []
        for match in matches:
            tool_name, input_json = match.groups()
            
            # Check if this looks like a tool name
            if not self._looks_like_tool_name(tool_name):
                continue
            
            try:
                input_data = json.loads(input_json)
                calls.append(RawToolCall(
                    tool_name=tool_name,
                    input_data=input_data,
                    raw_text=match.group(0),
                    pattern_type=ToolCallPattern.MARKDOWN_CALL,
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=0.75
                ))
            except json.JSONDecodeError:
                self.logger.warning(f"Invalid JSON in markdown tool call: {input_json}")
        
        return calls
    
    def _extract_natural_calls(self, response: str) -> List[RawToolCall]:
        """Extract natural language tool calls."""
        calls = []
        
        # Pattern: "use the X tool with Y"
        pattern = r'use the (\w+) tool with\s+({.*?})'
        matches = re.finditer(pattern, response, re.IGNORECASE | re.DOTALL)
        
        for match in matches:
            tool_name, input_json = match.groups()
            try:
                input_data = json.loads(input_json)
                calls.append(RawToolCall(
                    tool_name=tool_name,
                    input_data=input_data,
                    raw_text=match.group(0),
                    pattern_type=ToolCallPattern.NATURAL_CALL,
                    start_position=match.start(),
                    end_position=match.end(),
                    confidence=0.6
                ))
            except json.JSONDecodeError:
                pass
        
        return calls
    
    def _validate_tool_calls(self, raw_calls: List[RawToolCall]) -> List[ValidatedToolCall]:
        """Validate tool calls against registry and schemas."""
        validated = []
        
        for raw_call in raw_calls:
            validated_call = self._validate_single_tool_call(raw_call)
            validated.append(validated_call)
        
        return validated
    
    def _validate_single_tool_call(self, raw_call: RawToolCall) -> ValidatedToolCall:
        """Validate a single tool call."""
        # Check if tool exists in registry
        tool_schema = self.tool_registry.get_tool_schema(raw_call.tool_name)
        
        if not tool_schema:
            # Try fuzzy matching if enabled
            if self.config.enable_fuzzy_matching:
                similar_tool = self._find_similar_tool(raw_call.tool_name)
                if similar_tool:
                    return ValidatedToolCall(
                        raw_call=raw_call,
                        is_valid=False,
                        error=f"Tool '{raw_call.tool_name}' not found. Did you mean '{similar_tool}'?",
                        validation_warnings=[f"Fuzzy match suggestion: {similar_tool}"]
                    )
            
            return ValidatedToolCall(
                raw_call=raw_call,
                is_valid=False,
                error=f"Tool '{raw_call.tool_name}' not found in registry"
            )
        
        # Validate input against schema
        validation_result = self._validate_input_schema(raw_call.input_data, tool_schema.input_schema)
        
        # Normalize input if requested
        normalized_input = None
        if self.config.normalize_input and validation_result["valid"]:
            normalized_input = self._normalize_input(raw_call.input_data, tool_schema.input_schema)
        
        return ValidatedToolCall(
            raw_call=raw_call,
            is_valid=validation_result["valid"],
            error=validation_result.get("error"),
            tool_schema=tool_schema,
            normalized_input=normalized_input,
            validation_warnings=validation_result.get("warnings", [])
        )
    
    def _validate_input_schema(self, input_data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data against tool schema."""
        try:
            # Basic validation - check required fields
            required_fields = schema.get("required", [])
            missing_fields = [field for field in required_fields if field not in input_data]
            
            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing required fields: {', '.join(missing_fields)}"
                }
            
            # Type validation for properties
            properties = schema.get("properties", {})
            warnings = []
            
            for field, value in input_data.items():
                if field in properties:
                    field_schema = properties[field]
                    field_type = field_schema.get("type")
                    
                    # Basic type checking
                    if field_type == "string" and not isinstance(value, str):
                        warnings.append(f"Field '{field}' should be string, got {type(value).__name__}")
                    elif field_type == "integer" and not isinstance(value, int):
                        warnings.append(f"Field '{field}' should be integer, got {type(value).__name__}")
                    elif field_type == "boolean" and not isinstance(value, bool):
                        warnings.append(f"Field '{field}' should be boolean, got {type(value).__name__}")
            
            return {
                "valid": True,
                "warnings": warnings
            }
            
        except Exception as e:
            return {
                "valid": False,
                "error": f"Schema validation error: {str(e)}"
            }
    
    def _normalize_input(self, input_data: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
        """Normalize input data according to schema."""
        normalized = input_data.copy()
        properties = schema.get("properties", {})
        
        for field, value in input_data.items():
            if field in properties:
                field_schema = properties[field]
                
                # Apply default values
                if value is None and "default" in field_schema:
                    normalized[field] = field_schema["default"]
                
                # Type coercion
                field_type = field_schema.get("type")
                if field_type == "string" and not isinstance(value, str):
                    normalized[field] = str(value)
                elif field_type == "integer" and isinstance(value, str) and value.isdigit():
                    normalized[field] = int(value)
                elif field_type == "boolean" and isinstance(value, str):
                    normalized[field] = value.lower() in ("true", "yes", "1", "on")
        
        return normalized
    
    def _extract_reasoning(self, response: str) -> Optional[str]:
        """Extract reasoning/thinking from response."""
        for pattern in self._reasoning_patterns:
            match = re.search(pattern, response, re.IGNORECASE | re.DOTALL)
            if match:
                reasoning = match.group(1).strip()
                if len(reasoning) > 10:  # Must have substantial content
                    return reasoning
        
        return None
    
    def _calculate_confidence(self, response: str, tool_calls: List[ValidatedToolCall], 
                            context: Optional[Dict[str, Any]]) -> float:
        """Calculate confidence score for the response."""
        confidence = 0.8  # Base confidence
        
        # Adjust based on tool call validity
        if tool_calls:
            valid_calls = sum(1 for call in tool_calls if call.is_valid)
            confidence *= (valid_calls / len(tool_calls))
        
        # Adjust based on response length and structure
        if len(response) < 10:
            confidence *= 0.5
        elif len(response) > 1000:
            confidence *= 0.9
        
        # Adjust based on uncertainty indicators
        uncertainty_words = ["maybe", "possibly", "might", "not sure", "unclear"]
        uncertainty_count = sum(1 for word in uncertainty_words if word in response.lower())
        confidence *= max(0.3, 1.0 - (uncertainty_count * 0.1))
        
        return min(1.0, max(0.1, confidence))
    
    def _detect_clarification_needs(self, response: str) -> tuple[bool, List[str]]:
        """Detect if response indicates need for clarification."""
        questions = []
        
        for pattern in self._clarification_patterns:
            matches = re.finditer(pattern, response, re.IGNORECASE)
            for match in matches:
                # Extract the sentence containing the match
                start = response.rfind('.', 0, match.start()) + 1
                end = response.find('.', match.end())
                if end == -1:
                    end = len(response)
                
                sentence = response[start:end].strip()
                if sentence and sentence not in questions:
                    questions.append(sentence)
        
        # Also look for explicit questions
        explicit_questions = re.findall(r'[^.!?]*\?', response)
        questions.extend([q.strip() + '?' for q in explicit_questions if q.strip()])
        
        return len(questions) > 0, questions[:3]  # Limit to 3 questions
    
    def _clean_response_text(self, response: str, raw_calls: List[RawToolCall]) -> str:
        """Remove tool call markup from response text."""
        cleaned = response
        
        # Remove tool calls by position (in reverse order to maintain positions)
        for call in sorted(raw_calls, key=lambda c: c.start_position, reverse=True):
            cleaned = cleaned[:call.start_position] + cleaned[call.end_position:]
        
        # Clean up extra whitespace
        cleaned = re.sub(r'\n\s*\n\s*\n', '\n\n', cleaned)
        cleaned = cleaned.strip()
        
        return cleaned
    
    def _deduplicate_tool_calls(self, calls: List[RawToolCall]) -> List[RawToolCall]:
        """Remove duplicate tool calls."""
        unique_calls = []
        seen_signatures = set()
        
        for call in sorted(calls, key=lambda c: (c.confidence, -c.start_position), reverse=True):
            signature = f"{call.tool_name}:{json.dumps(call.input_data, sort_keys=True)}"
            if signature not in seen_signatures:
                unique_calls.append(call)
                seen_signatures.add(signature)
        
        return sorted(unique_calls, key=lambda c: c.start_position)
    
    def _looks_like_tool_name(self, name: str) -> bool:
        """Check if a name looks like a tool name."""
        # Tool names typically have underscores or are compound words
        return (
            '_' in name or 
            name.islower() or 
            any(word in name.lower() for word in ['read', 'write', 'analyze', 'get', 'set', 'create', 'delete', 'list', 'search'])
        )
    
    def _find_similar_tool(self, tool_name: str) -> Optional[str]:
        """Find similar tool name using fuzzy matching."""
        available_tools = self.tool_registry.list_tool_names()
        
        # Simple similarity based on common substrings
        best_match = None
        best_score = 0
        
        for available_tool in available_tools:
            score = self._calculate_similarity(tool_name, available_tool)
            if score > best_score and score > 0.6:
                best_score = score
                best_match = available_tool
        
        return best_match
    
    def _calculate_similarity(self, str1: str, str2: str) -> float:
        """Calculate similarity between two strings."""
        if str1 == str2:
            return 1.0
        
        # Normalize
        str1 = str1.lower()
        str2 = str2.lower()
        
        # Check for substring matches
        if str1 in str2 or str2 in str1:
            return 0.8
        
        # Simple character overlap
        chars1 = set(str1)
        chars2 = set(str2)
        overlap = len(chars1 & chars2)
        total = len(chars1 | chars2)
        
        return overlap / total if total > 0 else 0.0
    
    def _parse_simple_args(self, args_str: str) -> Optional[Dict[str, Any]]:
        """Parse simple key=value arguments."""
        try:
            # Remove quotes and parse key=value pairs
            args_str = args_str.strip().strip('"\'')
            
            if not args_str:
                return {}
            
            # Split by comma and parse key=value
            pairs = [pair.strip() for pair in args_str.split(',')]
            result = {}
            
            for pair in pairs:
                if '=' in pair:
                    key, value = pair.split('=', 1)
                    key = key.strip().strip('"\'')
                    value = value.strip().strip('"\'')
                    
                    # Try to parse as JSON value
                    try:
                        result[key] = json.loads(value)
                    except json.JSONDecodeError:
                        result[key] = value
            
            return result if result else None
            
        except Exception:
            return None
    
    def _compile_patterns(self) -> Dict[ToolCallPattern, callable]:
        """Compile all extraction patterns for efficiency."""
        return {
            ToolCallPattern.CODE_BLOCK: self._extract_code_block_calls,
            ToolCallPattern.FUNCTION_CALL: self._extract_function_calls,
            ToolCallPattern.JSON_CALL: self._extract_json_calls,
            ToolCallPattern.MARKDOWN_CALL: self._extract_markdown_calls,
            ToolCallPattern.NATURAL_CALL: self._extract_natural_calls
        }