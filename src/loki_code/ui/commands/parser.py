"""
Command parsing and intent recognition for Loki Code.

This module handles natural language parsing using the new model strategy system
for intelligent intent detection and entity extraction.
"""

import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

from ...core.model_strategies import ModelStrategyFactory, ContextHandoff
from ...core.commands.types import IntentType, ParsedInput as CoreParsedInput, ConversationContext as CoreConversationContext
from ...utils.logging import get_logger


# Re-export types from core for backward compatibility
ParsedInput = CoreParsedInput
ConversationContext = CoreConversationContext


class CommandParser:
    """Parses natural language input into structured commands using LLM strategies."""

    def __init__(self, model_manager=None, strategy_config: Optional[Dict[str, Any]] = None):
        """
        Initialize parser with model strategy system.
        
        Args:
            model_manager: ModelManager instance for LLM access
            strategy_config: Configuration for strategy selection
        """
        self.logger = get_logger(__name__)
        self.model_manager = model_manager
        
        # Initialize strategy factory if model manager is provided
        if model_manager:
            self.strategy_factory = ModelStrategyFactory(
                model_manager=model_manager,
                config=strategy_config or self._get_default_strategy_config()
            )
        else:
            self.strategy_factory = None
            self.logger.warning("No model manager provided, falling back to pattern-based parsing")
        
        # Fallback pattern-based system (for backward compatibility)
        self.intent_patterns = self._load_intent_patterns()
        self.entity_patterns = self._load_entity_patterns()

    async def parse_input(self, user_input: str, context: ConversationContext) -> ParsedInput:
        """Parse user input into structured format using LLM strategies."""
        
        # If we have a strategy factory, use LLM-based parsing
        if self.strategy_factory:
            return await self._llm_parse_input(user_input, context)
        else:
            # Fallback to pattern-based parsing
            return self._pattern_parse_input(user_input, context)
    
    async def _llm_parse_input(self, user_input: str, context: ConversationContext) -> ParsedInput:
        """Parse input using LLM strategy system."""
        try:
            # Convert context to dict format expected by strategies
            context_dict = self._convert_context_to_dict(context)
            
            # Create strategy for this parsing task
            strategy = await self.strategy_factory.create_strategy(
                context=context_dict,
                user_input=user_input
            )
            
            # Use strategy to understand intent
            context_handoff = await strategy.understand_intent(user_input, context_dict)
            
            # Convert ContextHandoff back to ParsedInput format
            parsed = self._convert_handoff_to_parsed_input(
                user_input=user_input,
                context_handoff=context_handoff
            )
            
            self.logger.debug(f"LLM parsed input: intent={parsed.intent.type.value}, confidence={parsed.confidence:.2f}")
            return parsed
            
        except Exception as e:
            self.logger.error(f"LLM parsing failed: {e}, falling back to patterns")
            return self._pattern_parse_input(user_input, context)
    
    def _pattern_parse_input(self, user_input: str, context: ConversationContext) -> ParsedInput:
        """Original pattern-based parsing as fallback."""
        
        # Basic preprocessing
        cleaned_input = self._preprocess_input(user_input)

        # Intent detection
        intent = self._detect_intent(cleaned_input)

        # Entity extraction
        entities = self._extract_entities(cleaned_input, intent)

        # Context analysis
        context_cues = self._analyze_context(cleaned_input, context)

        # Language and tool hints
        language_hints = self._detect_languages(cleaned_input, entities)
        tool_hints = self._suggest_tools(intent, entities)

        # Confidence scoring
        confidence = self._calculate_confidence(intent, entities, context_cues)

        parsed = ParsedInput(
            original_text=user_input,
            cleaned_text=cleaned_input,
            intent=intent,
            entities=entities,
            confidence=confidence,
            metadata={
                "context_cues": context_cues,
                "language_hints": language_hints,
                "tool_hints": tool_hints,
                "parsing_method": "pattern_based"
            }
        )

        self.logger.debug(f"Pattern parsed input: intent={intent.type.value}, confidence={confidence:.2f}")
        return parsed
    
    def _get_default_strategy_config(self) -> Dict[str, Any]:
        """Get default configuration for strategy factory."""
        return {
            "selection_mode": "automatic",
            "default_strategy": "single_model",
            "strategy_configs": {
                "single_model": {
                    "primary_model": "llama3.1:8b",
                    "max_tokens": 1500,
                    "temperature": 0.1,
                    "enable_reasoning": False  # Faster for parsing
                },
                "multi_model_specialized": {
                    "intent_model": "llama3.1:8b",
                    "code_model": "codellama:13b",
                    "classification_max_tokens": 300,
                    "generation_max_tokens": 1200
                }
            },
            "resource_constraints": {
                "max_memory_mb": 4096,
                "max_response_time_ms": 5000  # Fast parsing
            }
        }
    
    def _convert_context_to_dict(self, context: ConversationContext) -> Dict[str, Any]:
        """Convert ConversationContext to dictionary format."""
        return {
            "session_id": context.session_id,
            "project_path": context.project_path,
            "current_file": context.current_file,
            "project_context": {
                "language": self._infer_project_language(context)
            },
            # Note: We don't include conversation_history here as it should
            # be handled by the model's native context window
            "parsing_context": {
                "recent_files": getattr(context, "recent_files", []),
                "last_intent": getattr(context, "last_intent", None),
                "active_tools": getattr(context, "active_tools", [])
            }
        }
    
    def _infer_project_language(self, context: ConversationContext) -> Optional[str]:
        """Infer primary project language from context."""
        if context.current_file:
            ext = context.current_file.split('.')[-1].lower()
            lang_map = {
                "py": "python",
                "js": "javascript", 
                "ts": "typescript",
                "java": "java",
                "rs": "rust",
                "go": "go",
                "cpp": "cpp",
                "c": "c"
            }
            return lang_map.get(ext)
        return None
    
    def _convert_handoff_to_parsed_input(
        self, 
        user_input: str, 
        context_handoff: ContextHandoff
    ) -> ParsedInput:
        """Convert ContextHandoff to ParsedInput format."""
        
        # Create Intent object (for backward compatibility)
        from ...core.commands.types import Intent
        intent = Intent(
            type=context_handoff.intent_type,
            confidence=context_handoff.confidence_scores.get("overall", 0.8),
            reasoning=context_handoff.metadata.get("reasoning", "")
        )
        
        # Extract entities
        entities = context_handoff.extracted_entities
        
        # Calculate overall confidence
        confidence = context_handoff.confidence_scores.get("overall", 0.8)
        
        return ParsedInput(
            original_text=user_input,
            cleaned_text=user_input.strip(),  # LLM handles preprocessing
            intent=intent,
            entities=entities,
            confidence=confidence,
            metadata={
                "context_handoff": context_handoff.to_dict(),
                "strategy_type": context_handoff.strategy_type.value if context_handoff.strategy_type else None,
                "model_used": context_handoff.model_used,
                "processing_time_ms": context_handoff.processing_time_ms,
                "parsing_method": "llm_based"
            }
        )

    def _preprocess_input(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text.strip())

        # Normalize common variations
        replacements = {
            r"\banalyse\b": "analyze",
            r"\bfavourite\b": "favorite",
            r"\bcolour\b": "color",
            r"\bcentre\b": "center",
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        return text

    def _detect_intent(self, text: str):
        """Detect user intent from text."""
        from ...core.commands.types import Intent

        # Check each intent type in order of specificity
        for intent_type, patterns in self.intent_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info["pattern"]
                weight = pattern_info.get("weight", 1.0)

                if match := re.search(pattern, text, re.IGNORECASE):
                    confidence = self._pattern_confidence(pattern, text) * weight
                    return Intent(
                        type=intent_type,
                        confidence=confidence,
                        reasoning=f"Matched pattern: {pattern}"
                    )

        # Default to conversation intent
        return Intent(type=IntentType.CONVERSATION, confidence=0.5, reasoning="No specific patterns matched")

    def _load_intent_patterns(self) -> Dict[IntentType, List[Dict[str, Any]]]:
        """Load intent detection patterns with weights."""

        return {
            IntentType.HELP: [
                {"pattern": r"^\s*(help|h|\?)\s*$", "weight": 1.0},
                {"pattern": r"\b(help|assist|guide)\s+me\b", "weight": 0.9},
                {"pattern": r"\bhow\s+do\s+i\b", "weight": 0.8},
                {"pattern": r"\bwhat\s+(can\s+you\s+do|commands|options|are\s+the\s+commands)", "weight": 0.8},
            ],
            IntentType.SYSTEM_COMMAND: [
                {"pattern": r"^\s*(status|clear|reset|theme|exit|quit)\s*$", "weight": 1.0},
                {"pattern": r"^\s*(show|list)\s+(tools|commands|history)", "weight": 0.95},
                {"pattern": r"^\s*set\s+\w+", "weight": 0.9},
                {"pattern": r"^\s*(pwd|ls)\s*$", "weight": 0.9},
            ],
            IntentType.FILE_ANALYSIS: [
                {
                    "pattern": r"\b(analyze|examine|look\s+at|check|review|inspect)\s+.*\.(py|js|ts|java|cpp|c|rs|go)",
                    "weight": 0.95,
                },
                {"pattern": r"\banalyze\s+(file|code)", "weight": 0.9},
                {"pattern": r"\bwhat.*in\s+.*\.(py|js|ts)", "weight": 0.85},
                {"pattern": r"\bshow.*functions.*in", "weight": 0.8},
                {"pattern": r"\bread\s+.*\.(py|js|ts|java)", "weight": 0.8},
                {"pattern": r"\bopen\s+.*\.(py|js|ts|java)", "weight": 0.75},
            ],
            IntentType.CODE_GENERATION: [
                {"pattern": r"\b(create|write|generate|make|build)\s+.*function", "weight": 0.95},
                {"pattern": r"\b(create|write|generate|make|build)\s+.*class", "weight": 0.95},
                {"pattern": r"\b(create|write|make)\s+(an?\s+)?(empty\s+)?file", "weight": 0.9},
                {
                    "pattern": r"\b(create|write|make)\s+[\w./]+\.(py|js|ts|java|txt|md)",
                    "weight": 0.9,
                },
                {
                    "pattern": r"\b(create|write|make)\s+.*\.(py|js|ts|java|txt|md)\s+file",
                    "weight": 0.95,
                },
                {
                    "pattern": r"\b(create|write|make)\s+.*\b(python|javascript|typescript|java)\s+file",
                    "weight": 0.95,
                },
                {
                    "pattern": r"\b(create|write|make)\s+.*called\s+['\"]?[\w./]+\.(py|js|ts|java|txt|md)",
                    "weight": 0.95,
                },
                {"pattern": r"\bwrite.*code", "weight": 0.9},
                {"pattern": r"\bimplement.*", "weight": 0.85},
                {"pattern": r"\bgenerate.*script", "weight": 0.8},
                {"pattern": r"\bcreate\s+a\s+.*\s+(for|that|to)", "weight": 0.8},
            ],
            IntentType.CONVERSATION: [
                {"pattern": r"\bwhat\s+(did\s+you\s+call|is\s+the\s+name\s+of|was\s+the\s+file)", "weight": 0.9},
                {"pattern": r"\bwhat\s+(file|name).*created", "weight": 0.85},
                {"pattern": r"\bwhat\s+(happened|did\s+you\s+do)", "weight": 0.8},
                {"pattern": r"\bwhere\s+(did\s+you\s+put|is\s+the\s+file)", "weight": 0.8},
                {"pattern": r"\bcan\s+you\s+tell\s+me", "weight": 0.75},
                {"pattern": r"\bwhat\s+about", "weight": 0.7},
            ],
            IntentType.DEBUGGING: [
                {"pattern": r"\b(debug|fix|error|bug|issue|problem)\b", "weight": 0.9},
                {"pattern": r"\bwhat.*wrong", "weight": 0.85},
                {"pattern": r"\bfind.*problem", "weight": 0.85},
                {"pattern": r"\bwhy.*not\s+working", "weight": 0.8},
                {"pattern": r"\bthis\s+(doesn.t|does\s+not|isn.t|is\s+not)\s+work", "weight": 0.8},
                {"pattern": r"\berror.*help", "weight": 0.75},
            ],
            IntentType.SEARCH: [
                {"pattern": r"^\s*(find|search|grep|look\s+for)\s+", "weight": 0.9},
                {"pattern": r"\bfind.*in\s+.*\.(py|js|ts)", "weight": 0.85},
                {"pattern": r"\bsearch.*for", "weight": 0.8},
                {"pattern": r"\bwhere.*defined", "weight": 0.8},
                {"pattern": r"\blocate.*function", "weight": 0.75},
            ],
        }

    def _load_entity_patterns(self) -> Dict[str, str]:
        """Load entity extraction patterns."""

        return {
            "files": r"[^\s'\"]+\.(?:py|js|ts|java|cpp|c|rs|go|json|yaml|yml|md|txt|csv|xml|html|css|sh|bat)",
            "functions": r"\bfunction\s+(\w+)|def\s+(\w+)|\b(\w+)\s*\(",
            "classes": r"\bclass\s+(\w+)|class\s+(\w+)\s*\{",
            "variables": r"\b(var|let|const)\s+(\w+)|\b(\w+)\s*=",
            "imports": r"\b(import|from|require)\s+([^\s;]+)",
            "languages": r"\b(python|javascript|typescript|java|rust|cpp|c\+\+|go|shell|bash)\b",
            "directories": r"[^\s]*[/\\][^\s]*",
            "urls": r"https?://[^\s]+",
            "numbers": r"\b\d+\b",
        }

    def _extract_entities(self, text: str, intent: Intent) -> Dict[str, Any]:
        """Extract entities based on patterns and intent."""

        entities = {}

        # Extract based on patterns
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, text, re.IGNORECASE)
            if matches:
                # Flatten tuples from groups
                flattened = []
                for match in matches:
                    if isinstance(match, tuple):
                        flattened.extend([m for m in match if m])
                    else:
                        flattened.append(match)
                entities[entity_type] = list(set(flattened))  # Remove duplicates

        # Intent-specific entity extraction
        if intent.type == IntentType.FILE_ANALYSIS:
            # Look for file references more aggressively
            file_keywords = r"\b(file|script|module|document)\s+([^\s]+)"
            file_refs = re.findall(file_keywords, text, re.IGNORECASE)
            if file_refs:
                entities.setdefault("file_refs", []).extend([ref[1] for ref in file_refs])

        elif intent.type == IntentType.CODE_GENERATION:
            # Look for what to create
            create_patterns = [
                r"\b(create|make|build)\s+(?:a\s+)?(\w+)",
                r"\bgenerate\s+(?:a\s+)?(\w+)",
                r"\bwrite\s+(?:a\s+)?(\w+)",
            ]
            for pattern in create_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    entities.setdefault("create_targets", []).extend(
                        [match[1] if isinstance(match, tuple) else match for match in matches]
                    )

        return entities

    def _analyze_context(self, text: str, context: ConversationContext) -> Dict[str, Any]:
        """Analyze contextual cues."""

        context_cues = {}

        # Reference to previous conversation
        pronouns = re.findall(r"\b(this|that|it|these|those)\b", text, re.IGNORECASE)
        if pronouns:
            context_cues["references_previous"] = True
            context_cues["pronoun_count"] = len(pronouns)

        # Current project context
        if context.project_path:
            context_cues["has_project_context"] = True

        # Recent files context - handle both old and new context formats
        recent_files = getattr(context, "recent_files", [])
        if recent_files:
            mentioned_files = self._find_mentioned_files(text, recent_files)
            if mentioned_files:
                context_cues["references_recent_files"] = mentioned_files

        # Continuation indicators
        continuation_words = (
            r"\b(also|additionally|furthermore|moreover|besides|continue|next|then)\b"
        )
        if re.search(continuation_words, text, re.IGNORECASE):
            context_cues["continuation"] = True

        # Question indicators
        question_words = r"\b(what|how|why|when|where|which|who)\b"
        question_marks = text.count("?")
        if re.search(question_words, text, re.IGNORECASE) or question_marks > 0:
            context_cues["is_question"] = True
            context_cues["question_score"] = (
                len(re.findall(question_words, text, re.IGNORECASE)) + question_marks
            )

        return context_cues

    def _find_mentioned_files(self, text: str, recent_files: List[str]) -> List[str]:
        """Find references to recent files in the text."""

        mentioned = []
        for file_path in recent_files:
            filename = file_path.split("/")[-1]
            basename = filename.split(".")[0]

            # Check for full filename or basename mention
            if filename.lower() in text.lower() or basename.lower() in text.lower():
                mentioned.append(file_path)

        return mentioned

    def _detect_languages(self, text: str, entities: Dict[str, Any]) -> List[str]:
        """Detect programming languages mentioned or implied."""

        languages = []

        # Direct mentions
        if "languages" in entities:
            languages.extend(entities["languages"])

        # File extension implications
        if "files" in entities:
            for file in entities["files"]:
                ext = file.split(".")[-1].lower()
                lang_map = {
                    "py": "python",
                    "js": "javascript",
                    "ts": "typescript",
                    "java": "java",
                    "cpp": "cpp",
                    "c": "c",
                    "rs": "rust",
                    "go": "go",
                    "sh": "shell",
                }
                if ext in lang_map:
                    languages.append(lang_map[ext])

        return list(set(languages))  # Remove duplicates

    def _suggest_tools(self, intent, entities: Dict[str, Any]) -> List[str]:
        """Suggest appropriate tools based on intent and entities."""

        tools = []

        intent_tool_map = {
            IntentType.FILE_ANALYSIS: ["file_reader", "code_analyzer"],
            IntentType.CODE_GENERATION: ["code_generator", "file_writer"],
            IntentType.DEBUGGING: ["code_analyzer", "debugger"],
            IntentType.SEARCH: ["file_searcher", "code_searcher"],
        }

        if intent.type in intent_tool_map:
            tools.extend(intent_tool_map[intent.type])

        # Entity-based tool suggestions
        if "files" in entities:
            tools.extend(["file_reader", "file_writer"])

        if "directories" in entities:
            tools.append("directory_lister")

        return list(set(tools))  # Remove duplicates

    def _pattern_confidence(self, pattern: str, text: str) -> float:
        """Calculate confidence score for a pattern match."""

        # Base confidence
        confidence = 0.7

        # Adjust based on pattern specificity
        if len(pattern) > 50:  # Very specific patterns
            confidence += 0.2
        elif len(pattern) > 20:  # Moderately specific
            confidence += 0.1

        # Adjust based on match position
        match = re.search(pattern, text, re.IGNORECASE)
        if match and match.start() == 0:  # Matches at beginning
            confidence += 0.1

        # Adjust based on text length vs pattern
        text_words = len(text.split())
        if text_words <= 3:  # Short, precise commands
            confidence += 0.1
        elif text_words > 20:  # Long, complex text
            confidence -= 0.1

        return min(confidence, 1.0)

    def _calculate_confidence(
        self, intent, entities: Dict[str, Any], context_cues: Dict[str, Any]
    ) -> float:
        """Calculate overall confidence score."""

        base_confidence = intent.confidence

        # Boost confidence with supporting entities
        if entities:
            entity_boost = min(len(entities) * 0.05, 0.2)
            base_confidence += entity_boost

        # Context cues adjustment
        if context_cues.get("is_question") and intent.type == IntentType.CONVERSATION:
            base_confidence += 0.1

        if context_cues.get("references_recent_files") and intent.type == IntentType.FILE_ANALYSIS:
            base_confidence += 0.15

        # Penalize very ambiguous inputs
        if len(entities) == 0 and intent.type == IntentType.CONVERSATION:
            base_confidence -= 0.1

        return max(0.0, min(base_confidence, 1.0))
