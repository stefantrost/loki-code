"""
Intelligent conversation management for progressive user interaction.

This module provides sophisticated conversation management that adapts to 
user preferences, handles different interaction types, and maintains context
for natural, progressive interactions.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime

from ...utils.logging import get_logger


class InteractionType(Enum):
    """Types of user interactions."""
    CLARIFICATION = "clarification"        # Asking for clarification
    PERMISSION = "permission"             # Requesting permission
    CONFIRMATION = "confirmation"         # Confirming actions
    PROGRESS_UPDATE = "progress_update"   # Showing progress
    ERROR_REPORT = "error_report"        # Reporting errors
    SUGGESTION = "suggestion"            # Making suggestions
    GENERAL = "general"                  # General conversation


class ExplanationLevel(Enum):
    """Levels of explanation detail."""
    MINIMAL = "minimal"      # Just the essentials
    STANDARD = "standard"    # Normal level of detail
    DETAILED = "detailed"    # Comprehensive explanations
    VERBOSE = "verbose"      # Maximum detail and context


class PersonalityStyle(Enum):
    """Agent personality styles."""
    PROFESSIONAL = "professional"  # Formal, business-like
    FRIENDLY = "friendly"          # Warm and approachable
    HELPFUL = "helpful"           # Focused on assistance
    CONCISE = "concise"           # Brief and to the point
    ANALYTICAL = "analytical"     # Technical and thorough


@dataclass
class UserPreferences:
    """User preferences for conversation style."""
    explanation_level: ExplanationLevel = ExplanationLevel.STANDARD
    personality_style: PersonalityStyle = PersonalityStyle.HELPFUL
    show_reasoning: bool = True
    show_progress: bool = True
    ask_before_major_changes: bool = True
    prefer_step_by_step: bool = False
    technical_level: str = "intermediate"  # beginner, intermediate, advanced
    
    # Interaction preferences
    max_clarification_questions: int = 3
    prefer_examples: bool = True
    show_alternatives: bool = True
    
    # Learned preferences (updated during conversation)
    preferred_tools: List[str] = field(default_factory=list)
    common_tasks: List[str] = field(default_factory=list)
    interaction_patterns: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationEntry:
    """A single entry in the conversation history."""
    role: str  # "user", "assistant", "system"
    content: str
    interaction_type: InteractionType = InteractionType.GENERAL
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_string(self) -> str:
        """Convert to string representation."""
        time_str = self.timestamp.strftime("%H:%M:%S")
        return f"[{time_str}] {self.role}: {self.content[:100]}..."


@dataclass
class InteractionContext:
    """Context for a specific interaction."""
    user_intent: str = ""
    confidence_level: float = 1.0
    ambiguous_aspects: List[str] = field(default_factory=list)
    suggested_clarifications: List[str] = field(default_factory=list)
    risk_level: str = "low"
    requires_permission: bool = False


@dataclass
class ConversationConfig:
    """Configuration for conversation management."""
    max_history_entries: int = 50
    context_window_size: int = 10
    max_context_tokens: int = 2000  # Added for memory manager compatibility
    auto_save_preferences: bool = True
    adapt_to_user_style: bool = True
    learning_enabled: bool = True
    
    # Response formatting
    use_markdown: bool = True
    use_emojis: bool = True
    show_thinking_process: bool = False
    
    # Interaction timing
    response_delay_ms: int = 0
    typing_simulation: bool = False


class ConversationManager:
    """
    Manages progressive user interactions with intelligent adaptation.
    
    This system handles different types of user interactions, adapts to user
    preferences, and maintains conversation context for natural dialogue.
    """
    
    def __init__(self, config: ConversationConfig, user_preferences: Optional[UserPreferences] = None):
        self.config = config
        self.user_preferences = user_preferences or UserPreferences()
        self.logger = get_logger(__name__)
        
        # Conversation state
        self.conversation_history: List[ConversationEntry] = []
        self.current_context: Optional[InteractionContext] = None
        self.session_start_time = time.time()
        
        # Adaptive learning
        self.interaction_patterns: Dict[str, int] = {}
        self.successful_strategies: Dict[str, List[str]] = {}
        
        # Response templates
        self.response_templates = self._initialize_response_templates()
    
    async def interact_with_user(self, message: str, interaction_type: InteractionType = InteractionType.GENERAL,
                                context: Optional[InteractionContext] = None) -> str:
        """
        Handle user interaction with intelligent adaptation.
        
        This is the main entry point for user interactions. It adapts the
        response style based on user preferences and interaction history.
        """
        try:
            # Store current context
            self.current_context = context or InteractionContext()
            
            # Add to conversation history
            self._add_to_history("assistant", message, interaction_type)
            
            # Adapt message based on user preferences and interaction type
            adapted_message = self._adapt_message(message, interaction_type, context)
            
            # Learn from this interaction
            if self.config.learning_enabled:
                self._learn_from_interaction(interaction_type, adapted_message)
            
            return adapted_message
            
        except Exception as e:
            self.logger.error(f"Error in user interaction: {e}")
            return self._get_fallback_response(message, interaction_type)
    
    async def get_user_input(self, prompt: str, interaction_type: InteractionType = InteractionType.GENERAL) -> str:
        """
        Get input from user with context-aware prompting.
        
        This method handles getting user input and adapts the prompt
        based on user preferences and interaction context.
        """
        try:
            # Adapt prompt to user preferences
            adapted_prompt = self._adapt_prompt(prompt, interaction_type)
            
            # Display prompt (in real implementation, this would be UI)
            print(adapted_prompt)
            
            # Get user response
            try:
                response = input().strip()
            except (EOFError, KeyboardInterrupt):
                response = ""
            
            # Add to conversation history
            self._add_to_history("user", response, interaction_type)
            
            return response
            
        except Exception as e:
            self.logger.error(f"Error getting user input: {e}")
            return ""
    
    async def ask_clarification(self, understanding: InteractionContext) -> Optional[str]:
        """
        Ask intelligent clarification questions.
        
        This method generates and asks clarification questions based on
        what aspects of the user's request are ambiguous.
        """
        if not understanding.ambiguous_aspects:
            return None
        
        # Limit clarification questions based on user preferences
        max_questions = self.user_preferences.max_clarification_questions
        questions = understanding.suggested_clarifications[:max_questions]
        
        if not questions:
            # Generate questions based on ambiguous aspects
            questions = [self._generate_clarification_question(aspect) 
                        for aspect in understanding.ambiguous_aspects[:max_questions]]
        
        # Format clarification request
        clarification_message = self._format_clarification_request(questions, understanding)
        
        return await self.get_user_input(clarification_message, InteractionType.CLARIFICATION)
    
    async def show_progress(self, current_step: str, total_steps: int, current_step_num: int) -> None:
        """Show progress to user if they prefer progress updates."""
        if not self.user_preferences.show_progress:
            return
        
        progress_message = self._format_progress_message(current_step, total_steps, current_step_num)
        await self.interact_with_user(progress_message, InteractionType.PROGRESS_UPDATE)
    
    async def report_error(self, error: Exception, context: str, recovery_suggestions: List[str] = None) -> str:
        """Report an error to the user with helpful context."""
        error_message = self._format_error_message(error, context, recovery_suggestions or [])
        return await self.interact_with_user(error_message, InteractionType.ERROR_REPORT)
    
    async def make_suggestion(self, suggestion: str, reasoning: str = "", alternatives: List[str] = None) -> str:
        """Make a suggestion to the user with optional reasoning and alternatives."""
        suggestion_message = self._format_suggestion_message(suggestion, reasoning, alternatives or [])
        return await self.interact_with_user(suggestion_message, InteractionType.SUGGESTION)
    
    def _adapt_message(self, message: str, interaction_type: InteractionType, 
                      context: Optional[InteractionContext]) -> str:
        """Adapt message based on user preferences and interaction type."""
        
        # Get base template for interaction type
        template = self.response_templates.get(interaction_type.value, {})
        
        # Adapt for personality style
        message = self._apply_personality_style(message, self.user_preferences.personality_style)
        
        # Adapt for explanation level
        message = self._apply_explanation_level(message, self.user_preferences.explanation_level)
        
        # Add formatting based on preferences
        if self.config.use_markdown:
            message = self._add_markdown_formatting(message, interaction_type)
        
        if self.config.use_emojis and self.user_preferences.personality_style == PersonalityStyle.FRIENDLY:
            message = self._add_emojis(message, interaction_type)
        
        # Show reasoning if requested
        if self.user_preferences.show_reasoning and context and context.confidence_level < 0.8:
            reasoning = self._generate_reasoning_explanation(context)
            if reasoning:
                message = f"{message}\\n\\n💭 **My reasoning**: {reasoning}"
        
        return message
    
    def _adapt_prompt(self, prompt: str, interaction_type: InteractionType) -> str:
        """Adapt prompt based on user preferences."""
        
        # Adjust formality based on personality style
        if self.user_preferences.personality_style == PersonalityStyle.PROFESSIONAL:
            prompt = self._make_formal(prompt)
        elif self.user_preferences.personality_style == PersonalityStyle.FRIENDLY:
            prompt = self._make_friendly(prompt)
        
        # Add context if user prefers detailed explanations
        if self.user_preferences.explanation_level in [ExplanationLevel.DETAILED, ExplanationLevel.VERBOSE]:
            prompt = self._add_context_to_prompt(prompt, interaction_type)
        
        return prompt
    
    def _apply_personality_style(self, message: str, style: PersonalityStyle) -> str:
        """Apply personality style to message."""
        
        style_adjustments = {
            PersonalityStyle.PROFESSIONAL: {
                "greetings": ["I'll assist you with", "I can help you", "Let me provide"],
                "confirmations": ["Certainly", "Of course", "I'll proceed"],
                "tone": "formal"
            },
            PersonalityStyle.FRIENDLY: {
                "greetings": ["I'd be happy to help", "Let's work on", "I'm excited to help"],
                "confirmations": ["Absolutely!", "Sounds great!", "Let's do it!"],
                "tone": "warm"
            },
            PersonalityStyle.HELPFUL: {
                "greetings": ["I'll help you", "Let me assist", "I can support you"],
                "confirmations": ["Sure thing", "I'll take care of it", "Consider it done"],
                "tone": "supportive"
            },
            PersonalityStyle.CONCISE: {
                "greetings": ["I'll", "Let me", "I can"],
                "confirmations": ["Yes", "Done", "Proceeding"],
                "tone": "brief"
            },
            PersonalityStyle.ANALYTICAL: {
                "greetings": ["I'll analyze", "Let me examine", "I'll investigate"],
                "confirmations": ["Confirmed", "Validated", "Proceeding with analysis"],
                "tone": "technical"
            }
        }
        
        adjustments = style_adjustments.get(style, style_adjustments[PersonalityStyle.HELPFUL])
        
        # Apply tone adjustments (simplified implementation)
        if adjustments["tone"] == "brief" and len(message) > 100:
            # Truncate for concise style
            message = message[:100] + "..."
        
        return message
    
    def _apply_explanation_level(self, message: str, level: ExplanationLevel) -> str:
        """Apply explanation level to message."""
        
        if level == ExplanationLevel.MINIMAL:
            # Keep only essential information
            sentences = message.split('. ')
            message = '. '.join(sentences[:2])  # Keep first 2 sentences
            
        elif level == ExplanationLevel.VERBOSE:
            # Add more context and details
            if "I'll" in message or "I can" in message:
                message += "\\n\\nThis approach will ensure we proceed safely and efficiently."
        
        return message
    
    def _add_markdown_formatting(self, message: str, interaction_type: InteractionType) -> str:
        """Add markdown formatting to message."""
        
        # Add formatting based on interaction type
        if interaction_type == InteractionType.ERROR_REPORT:
            if "Error:" in message:
                message = message.replace("Error:", "**⚠️ Error:**")
        
        elif interaction_type == InteractionType.SUGGESTION:
            if "Suggestion:" not in message:
                message = f"**💡 Suggestion**: {message}"
        
        elif interaction_type == InteractionType.PROGRESS_UPDATE:
            if "Progress:" not in message:
                message = f"**📈 Progress**: {message}"
        
        return message
    
    def _add_emojis(self, message: str, interaction_type: InteractionType) -> str:
        """Add emojis based on interaction type."""
        
        emoji_map = {
            InteractionType.CLARIFICATION: "🤔",
            InteractionType.PERMISSION: "🔐", 
            InteractionType.CONFIRMATION: "✅",
            InteractionType.PROGRESS_UPDATE: "⚡",
            InteractionType.ERROR_REPORT: "⚠️",
            InteractionType.SUGGESTION: "💡",
            InteractionType.GENERAL: "🤖"
        }
        
        emoji = emoji_map.get(interaction_type, "")
        if emoji and not message.startswith(emoji):
            message = f"{emoji} {message}"
        
        return message
    
    def _generate_clarification_question(self, ambiguous_aspect: str) -> str:
        """Generate a clarification question for an ambiguous aspect."""
        
        question_templates = {
            "file_target": "Which specific file would you like me to focus on?",
            "operation_scope": "Should I apply this to the entire project or specific files?",
            "implementation_approach": "What approach would you prefer for this implementation?",
            "priority_level": "How urgent is this task?",
            "output_format": "What format would you like for the output?",
            "detail_level": "How much detail would you like in the analysis?"
        }
        
        return question_templates.get(ambiguous_aspect, 
                                    f"Could you clarify what you mean by '{ambiguous_aspect}'?")
    
    def _format_clarification_request(self, questions: List[str], understanding: InteractionContext) -> str:
        """Format a clarification request message."""
        
        intro = "🤔 I want to make sure I understand correctly."
        
        if understanding.confidence_level < 0.5:
            intro = "🤔 I need some clarification to help you effectively."
        elif understanding.confidence_level < 0.7:
            intro = "🤔 I have a good idea of what you want, but let me confirm a few details."
        
        if len(questions) == 1:
            return f"{intro}\\n\\n{questions[0]}"
        else:
            question_list = "\\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))
            return f"{intro}\\n\\n{question_list}\\n\\nYou can answer any or all of these to help me assist you better."
    
    def _format_progress_message(self, current_step: str, total_steps: int, current_step_num: int) -> str:
        """Format a progress update message."""
        
        progress_bar = "▓" * current_step_num + "░" * (total_steps - current_step_num)
        percentage = int((current_step_num / total_steps) * 100)
        
        return f"⚡ **Progress** ({percentage}%): {current_step}\\n`{progress_bar}`\\nStep {current_step_num} of {total_steps}"
    
    def _format_error_message(self, error: Exception, context: str, recovery_suggestions: List[str]) -> str:
        """Format an error report message."""
        
        error_type = type(error).__name__
        error_message = str(error)
        
        message = f"⚠️ **I encountered an issue**\\n\\n"
        message += f"**Error Type**: {error_type}\\n"
        message += f"**Context**: {context}\\n"
        message += f"**Details**: {error_message}\\n"
        
        if recovery_suggestions:
            message += f"\\n**💡 Suggestions**:\\n"
            for i, suggestion in enumerate(recovery_suggestions, 1):
                message += f"{i}. {suggestion}\\n"
        
        message += "\\nHow would you like me to proceed?"
        
        return message
    
    def _format_suggestion_message(self, suggestion: str, reasoning: str, alternatives: List[str]) -> str:
        """Format a suggestion message."""
        
        message = f"💡 **Suggestion**: {suggestion}"
        
        if reasoning and self.user_preferences.show_reasoning:
            message += f"\\n\\n**Why**: {reasoning}"
        
        if alternatives and self.user_preferences.show_alternatives:
            message += f"\\n\\n**Alternatives**:\\n"
            for i, alt in enumerate(alternatives, 1):
                message += f"{i}. {alt}\\n"
        
        return message
    
    def _generate_reasoning_explanation(self, context: InteractionContext) -> str:
        """Generate an explanation of the agent's reasoning."""
        
        explanations = []
        
        if context.confidence_level < 0.8:
            explanations.append(f"I'm {int(context.confidence_level * 100)}% confident in my understanding")
        
        if context.ambiguous_aspects:
            explanations.append(f"Some aspects need clarification: {', '.join(context.ambiguous_aspects)}")
        
        if context.risk_level != "low":
            explanations.append(f"This has {context.risk_level} risk level")
        
        return ". ".join(explanations) if explanations else ""
    
    def _make_formal(self, text: str) -> str:
        """Make text more formal."""
        # Simple transformations for formality
        text = text.replace("I'll", "I will")
        text = text.replace("can't", "cannot")
        text = text.replace("won't", "will not")
        return text
    
    def _make_friendly(self, text: str) -> str:
        """Make text more friendly."""
        # Add friendly touches
        if not text.endswith(("!", "?", ".")):
            text += "!"
        return text
    
    def _add_context_to_prompt(self, prompt: str, interaction_type: InteractionType) -> str:
        """Add context to prompt for detailed explanation preference."""
        
        context_additions = {
            InteractionType.PERMISSION: "\\nThis helps me ensure I'm acting within appropriate boundaries.",
            InteractionType.CLARIFICATION: "\\nThis will help me provide the most relevant assistance.",
            InteractionType.CONFIRMATION: "\\nI want to make sure we're aligned before proceeding."
        }
        
        addition = context_additions.get(interaction_type, "")
        return prompt + addition
    
    def _add_to_history(self, role: str, content: str, interaction_type: InteractionType):
        """Add entry to conversation history."""
        
        entry = ConversationEntry(
            role=role,
            content=content,
            interaction_type=interaction_type
        )
        
        self.conversation_history.append(entry)
        
        # Trim history if it gets too long
        if len(self.conversation_history) > self.config.max_history_entries:
            self.conversation_history = self.conversation_history[-self.config.max_history_entries:]
    
    def _learn_from_interaction(self, interaction_type: InteractionType, response: str):
        """Learn from user interactions to improve future responses."""
        
        # Track interaction patterns
        type_key = interaction_type.value
        self.interaction_patterns[type_key] = self.interaction_patterns.get(type_key, 0) + 1
        
        # Track successful strategies (simplified)
        if len(response) > 0:
            strategy_key = f"{interaction_type.value}_{self.user_preferences.personality_style.value}"
            if strategy_key not in self.successful_strategies:
                self.successful_strategies[strategy_key] = []
            
            # Store response characteristics for learning
            characteristics = {
                "length": len(response),
                "has_examples": "example" in response.lower(),
                "has_reasoning": "reasoning" in response.lower(),
                "formal_tone": any(word in response for word in ["shall", "will", "certainly"])
            }
            
            self.successful_strategies[strategy_key].append(characteristics)
    
    def _get_fallback_response(self, message: str, interaction_type: InteractionType) -> str:
        """Get a fallback response if normal processing fails."""
        
        fallback_responses = {
            InteractionType.CLARIFICATION: "I need more information to help you effectively. Could you provide more details?",
            InteractionType.PERMISSION: "I'd like to proceed, but I need your permission first.",
            InteractionType.CONFIRMATION: "Please confirm if you'd like me to proceed with this action.",
            InteractionType.ERROR_REPORT: "I encountered an issue and need guidance on how to proceed.",
            InteractionType.GENERAL: "I'm here to help. Could you clarify what you'd like me to do?"
        }
        
        return fallback_responses.get(interaction_type, 
                                    "I'm ready to help. What would you like me to do?")
    
    def _initialize_response_templates(self) -> Dict[str, Dict[str, str]]:
        """Initialize response templates for different interaction types."""
        
        return {
            "clarification": {
                "intro": "I want to make sure I understand correctly.",
                "questions": "Could you clarify:",
                "outro": "This will help me provide the best assistance."
            },
            "permission": {
                "intro": "I need permission to:",
                "details": "This action will:",
                "outro": "Is this okay?"
            },
            "confirmation": {
                "intro": "I'm ready to:",
                "details": "This will:",
                "outro": "Should I proceed?"
            },
            "progress_update": {
                "intro": "Currently working on:",
                "status": "Progress:",
                "outro": "Next up:"
            },
            "error_report": {
                "intro": "I encountered an issue:",
                "details": "The problem:",
                "outro": "How should I proceed?"
            },
            "suggestion": {
                "intro": "I suggest:",
                "reasoning": "Because:",
                "outro": "What do you think?"
            }
        }
    
    def get_conversation_summary(self) -> Dict[str, Any]:
        """Get a summary of the conversation state."""
        
        return {
            "total_entries": len(self.conversation_history),
            "session_duration_minutes": (time.time() - self.session_start_time) / 60,
            "interaction_patterns": self.interaction_patterns,
            "user_preferences": {
                "explanation_level": self.user_preferences.explanation_level.value,
                "personality_style": self.user_preferences.personality_style.value,
                "technical_level": self.user_preferences.technical_level
            },
            "recent_interactions": [
                entry.interaction_type.value 
                for entry in self.conversation_history[-5:]
            ]
        }
    
    def update_user_preferences(self, **preferences):
        """Update user preferences based on feedback or learning."""
        
        for key, value in preferences.items():
            if hasattr(self.user_preferences, key):
                setattr(self.user_preferences, key, value)
                self.logger.info(f"Updated user preference {key} to {value}")
    
    def get_recent_context(self, max_entries: int = 5) -> List[ConversationEntry]:
        """Get recent conversation context."""
        
        return self.conversation_history[-max_entries:] if self.conversation_history else []