"""
Command parsing and intent recognition for Loki Code.

This module handles natural language parsing, intent detection, and entity extraction
from user input to prepare it for intelligent routing.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from enum import Enum

from ...utils.logging import get_logger


class IntentType(Enum):
    """Types of user intents that can be detected."""
    FILE_ANALYSIS = "file_analysis"
    CODE_GENERATION = "code_generation"
    CODE_EXPLANATION = "code_explanation"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    TOOL_EXECUTION = "tool_execution"
    SYSTEM_COMMAND = "system_command"
    CONVERSATION = "conversation"
    HELP = "help"
    SEARCH = "search"


@dataclass
class Intent:
    """Represents a detected user intent."""
    type: IntentType
    confidence: float
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ParsedInput:
    """Structured representation of parsed user input."""
    original_text: str
    cleaned_text: str
    intent: Intent
    entities: Dict[str, Any]
    context_cues: Dict[str, Any]
    confidence: float
    language_hints: List[str] = field(default_factory=list)
    tool_hints: List[str] = field(default_factory=list)


@dataclass 
class ConversationContext:
    """Context information for command processing."""
    session_id: str
    project_path: Optional[str] = None
    current_file: Optional[str] = None
    recent_files: List[str] = field(default_factory=list)
    conversation_history: List[str] = field(default_factory=list)
    last_intent: Optional[IntentType] = None
    active_tools: List[str] = field(default_factory=list)


class CommandParser:
    """Parses natural language input into structured commands."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.intent_patterns = self._load_intent_patterns()
        self.entity_patterns = self._load_entity_patterns()
        
    def parse_input(self, user_input: str, 
                   context: ConversationContext) -> ParsedInput:
        """Parse user input into structured format."""
        
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
            context_cues=context_cues,
            confidence=confidence,
            language_hints=language_hints,
            tool_hints=tool_hints
        )
        
        self.logger.debug(f"Parsed input: intent={intent.type.value}, confidence={confidence:.2f}")
        return parsed
    
    def _preprocess_input(self, text: str) -> str:
        """Clean and normalize input text."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Normalize common variations
        replacements = {
            r'\banalyse\b': 'analyze',
            r'\bfavourite\b': 'favorite',
            r'\bcolour\b': 'color',
            r'\bcentre\b': 'center',
        }
        
        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)
        
        return text
    
    def _detect_intent(self, text: str) -> Intent:
        """Detect user intent from text."""
        
        # Check each intent type in order of specificity
        for intent_type, patterns in self.intent_patterns.items():
            for pattern_info in patterns:
                pattern = pattern_info['pattern']
                weight = pattern_info.get('weight', 1.0)
                
                if match := re.search(pattern, text, re.IGNORECASE):
                    confidence = self._pattern_confidence(pattern, text) * weight
                    return Intent(
                        type=intent_type,
                        confidence=confidence,
                        details={'match': match.group(), 'pattern': pattern}
                    )
        
        # Default to conversation intent
        return Intent(type=IntentType.CONVERSATION, confidence=0.5)
    
    def _load_intent_patterns(self) -> Dict[IntentType, List[Dict[str, Any]]]:
        """Load intent detection patterns with weights."""
        
        return {
            IntentType.HELP: [
                {'pattern': r'^\s*(help|h|\?)\s*$', 'weight': 1.0},
                {'pattern': r'\b(help|assist|guide)\s+me\b', 'weight': 0.9},
                {'pattern': r'\bhow\s+do\s+i\b', 'weight': 0.8},
                {'pattern': r'\bwhat\s+(can|commands|options)', 'weight': 0.8},
            ],
            
            IntentType.SYSTEM_COMMAND: [
                {'pattern': r'^\s*(status|clear|reset|theme|exit|quit)\s*$', 'weight': 1.0},
                {'pattern': r'^\s*(show|list)\s+(tools|commands|history)', 'weight': 0.95},
                {'pattern': r'^\s*set\s+\w+', 'weight': 0.9},
                {'pattern': r'^\s*(pwd|ls)\s*$', 'weight': 0.9},
            ],
            
            IntentType.FILE_ANALYSIS: [
                {'pattern': r'\b(analyze|examine|look\s+at|check|review|inspect)\s+.*\.(py|js|ts|java|cpp|c|rs|go)', 'weight': 0.95},
                {'pattern': r'\banalyze\s+(file|code)', 'weight': 0.9},
                {'pattern': r'\bwhat.*in\s+.*\.(py|js|ts)', 'weight': 0.85},
                {'pattern': r'\bshow.*functions.*in', 'weight': 0.8},
                {'pattern': r'\bread\s+.*\.(py|js|ts|java)', 'weight': 0.8},
                {'pattern': r'\bopen\s+.*\.(py|js|ts|java)', 'weight': 0.75},
            ],
            
            IntentType.CODE_GENERATION: [
                {'pattern': r'\b(create|write|generate|make|build)\s+.*function', 'weight': 0.95},
                {'pattern': r'\b(create|write|generate|make|build)\s+.*class', 'weight': 0.95},
                {'pattern': r'\bwrite.*code', 'weight': 0.9},
                {'pattern': r'\bimplement.*', 'weight': 0.85},
                {'pattern': r'\bgenerate.*script', 'weight': 0.8},
                {'pattern': r'\bcreate\s+a\s+.*\s+(for|that|to)', 'weight': 0.8},
            ],
            
            IntentType.CODE_EXPLANATION: [
                {'pattern': r'\b(explain|what\s+does|how\s+does|what\s+is)\s+.*', 'weight': 0.85},
                {'pattern': r'\bwhat.*mean', 'weight': 0.8},
                {'pattern': r'\bhow.*work', 'weight': 0.8},
                {'pattern': r'\bexplain.*code', 'weight': 0.9},
                {'pattern': r'\btell\s+me\s+about', 'weight': 0.75},
                {'pattern': r'\bwhat\s+(is|are)\s+this', 'weight': 0.7},
            ],
            
            IntentType.DEBUGGING: [
                {'pattern': r'\b(debug|fix|error|bug|issue|problem)\b', 'weight': 0.9},
                {'pattern': r'\bwhat.*wrong', 'weight': 0.85},
                {'pattern': r'\bfind.*problem', 'weight': 0.85},
                {'pattern': r'\bwhy.*not\s+working', 'weight': 0.8},
                {'pattern': r'\bthis\s+(doesn.t|does\s+not|isn.t|is\s+not)\s+work', 'weight': 0.8},
                {'pattern': r'\berror.*help', 'weight': 0.75},
            ],
            
            IntentType.REFACTORING: [
                {'pattern': r'\b(refactor|improve|optimize|clean\s+up|beautify)\b', 'weight': 0.9},
                {'pattern': r'\bmake.*better', 'weight': 0.8},
                {'pattern': r'\bimprove.*code', 'weight': 0.85},
                {'pattern': r'\boptimize.*performance', 'weight': 0.85},
                {'pattern': r'\bclean.*up', 'weight': 0.8},
                {'pattern': r'\brewrite.*better', 'weight': 0.8},
            ],
            
            IntentType.SEARCH: [
                {'pattern': r'^\s*(find|search|grep|look\s+for)\s+', 'weight': 0.9},
                {'pattern': r'\bfind.*in\s+.*\.(py|js|ts)', 'weight': 0.85},
                {'pattern': r'\bsearch.*for', 'weight': 0.8},
                {'pattern': r'\bwhere.*defined', 'weight': 0.8},
                {'pattern': r'\blocate.*function', 'weight': 0.75},
            ],
            
            IntentType.TOOL_EXECUTION: [
                {'pattern': r'^\s*(read|write|list|search|find)\s+[^a-zA-Z]', 'weight': 0.9},
                {'pattern': r'^\s*(run|execute)\s+', 'weight': 0.9},
                {'pattern': r'--\w+', 'weight': 0.8},  # Command line style arguments
                {'pattern': r'^\s*\w+\s+[^\s]+\.(py|js|ts)', 'weight': 0.75},
            ],
        }
    
    def _load_entity_patterns(self) -> Dict[str, str]:
        """Load entity extraction patterns."""
        
        return {
            'files': r'[^\s]+\.(py|js|ts|java|cpp|c|rs|go|json|yaml|yml|md|txt|csv|xml|html|css|sh|bat)',
            'functions': r'\bfunction\s+(\w+)|def\s+(\w+)|\b(\w+)\s*\(',
            'classes': r'\bclass\s+(\w+)|class\s+(\w+)\s*\{',
            'variables': r'\b(var|let|const)\s+(\w+)|\b(\w+)\s*=',
            'imports': r'\b(import|from|require)\s+([^\s;]+)',
            'languages': r'\b(python|javascript|typescript|java|rust|cpp|c\+\+|go|shell|bash)\b',
            'directories': r'[^\s]*[/\\][^\s]*',
            'urls': r'https?://[^\s]+',
            'numbers': r'\b\d+\b',
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
            file_keywords = r'\b(file|script|module|document)\s+([^\s]+)'
            file_refs = re.findall(file_keywords, text, re.IGNORECASE)
            if file_refs:
                entities.setdefault('file_refs', []).extend([ref[1] for ref in file_refs])
        
        elif intent.type == IntentType.CODE_GENERATION:
            # Look for what to create
            create_patterns = [
                r'\b(create|make|build)\s+(?:a\s+)?(\w+)',
                r'\bgenerate\s+(?:a\s+)?(\w+)',
                r'\bwrite\s+(?:a\s+)?(\w+)'
            ]
            for pattern in create_patterns:
                matches = re.findall(pattern, text, re.IGNORECASE)
                if matches:
                    entities.setdefault('create_targets', []).extend(
                        [match[1] if isinstance(match, tuple) else match for match in matches]
                    )
        
        return entities
    
    def _analyze_context(self, text: str, context: ConversationContext) -> Dict[str, Any]:
        """Analyze contextual cues."""
        
        context_cues = {}
        
        # Reference to previous conversation
        pronouns = re.findall(r'\b(this|that|it|these|those)\b', text, re.IGNORECASE)
        if pronouns:
            context_cues['references_previous'] = True
            context_cues['pronoun_count'] = len(pronouns)
        
        # Current project context
        if context.project_path:
            context_cues['has_project_context'] = True
            
        # Recent files context
        if context.recent_files:
            mentioned_files = self._find_mentioned_files(text, context.recent_files)
            if mentioned_files:
                context_cues['references_recent_files'] = mentioned_files
        
        # Continuation indicators
        continuation_words = r'\b(also|additionally|furthermore|moreover|besides|continue|next|then)\b'
        if re.search(continuation_words, text, re.IGNORECASE):
            context_cues['continuation'] = True
        
        # Question indicators
        question_words = r'\b(what|how|why|when|where|which|who)\b'
        question_marks = text.count('?')
        if re.search(question_words, text, re.IGNORECASE) or question_marks > 0:
            context_cues['is_question'] = True
            context_cues['question_score'] = len(re.findall(question_words, text, re.IGNORECASE)) + question_marks
        
        return context_cues
    
    def _find_mentioned_files(self, text: str, recent_files: List[str]) -> List[str]:
        """Find references to recent files in the text."""
        
        mentioned = []
        for file_path in recent_files:
            filename = file_path.split('/')[-1]
            basename = filename.split('.')[0]
            
            # Check for full filename or basename mention
            if filename.lower() in text.lower() or basename.lower() in text.lower():
                mentioned.append(file_path)
        
        return mentioned
    
    def _detect_languages(self, text: str, entities: Dict[str, Any]) -> List[str]:
        """Detect programming languages mentioned or implied."""
        
        languages = []
        
        # Direct mentions
        if 'languages' in entities:
            languages.extend(entities['languages'])
        
        # File extension implications
        if 'files' in entities:
            for file in entities['files']:
                ext = file.split('.')[-1].lower()
                lang_map = {
                    'py': 'python',
                    'js': 'javascript', 
                    'ts': 'typescript',
                    'java': 'java',
                    'cpp': 'cpp',
                    'c': 'c',
                    'rs': 'rust',
                    'go': 'go',
                    'sh': 'shell'
                }
                if ext in lang_map:
                    languages.append(lang_map[ext])
        
        return list(set(languages))  # Remove duplicates
    
    def _suggest_tools(self, intent: Intent, entities: Dict[str, Any]) -> List[str]:
        """Suggest appropriate tools based on intent and entities."""
        
        tools = []
        
        intent_tool_map = {
            IntentType.FILE_ANALYSIS: ['file_reader', 'code_analyzer'],
            IntentType.CODE_GENERATION: ['code_generator', 'file_writer'],
            IntentType.CODE_EXPLANATION: ['file_reader', 'code_analyzer'],
            IntentType.DEBUGGING: ['code_analyzer', 'debugger'],
            IntentType.REFACTORING: ['code_analyzer', 'refactoring_tool'],
            IntentType.SEARCH: ['file_searcher', 'code_searcher'],
            IntentType.TOOL_EXECUTION: [],  # Will be determined by routing
        }
        
        if intent.type in intent_tool_map:
            tools.extend(intent_tool_map[intent.type])
        
        # Entity-based tool suggestions
        if 'files' in entities:
            tools.extend(['file_reader', 'file_writer'])
        
        if 'directories' in entities:
            tools.append('directory_lister')
        
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
    
    def _calculate_confidence(self, intent: Intent, entities: Dict[str, Any], 
                            context_cues: Dict[str, Any]) -> float:
        """Calculate overall confidence score."""
        
        base_confidence = intent.confidence
        
        # Boost confidence with supporting entities
        if entities:
            entity_boost = min(len(entities) * 0.05, 0.2)
            base_confidence += entity_boost
        
        # Context cues adjustment
        if context_cues.get('is_question') and intent.type == IntentType.CODE_EXPLANATION:
            base_confidence += 0.1
        
        if context_cues.get('references_recent_files') and intent.type == IntentType.FILE_ANALYSIS:
            base_confidence += 0.15
        
        # Penalize very ambiguous inputs
        if len(entities) == 0 and intent.type == IntentType.CONVERSATION:
            base_confidence -= 0.1
        
        return max(0.0, min(base_confidence, 1.0))