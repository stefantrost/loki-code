"""
Task Classification System for Loki Code.

This module provides intelligent task classification to enable task-aware
model selection. It analyzes prompts and contexts to determine the most
appropriate model for each specific task.
"""

import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Tuple, Optional, Any
import logging

from ..utils.logging import get_logger


class TaskType(Enum):
    """Task types for model selection."""
    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    CODE_ANALYSIS = "code_analysis"
    CHAT_CONVERSATION = "chat_conversation"
    PLANNING = "planning"
    DEBUGGING = "debugging"
    REFACTORING = "refactoring"
    DOCUMENTATION = "documentation"
    EXPLANATION = "explanation"
    GENERAL_QUESTION = "general_question"


@dataclass
class TaskClassification:
    """Result of task classification."""
    task_type: TaskType
    confidence: float  # 0.0 to 1.0
    complexity: float  # 0.0 to 1.0
    estimated_tokens: int
    context_clues: List[str]
    reasoning: str


@dataclass
class TaskContext:
    """Additional context for task classification."""
    conversation_history: Optional[List[str]] = None
    file_context: Optional[str] = None
    project_type: Optional[str] = None
    previous_task_type: Optional[TaskType] = None
    user_intent: Optional[str] = None


class TaskClassifier:
    """
    Intelligent task classifier for prompt analysis.
    
    Uses keyword analysis, pattern matching, and context clues to determine
    the type of task and its complexity, enabling optimal model selection.
    """
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._initialize_patterns()
        self._performance_history: Dict[TaskType, List[float]] = {}
    
    def _initialize_patterns(self):
        """Initialize keyword patterns for task classification."""
        self.task_patterns = {
            TaskType.CODE_GENERATION: {
                'keywords': [
                    'write', 'create', 'generate', 'implement', 'build', 'make',
                    'function', 'class', 'method', 'script', 'program', 'code',
                    'algorithm', 'solution', 'develop'
                ],
                'patterns': [
                    r'write\s+(?:a\s+)?(?:function|class|method|script)',
                    r'create\s+(?:a\s+)?(?:function|class|method|script)',
                    r'implement\s+(?:a\s+)?(?:function|class|method|algorithm)',
                    r'generate\s+(?:code|script|function)',
                    r'build\s+(?:a\s+)?(?:function|class|application)'
                ],
                'code_indicators': [
                    'def ', 'class ', 'function', 'return', 'import',
                    'for loop', 'if statement', 'while loop'
                ]
            },
            
            TaskType.CODE_REVIEW: {
                'keywords': [
                    'review', 'check', 'analyze', 'examine', 'audit', 'inspect',
                    'validate', 'verify', 'assess', 'evaluate', 'critique',
                    'feedback', 'improve', 'optimize'
                ],
                'patterns': [
                    r'review\s+(?:this\s+)?code',
                    r'check\s+(?:this\s+)?(?:code|function|implementation)',
                    r'analyze\s+(?:this\s+)?(?:code|function|implementation)',
                    r'what(?:\'s|\s+is)\s+wrong\s+with',
                    r'can\s+you\s+improve'
                ]
            },
            
            TaskType.DEBUGGING: {
                'keywords': [
                    'debug', 'fix', 'error', 'bug', 'issue', 'problem',
                    'broken', 'failing', 'crash', 'exception', 'traceback',
                    'not working', 'wrong', 'incorrect'
                ],
                'patterns': [
                    r'(?:fix|debug|solve)\s+(?:this\s+)?(?:bug|error|issue|problem)',
                    r'(?:why\s+)?(?:is\s+)?(?:this\s+)?(?:not\s+working|failing|broken)',
                    r'getting\s+(?:an\s+)?error',
                    r'exception.*traceback',
                    r'something(?:\'s|\s+is)\s+wrong'
                ]
            },
            
            TaskType.REFACTORING: {
                'keywords': [
                    'refactor', 'restructure', 'reorganize', 'clean up',
                    'optimize', 'improve', 'modernize', 'simplify',
                    'rewrite', 'redesign'
                ],
                'patterns': [
                    r'refactor\s+(?:this\s+)?code',
                    r'clean\s+up\s+(?:this\s+)?code',
                    r'make\s+(?:this\s+)?(?:code\s+)?(?:better|cleaner|more\s+efficient)',
                    r'improve\s+(?:this\s+)?(?:code|implementation)',
                    r'optimize\s+(?:this\s+)?(?:code|function)'
                ]
            },
            
            TaskType.DOCUMENTATION: {
                'keywords': [
                    'document', 'explain', 'describe', 'comment', 'docstring',
                    'readme', 'documentation', 'comments', 'annotate'
                ],
                'patterns': [
                    r'(?:write|add|create)\s+(?:documentation|comments|docstring)',
                    r'document\s+(?:this\s+)?(?:code|function|class)',
                    r'explain\s+(?:what\s+)?(?:this\s+)?(?:code|function)\s+does',
                    r'add\s+comments\s+to'
                ]
            },
            
            TaskType.PLANNING: {
                'keywords': [
                    'plan', 'design', 'architecture', 'approach', 'strategy',
                    'structure', 'organize', 'breakdown', 'steps', 'phases'
                ],
                'patterns': [
                    r'(?:how\s+)?(?:should\s+)?(?:i\s+)?(?:plan|design|structure)',
                    r'what(?:\'s|\s+is)\s+the\s+(?:best\s+)?(?:approach|strategy)',
                    r'break\s+(?:this\s+)?down\s+into\s+steps',
                    r'(?:overall\s+)?architecture'
                ]
            },
            
            TaskType.EXPLANATION: {
                'keywords': [
                    'explain', 'what', 'how', 'why', 'understand', 'mean',
                    'clarify', 'elaborate', 'describe', 'tell me'
                ],
                'patterns': [
                    r'(?:what\s+)?(?:does\s+)?(?:this\s+)?(?:mean|do)',
                    r'(?:how\s+)?(?:does\s+)?(?:this\s+)?work',
                    r'(?:can\s+you\s+)?explain',
                    r'(?:help\s+me\s+)?understand',
                    r'what(?:\'s|\s+is)\s+(?:the\s+)?(?:difference|purpose)'
                ]
            }
        }
    
    def classify_prompt(
        self, 
        prompt: str, 
        context: Optional[TaskContext] = None
    ) -> TaskClassification:
        """Classify a prompt to determine task type and complexity.
        
        Args:
            prompt: The user prompt to classify
            context: Additional context for classification
            
        Returns:
            TaskClassification with task type, confidence, and complexity
        """
        start_time = time.perf_counter()
        
        # Clean and prepare prompt
        clean_prompt = prompt.lower().strip()
        
        # Calculate scores for each task type
        task_scores = {}
        context_clues = []
        
        for task_type, patterns in self.task_patterns.items():
            score = self._calculate_task_score(clean_prompt, patterns, context_clues)
            task_scores[task_type] = score
        
        # Find best match
        best_task = max(task_scores.items(), key=lambda x: x[1])
        task_type, confidence = best_task
        
        # Default to general question if confidence is too low
        if confidence < 0.15:
            task_type = TaskType.GENERAL_QUESTION
            confidence = 0.5
        
        # Estimate complexity
        complexity = self._estimate_complexity(prompt, task_type, context)
        
        # Estimate token count
        estimated_tokens = self._estimate_tokens(prompt, task_type)
        
        # Generate reasoning
        reasoning = self._generate_reasoning(task_type, confidence, complexity, context_clues)
        
        classification_time = (time.perf_counter() - start_time) * 1000
        self.logger.debug(f"Task classification completed in {classification_time:.1f}ms: {task_type.value}")
        
        return TaskClassification(
            task_type=task_type,
            confidence=confidence,
            complexity=complexity,
            estimated_tokens=estimated_tokens,
            context_clues=context_clues,
            reasoning=reasoning
        )
    
    def _calculate_task_score(self, prompt: str, patterns: Dict, context_clues: List[str]) -> float:
        """Calculate score for a specific task type."""
        score_components = [
            self._score_keywords(prompt, patterns.get('keywords', []), context_clues, weight=0.6),
            self._score_patterns(prompt, patterns.get('patterns', []), context_clues, weight=0.8),
            self._score_code_indicators(prompt, patterns.get('code_indicators', []), context_clues, weight=0.1)
        ]
        return min(sum(score_components), 1.0)

    def _score_keywords(self, prompt: str, keywords: List[str], context_clues: List[str], weight: float) -> float:
        """Score based on keyword matches."""
        if not keywords:
            return 0.0
        
        matches = sum(1 for keyword in keywords if keyword in prompt)
        context_clues.extend(f"keyword: {kw}" for kw in keywords if kw in prompt)
        return (matches / len(keywords)) * weight

    def _score_patterns(self, prompt: str, patterns: List[str], context_clues: List[str], weight: float) -> float:
        """Score based on regex pattern matches."""
        if not patterns:
            return 0.0
        
        matches = sum(1 for pattern in patterns if re.search(pattern, prompt, re.IGNORECASE))
        context_clues.extend(f"pattern: {pat}" for pat in patterns if re.search(pat, prompt, re.IGNORECASE))
        return (matches / len(patterns)) * weight

    def _score_code_indicators(self, prompt: str, indicators: List[str], context_clues: List[str], weight: float) -> float:
        """Score based on code indicator matches."""
        if not indicators:
            return 0.0
        
        matches = sum(1 for indicator in indicators if indicator in prompt)
        context_clues.extend(f"code: {ind}" for ind in indicators if ind in prompt)
        return (matches / len(indicators)) * weight
    
    def _estimate_complexity(
        self, 
        prompt: str, 
        task_type: TaskType, 
        context: Optional[TaskContext] = None
    ) -> float:
        """Estimate task complexity from 0.0 (simple) to 1.0 (complex)."""
        complexity = 0.0
        
        # Base complexity by task type
        base_complexity = {
            TaskType.CHAT_CONVERSATION: 0.2,
            TaskType.GENERAL_QUESTION: 0.3,
            TaskType.EXPLANATION: 0.4,
            TaskType.DOCUMENTATION: 0.4,
            TaskType.CODE_REVIEW: 0.5,
            TaskType.DEBUGGING: 0.6,
            TaskType.CODE_GENERATION: 0.6,
            TaskType.REFACTORING: 0.7,
            TaskType.PLANNING: 0.8,
            TaskType.CODE_ANALYSIS: 0.7
        }
        
        complexity = base_complexity.get(task_type, 0.5)
        
        # Adjust based on prompt length (longer = more complex)
        word_count = len(prompt.split())
        if word_count > 100:
            complexity += 0.2
        elif word_count > 50:
            complexity += 0.1
        elif word_count < 10:
            complexity -= 0.1
        
        # Technical complexity indicators
        complex_indicators = [
            'algorithm', 'optimization', 'performance', 'scalability',
            'architecture', 'design pattern', 'concurrent', 'parallel',
            'database', 'distributed', 'microservice', 'api', 'framework'
        ]
        
        for indicator in complex_indicators:
            if indicator in prompt.lower():
                complexity += 0.1
        
        # Context-based adjustments
        if context:
            if context.project_type in ['enterprise', 'production', 'large-scale']:
                complexity += 0.2
            
            if context.conversation_history and len(context.conversation_history) > 5:
                complexity += 0.1  # Complex ongoing conversation
        
        return min(complexity, 1.0)
    
    def _estimate_tokens(self, prompt: str, task_type: TaskType) -> int:
        """Estimate token count for the response."""
        # Rough token estimation (1 word ≈ 1.3 tokens on average)
        base_tokens = len(prompt.split()) * 1.3
        
        # Task-specific multipliers for expected response length
        response_multipliers = {
            TaskType.CHAT_CONVERSATION: 1.5,
            TaskType.GENERAL_QUESTION: 2.0,
            TaskType.EXPLANATION: 3.0,
            TaskType.CODE_REVIEW: 2.5,
            TaskType.DEBUGGING: 3.0,
            TaskType.CODE_GENERATION: 4.0,
            TaskType.REFACTORING: 3.5,
            TaskType.DOCUMENTATION: 2.5,
            TaskType.PLANNING: 4.0,
            TaskType.CODE_ANALYSIS: 3.5
        }
        
        multiplier = response_multipliers.get(task_type, 2.0)
        estimated_tokens = int(base_tokens * multiplier)
        
        # Ensure reasonable bounds
        return max(50, min(estimated_tokens, 4000))
    
    def _generate_reasoning(
        self, 
        task_type: TaskType, 
        confidence: float, 
        complexity: float, 
        context_clues: List[str]
    ) -> str:
        """Generate human-readable reasoning for the classification."""
        reasoning_parts = [
            f"Classified as {task_type.value} with {confidence:.2f} confidence"
        ]
        
        if complexity > 0.7:
            reasoning_parts.append("High complexity task detected")
        elif complexity < 0.3:
            reasoning_parts.append("Simple task detected")
        
        if context_clues:
            top_clues = context_clues[:3]  # Show top 3 clues
            reasoning_parts.append(f"Key indicators: {', '.join(top_clues)}")
        
        return ". ".join(reasoning_parts)
    
    def update_performance(self, task_type: TaskType, success_rate: float):
        """Update performance history for task classification accuracy."""
        if task_type not in self._performance_history:
            self._performance_history[task_type] = []
        
        self._performance_history[task_type].append(success_rate)
        
        # Keep only recent history
        if len(self._performance_history[task_type]) > 100:
            self._performance_history[task_type] = self._performance_history[task_type][-100:]
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get statistics about classification performance."""
        stats = {}
        
        for task_type, history in self._performance_history.items():
            if history:
                stats[task_type.value] = {
                    'average_success_rate': sum(history) / len(history),
                    'recent_performance': history[-10:],
                    'total_classifications': len(history)
                }
        
        return stats
    
    def is_code_related(self, task_type: TaskType) -> bool:
        """Check if a task type is code-related."""
        code_tasks = {
            TaskType.CODE_GENERATION,
            TaskType.CODE_REVIEW,
            TaskType.CODE_ANALYSIS,
            TaskType.DEBUGGING,
            TaskType.REFACTORING
        }
        return task_type in code_tasks
    
    def requires_large_model(self, classification: TaskClassification) -> bool:
        """Determine if a task requires a large/capable model."""
        # High complexity tasks need large models
        if classification.complexity > 0.7:
            return True
        
        # Certain task types benefit from large models
        large_model_tasks = {
            TaskType.PLANNING,
            TaskType.CODE_ANALYSIS,
            TaskType.REFACTORING
        }
        
        return classification.task_type in large_model_tasks
    
    def prefers_fast_model(self, classification: TaskClassification) -> bool:
        """Determine if a task can use a fast/smaller model."""
        # Simple tasks can use fast models
        if classification.complexity < 0.4:
            return True
        
        # Certain task types are good for fast models
        fast_model_tasks = {
            TaskType.CHAT_CONVERSATION,
            TaskType.GENERAL_QUESTION,
            TaskType.EXPLANATION
        }
        
        return classification.task_type in fast_model_tasks