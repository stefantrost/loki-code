"""
Intelligent Model Manager for Loki Code.

This module implements task-aware model selection, resource optimization,
and intelligent routing between different LLM models based on task
requirements and system constraints.
"""

import asyncio
import time
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from enum import Enum
import psutil
import logging

from .task_classifier import TaskClassifier, TaskClassification, TaskType, TaskContext
from .providers import (
    BaseLLMProvider, create_llm_provider, GenerationRequest, GenerationResponse,
    ProviderConnectionError, ProviderModelError, ProviderTimeoutError
)
from ..config.models import LokiCodeConfig
from ..utils.logging import get_logger


class SelectionStrategy(Enum):
    """Model selection strategies."""
    SPEED_OPTIMIZED = "speed_optimized"
    QUALITY_OPTIMIZED = "quality_optimized"
    BALANCED = "balanced"
    COST_OPTIMIZED = "cost_optimized"


@dataclass
class ModelConfig:
    """Configuration for a specific model."""
    provider: str
    model_name: str
    task_types: List[TaskType]
    max_complexity: float = 1.0
    min_complexity: float = 0.0
    priority: int = 1  # Higher = preferred
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    cost_per_token: Optional[float] = None
    memory_requirement: int = 1024  # MB
    concurrent_limit: int = 1


@dataclass
class ResourceConstraints:
    """System resource constraints."""
    max_memory_mb: int = 8192
    max_concurrent_models: int = 2
    prefer_speed: bool = False
    prefer_quality: bool = False
    max_cost_per_request: Optional[float] = None


@dataclass
class ModelPerformance:
    """Performance metrics for a model."""
    model_name: str
    task_type: TaskType
    response_time: float
    tokens_per_second: float
    memory_usage: int
    success_rate: float
    user_satisfaction: Optional[float] = None
    timestamp: float = field(default_factory=time.time)


@dataclass
class ActiveModel:
    """Information about an active model instance."""
    provider: BaseLLMProvider
    model_name: str
    load_time: float
    last_used: float
    memory_usage: int
    request_count: int = 0
    error_count: int = 0


class ModelManager:
    """
    Intelligent model manager for task-aware model selection.
    
    Manages multiple LLM models, automatically selects the best model
    for each task, optimizes resource usage, and provides fallback
    mechanisms for robust operation.
    """
    
    def __init__(self, config: LokiCodeConfig):
        """Initialize the model manager.
        
        Args:
            config: Configuration containing model and provider settings
        """
        self.config = config
        self.logger = get_logger(__name__)
        self.task_classifier = TaskClassifier()
        
        # Active models and performance tracking
        self._active_models: Dict[str, ActiveModel] = {}
        self._performance_history: List[ModelPerformance] = []
        self._model_configs: Dict[str, ModelConfig] = {}
        self._fallback_models: List[str] = []
        
        # Resource management
        self._resource_lock = threading.RLock()
        self._cleanup_interval = config.performance.cleanup_interval_seconds
        self._last_cleanup = time.time()
        
        # Load model configurations
        self._load_model_configurations()
        
        # Selection strategy
        self.selection_strategy = SelectionStrategy.BALANCED
        
    def _load_model_configurations(self):
        """Load model configurations from config."""
        # Default single model configuration (backward compatibility)
        default_model = ModelConfig(
            provider=self.config.llm.provider,
            model_name=self.config.llm.model,
            task_types=list(TaskType),  # Supports all task types
            temperature=self.config.llm.temperature,
            max_tokens=self.config.llm.max_tokens,
            priority=1
        )
        self._model_configs[self.config.llm.model] = default_model
        self._fallback_models = [self.config.llm.model]
        
        # Load additional model configurations if available
        if hasattr(self.config, 'models') and self.config.models:
            self._load_multi_model_config()
        
        self.logger.info(f"Loaded {len(self._model_configs)} model configurations")
    
    def _load_multi_model_config(self):
        """Load multi-model configuration."""
        # This would be extended when we update the config schema
        # For now, we support the basic single-model setup
        pass
    
    async def generate(
        self, 
        prompt: str, 
        task_context: Optional[TaskContext] = None,
        override_model: Optional[str] = None
    ) -> GenerationResponse:
        """Generate text using the optimal model for the task.
        
        Args:
            prompt: The input prompt
            task_context: Additional context for task classification
            override_model: Specific model to use (overrides selection)
            
        Returns:
            GenerationResponse from the selected model
        """
        start_time = time.perf_counter()
        
        try:
            # Step 1: Classify task and select model
            classification, selected_model = await self._classify_and_select_model(
                prompt, task_context, override_model
            )
            
            # Step 2: Create generation request
            provider, request = await self._create_generation_request(
                prompt, selected_model
            )
            
            # Step 3: Generate response
            response = await provider.generate(request)
            
            # Step 4: Track performance metrics
            generation_time = (time.perf_counter() - start_time) * 1000
            await self._track_performance_metrics(
                selected_model, classification.task_type, generation_time, response
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Generation failed: {e}")
            # Try fallback if not using override model
            if not override_model:
                classification = self.task_classifier.classify_prompt(prompt, task_context)
                return await self._try_fallback(prompt, classification, e)
            raise
    
    async def stream_generate(
        self, 
        prompt: str, 
        task_context: Optional[TaskContext] = None,
        override_model: Optional[str] = None
    ):
        """Generate text with streaming using the optimal model.
        
        Args:
            prompt: The input prompt
            task_context: Additional context for task classification
            override_model: Specific model to use (overrides selection)
            
        Yields:
            Text tokens as they are generated
        """
        try:
            # Classify the task
            classification = self.task_classifier.classify_prompt(prompt, task_context)
            
            # Select the optimal model
            if override_model:
                selected_model = override_model
            else:
                selected_model = await self._select_optimal_model(classification)
            
            # Get or create provider
            provider = await self._get_provider(selected_model)
            
            # Create generation request
            model_config = self._model_configs.get(selected_model)
            request = GenerationRequest(
                prompt=prompt,
                model=selected_model,
                temperature=model_config.temperature if model_config else None,
                max_tokens=model_config.max_tokens if model_config else None,
                stream=True
            )
            
            # Stream response
            start_time = time.perf_counter()
            token_count = 0
            
            async for token in provider.stream_generate(request):
                token_count += 1
                yield token
            
            # Record performance
            generation_time = (time.perf_counter() - start_time) * 1000
            await self._record_performance(
                selected_model, 
                classification.task_type, 
                generation_time, 
                token_count
            )
            
        except Exception as e:
            self.logger.error(f"Streaming generation failed: {e}")
            if not override_model:
                # For streaming, we can't easily fall back, so just log and re-raise
                pass
            raise
    
    async def _classify_and_select_model(
        self, 
        prompt: str, 
        task_context: Optional[TaskContext], 
        override_model: Optional[str]
    ) -> Tuple[TaskClassification, str]:
        """Handle task classification and model selection logic.
        
        Args:
            prompt: The input prompt
            task_context: Additional context for task classification
            override_model: Specific model to use (overrides selection)
            
        Returns:
            Tuple of (classification result, selected model name)
        """
        # Classify the task
        classification = self.task_classifier.classify_prompt(prompt, task_context)
        self.logger.debug(f"Task classified: {classification.task_type.value} "
                        f"(confidence: {classification.confidence:.2f}, "
                        f"complexity: {classification.complexity:.2f})")
        
        # Select the optimal model
        if override_model:
            selected_model = override_model
            self.logger.info(f"Using override model: {selected_model}")
        else:
            selected_model = await self._select_optimal_model(classification)
        
        return classification, selected_model
    
    async def _create_generation_request(
        self, 
        prompt: str, 
        selected_model: str
    ) -> Tuple[BaseLLMProvider, GenerationRequest]:
        """Create standardized generation request.
        
        Args:
            prompt: The input prompt
            selected_model: Selected model name
            
        Returns:
            Tuple of (provider instance, generation request)
        """
        # Get or create provider
        provider = await self._get_provider(selected_model)
        
        # Create generation request with model-specific configuration
        model_config = self._model_configs.get(selected_model)
        request = GenerationRequest(
            prompt=prompt,
            model=selected_model,
            temperature=model_config.temperature if model_config else None,
            max_tokens=model_config.max_tokens if model_config else None
        )
        
        return provider, request
    
    async def _track_performance_metrics(
        self, 
        model_name: str, 
        task_type: TaskType, 
        generation_time_ms: float, 
        response: GenerationResponse
    ) -> None:
        """Centralized performance tracking.
        
        Args:
            model_name: Name of the model used
            task_type: Type of task performed
            generation_time_ms: Generation time in milliseconds
            response: The generation response
        """
        # Calculate token count
        token_count = len(response.content.split())
        
        # Record performance metrics
        await self._record_performance(
            model_name, 
            task_type, 
            generation_time_ms, 
            token_count
        )
        
        self.logger.info(f"Generated response using {model_name} "
                       f"in {generation_time_ms:.1f}ms ({token_count} tokens)")
    
    async def _select_optimal_model(self, classification: TaskClassification) -> str:
        """Select the optimal model for a given task classification.
        
        Args:
            classification: Task classification result
            
        Returns:
            Name of the selected model
        """
        # Get eligible models for this task type
        eligible_models = self._get_eligible_models(classification)
        
        if not eligible_models:
            # Fall back to default model
            return self._fallback_models[0]
        
        # Apply selection strategy
        if self.selection_strategy == SelectionStrategy.SPEED_OPTIMIZED:
            return self._select_fastest_model(eligible_models, classification)
        elif self.selection_strategy == SelectionStrategy.QUALITY_OPTIMIZED:
            return self._select_best_quality_model(eligible_models, classification)
        elif self.selection_strategy == SelectionStrategy.COST_OPTIMIZED:
            return self._select_cheapest_model(eligible_models, classification)
        else:  # BALANCED
            return self._select_balanced_model(eligible_models, classification)
    
    def _get_eligible_models(self, classification: TaskClassification) -> List[str]:
        """Get models eligible for a task classification."""
        eligible = []
        
        for model_name, config in self._model_configs.items():
            # Check if model supports this task type
            if classification.task_type in config.task_types:
                # Check complexity bounds
                if (config.min_complexity <= classification.complexity <= config.max_complexity):
                    eligible.append(model_name)
        
        return eligible
    
    def _select_fastest_model(self, eligible_models: List[str], classification: TaskClassification) -> str:
        """Select the fastest model from eligible models."""
        # Use performance history to find fastest model
        fastest_model = eligible_models[0]
        best_speed = float('inf')
        
        for model_name in eligible_models:
            avg_response_time = self._get_average_response_time(model_name, classification.task_type)
            if avg_response_time < best_speed:
                best_speed = avg_response_time
                fastest_model = model_name
        
        return fastest_model
    
    def _select_best_quality_model(self, eligible_models: List[str], classification: TaskClassification) -> str:
        """Select the highest quality model from eligible models."""
        # Prefer models with higher priority and better success rates
        best_model = eligible_models[0]
        best_score = 0
        
        for model_name in eligible_models:
            config = self._model_configs[model_name]
            success_rate = self._get_success_rate(model_name, classification.task_type)
            
            # Score based on priority and success rate
            score = config.priority * success_rate
            
            if score > best_score:
                best_score = score
                best_model = model_name
        
        return best_model
    
    def _select_cheapest_model(self, eligible_models: List[str], classification: TaskClassification) -> str:
        """Select the most cost-effective model."""
        cheapest_model = eligible_models[0]
        lowest_cost = float('inf')
        
        for model_name in eligible_models:
            config = self._model_configs[model_name]
            if config.cost_per_token:
                estimated_cost = config.cost_per_token * classification.estimated_tokens
                if estimated_cost < lowest_cost:
                    lowest_cost = estimated_cost
                    cheapest_model = model_name
        
        return cheapest_model
    
    def _select_balanced_model(self, eligible_models: List[str], classification: TaskClassification) -> str:
        """Select a balanced model considering speed, quality, and cost."""
        best_model = eligible_models[0]
        best_score = 0
        
        for model_name in eligible_models:
            config = self._model_configs[model_name]
            
            # Balanced scoring
            speed_score = 1.0 / max(self._get_average_response_time(model_name, classification.task_type), 0.1)
            quality_score = config.priority * self._get_success_rate(model_name, classification.task_type)
            
            # Normalize and combine scores
            combined_score = (speed_score * 0.3) + (quality_score * 0.7)
            
            if combined_score > best_score:
                best_score = combined_score
                best_model = model_name
        
        return best_model
    
    async def _get_provider(self, model_name: str) -> BaseLLMProvider:
        """Get or create a provider for the specified model."""
        with self._resource_lock:
            # Check if model is already active
            if model_name in self._active_models:
                active_model = self._active_models[model_name]
                active_model.last_used = time.time()
                active_model.request_count += 1
                return active_model.provider
            
            # Check resource constraints
            await self._cleanup_unused_models()
            
            if len(self._active_models) >= self._get_max_concurrent_models():
                # Unload least recently used model
                await self._unload_lru_model()
            
            # Create new provider
            model_config = self._model_configs.get(model_name)
            if not model_config:
                raise ValueError(f"No configuration found for model: {model_name}")
            
            # Create provider with model-specific config
            temp_config = self.config
            temp_config.llm.model = model_name
            
            provider = create_llm_provider(temp_config)
            
            # Track active model
            self._active_models[model_name] = ActiveModel(
                provider=provider,
                model_name=model_name,
                load_time=time.time(),
                last_used=time.time(),
                memory_usage=model_config.memory_requirement
            )
            
            self.logger.info(f"Loaded model: {model_name}")
            return provider
    
    async def _cleanup_unused_models(self):
        """Clean up unused models to free resources."""
        current_time = time.time()
        
        # Only run cleanup periodically
        if current_time - self._last_cleanup < self._cleanup_interval:
            return
        
        self._last_cleanup = current_time
        
        # Find models that haven't been used recently
        unused_models = []
        for model_name, active_model in self._active_models.items():
            if current_time - active_model.last_used > self._cleanup_interval:
                unused_models.append(model_name)
        
        # Unload unused models
        for model_name in unused_models:
            await self._unload_model(model_name)
    
    async def _unload_lru_model(self):
        """Unload the least recently used model."""
        if not self._active_models:
            return
        
        # Find LRU model
        lru_model = min(self._active_models.items(), key=lambda x: x[1].last_used)
        await self._unload_model(lru_model[0])
    
    async def _unload_model(self, model_name: str):
        """Unload a specific model."""
        if model_name in self._active_models:
            active_model = self._active_models[model_name]
            
            # Cleanup provider resources
            if hasattr(active_model.provider, 'cleanup'):
                await active_model.provider.cleanup()
            
            del self._active_models[model_name]
            self.logger.info(f"Unloaded model: {model_name}")
    
    async def _try_fallback(
        self, 
        prompt: str, 
        classification: TaskClassification, 
        original_error: Exception
    ) -> GenerationResponse:
        """Try fallback models when primary model fails."""
        for fallback_model in self._fallback_models:
            try:
                self.logger.warning(f"Trying fallback model: {fallback_model}")
                
                provider = await self._get_provider(fallback_model)
                request = GenerationRequest(prompt=prompt, model=fallback_model)
                
                response = await provider.generate(request)
                self.logger.info(f"Fallback successful with {fallback_model}")
                return response
                
            except Exception as e:
                self.logger.error(f"Fallback model {fallback_model} also failed: {e}")
                continue
        
        # All fallbacks failed
        raise original_error
    
    async def _record_performance(
        self, 
        model_name: str, 
        task_type: TaskType, 
        response_time: float, 
        token_count: int
    ):
        """Record performance metrics for a model."""
        memory_usage = self._get_current_memory_usage()
        tokens_per_second = token_count / max(response_time / 1000, 0.001)
        
        performance = ModelPerformance(
            model_name=model_name,
            task_type=task_type,
            response_time=response_time,
            tokens_per_second=tokens_per_second,
            memory_usage=memory_usage,
            success_rate=1.0  # This request succeeded
        )
        
        self._performance_history.append(performance)
        
        # Keep only recent history
        if len(self._performance_history) > 1000:
            self._performance_history = self._performance_history[-1000:]
        
        # Update active model stats
        if model_name in self._active_models:
            active_model = self._active_models[model_name]
            active_model.request_count += 1
    
    def _get_average_response_time(self, model_name: str, task_type: TaskType) -> float:
        """Get average response time for a model and task type."""
        relevant_history = [
            p for p in self._performance_history 
            if p.model_name == model_name and p.task_type == task_type
        ]
        
        if not relevant_history:
            return 1000.0  # Default 1 second if no history
        
        return sum(p.response_time for p in relevant_history) / len(relevant_history)
    
    def _get_success_rate(self, model_name: str, task_type: TaskType) -> float:
        """Get success rate for a model and task type."""
        relevant_history = [
            p for p in self._performance_history 
            if p.model_name == model_name and p.task_type == task_type
        ]
        
        if not relevant_history:
            return 1.0  # Assume 100% if no history
        
        return sum(p.success_rate for p in relevant_history) / len(relevant_history)
    
    def _get_max_concurrent_models(self) -> int:
        """Get maximum number of concurrent models based on system resources."""
        return min(2, max(1, int(psutil.virtual_memory().available / (2 * 1024**3))))  # 2GB per model
    
    def _get_current_memory_usage(self) -> int:
        """Get current memory usage in MB."""
        return int(psutil.Process().memory_info().rss / 1024**2)
    
    # Public API methods
    
    def get_active_models(self) -> Dict[str, Dict[str, Any]]:
        """Get information about currently active models."""
        result = {}
        for model_name, active_model in self._active_models.items():
            result[model_name] = {
                'load_time': active_model.load_time,
                'last_used': active_model.last_used,
                'memory_usage': active_model.memory_usage,
                'request_count': active_model.request_count,
                'error_count': active_model.error_count
            }
        return result
    
    def get_model_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Get all model configurations."""
        result = {}
        for model_name, config in self._model_configs.items():
            result[model_name] = {
                'provider': config.provider,
                'task_types': [t.value for t in config.task_types],
                'max_complexity': config.max_complexity,
                'min_complexity': config.min_complexity,
                'priority': config.priority,
                'temperature': config.temperature,
                'max_tokens': config.max_tokens
            }
        return result
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self._performance_history:
            return {}
        
        stats = {}
        
        # Group by model and task type
        for model_name in set(p.model_name for p in self._performance_history):
            model_stats = {}
            model_history = [p for p in self._performance_history if p.model_name == model_name]
            
            for task_type in set(p.task_type for p in model_history):
                task_history = [p for p in model_history if p.task_type == task_type]
                
                model_stats[task_type.value] = {
                    'avg_response_time': sum(p.response_time for p in task_history) / len(task_history),
                    'avg_tokens_per_second': sum(p.tokens_per_second for p in task_history) / len(task_history),
                    'success_rate': sum(p.success_rate for p in task_history) / len(task_history),
                    'request_count': len(task_history)
                }
            
            stats[model_name] = model_stats
        
        return stats
    
    def set_selection_strategy(self, strategy: SelectionStrategy):
        """Set the model selection strategy."""
        self.selection_strategy = strategy
        self.logger.info(f"Model selection strategy set to: {strategy.value}")
    
    async def health_check_all_models(self) -> Dict[str, bool]:
        """Health check all configured models."""
        results = {}
        
        for model_name in self._model_configs.keys():
            try:
                provider = await self._get_provider(model_name)
                is_healthy = await provider.health_check()
                results[model_name] = is_healthy
            except Exception as e:
                self.logger.error(f"Health check failed for {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    async def cleanup_all_models(self):
        """Clean up all models and free resources."""
        model_names = list(self._active_models.keys())
        for model_name in model_names:
            await self._unload_model(model_name)
        
        self.logger.info("All models cleaned up")