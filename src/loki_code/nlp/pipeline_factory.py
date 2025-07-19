"""
spaCy pipeline factory for creating and managing NLP pipelines.
"""

import spacy
from spacy.language import Language
from typing import Dict, Any, Optional, List
import os
from pathlib import Path

from .command_analyzer import CommandAnalyzer
from ..utils.logging import get_logger


class PipelineFactory:
    """Factory for creating and managing spaCy NLP pipelines."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self._pipelines: Dict[str, Language] = {}
    
    def create_pipeline(self, config: Dict[str, Any]) -> Language:
        """Create a spaCy pipeline based on configuration."""
        pipeline_name = config.get("name", "default")
        base_model = config.get("base_model", "en_core_web_sm")
        components = config.get("components", [])
        
        try:
            # Load base model
            nlp = spacy.load(base_model)
            
            # Add custom components
            for component_config in components:
                self._add_component(nlp, component_config)
            
            # Cache pipeline
            self._pipelines[pipeline_name] = nlp
            
            self.logger.info(f"Created pipeline '{pipeline_name}' with components: {nlp.pipe_names}")
            return nlp
            
        except Exception as e:
            self.logger.error(f"Failed to create pipeline '{pipeline_name}': {e}")
            raise
    
    def get_pipeline(self, name: str) -> Optional[Language]:
        """Get a cached pipeline by name."""
        return self._pipelines.get(name)
    
    def create_command_analyzer_pipeline(self) -> CommandAnalyzer:
        """Create a specialized pipeline for command analysis."""
        config = {
            "name": "command_analyzer",
            "base_model": "en_core_web_sm",
            "components": [
                {
                    "name": "intent_classifier",
                    "type": "custom",
                    "config": {}
                },
                {
                    "name": "entity_extractor",
                    "type": "custom", 
                    "config": {}
                }
            ]
        }
        
        # Create and return command analyzer
        return CommandAnalyzer()
    
    def _add_component(self, nlp: Language, component_config: Dict[str, Any]) -> None:
        """Add a component to the spaCy pipeline."""
        component_name = component_config["name"]
        component_type = component_config.get("type", "builtin")
        config = component_config.get("config", {})
        
        if component_type == "custom":
            # Add custom components
            if component_name == "intent_classifier":
                self._add_intent_classifier(nlp, config)
            elif component_name == "entity_extractor":
                self._add_entity_extractor(nlp, config)
            else:
                self.logger.warning(f"Unknown custom component: {component_name}")
        
        elif component_type == "builtin":
            # Add built-in spaCy components
            if component_name not in nlp.pipe_names:
                nlp.add_pipe(component_name, config=config)
    
    def _add_intent_classifier(self, nlp: Language, config: Dict[str, Any]) -> None:
        """Add intent classifier component to pipeline."""
        if "intent_classifier" not in nlp.pipe_names:
            # Register the component if not already registered
            if not spacy.util.registry.has("components", "intent_classifier"):
                from .intent_classifier import create_intent_classifier
                spacy.registry.components("intent_classifier")(create_intent_classifier)
            
            nlp.add_pipe("intent_classifier", config=config)
    
    def _add_entity_extractor(self, nlp: Language, config: Dict[str, Any]) -> None:
        """Add entity extractor component to pipeline."""
        # Entity extraction is handled separately in EntityExtractor class
        # This is a placeholder for future spaCy component integration
        pass
    
    def list_available_models(self) -> List[str]:
        """List available spaCy models."""
        try:
            import subprocess
            result = subprocess.run(
                ["python", "-m", "spacy", "info"], 
                capture_output=True, 
                text=True
            )
            
            # Parse available models from spacy info output
            models = []
            if result.returncode == 0:
                # This is a simplified parser - in practice you'd want more robust parsing
                for line in result.stdout.split('\n'):
                    if 'en_core_web' in line or 'en_' in line:
                        models.append(line.strip())
            
            return models
            
        except Exception as e:
            self.logger.error(f"Failed to list available models: {e}")
            return ["en_core_web_sm"]  # Default fallback
    
    def validate_pipeline_config(self, config: Dict[str, Any]) -> bool:
        """Validate pipeline configuration."""
        required_fields = ["name", "base_model"]
        
        for field in required_fields:
            if field not in config:
                self.logger.error(f"Missing required field in pipeline config: {field}")
                return False
        
        # Validate base model exists
        base_model = config["base_model"]
        try:
            spacy.load(base_model)
        except Exception as e:
            self.logger.error(f"Cannot load base model '{base_model}': {e}")
            return False
        
        # Validate components
        components = config.get("components", [])
        for component in components:
            if "name" not in component:
                self.logger.error("Component missing 'name' field")
                return False
        
        return True
    
    def get_default_config(self) -> Dict[str, Any]:
        """Get default pipeline configuration."""
        return {
            "name": "default_command_pipeline",
            "base_model": "en_core_web_sm",
            "components": [
                {
                    "name": "intent_classifier",
                    "type": "custom",
                    "config": {
                        "confidence_threshold": 0.7
                    }
                }
            ],
            "settings": {
                "max_length": 1000000,  # Maximum document length
                "disable": [],  # Components to disable
            }
        }