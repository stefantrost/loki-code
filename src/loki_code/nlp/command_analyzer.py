"""
Main command analyzer that orchestrates intent classification and entity extraction.
"""

import spacy
from typing import Dict, List, Optional, Any

from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor
from ..core.commands.types import ParsedInput, ConversationContext, Intent
from ..utils.logging import get_logger


class CommandAnalyzer:
    """Main orchestrator for NLP-based command analysis."""
    
    def __init__(self, model_name: str = "en_core_web_sm"):
        self.logger = get_logger(__name__)
        self.nlp = spacy.load(model_name)
        self.intent_classifier = IntentClassifier()
        self.entity_extractor = EntityExtractor(self.nlp)
    
    def analyze_command(self, user_input: str, context: ConversationContext) -> ParsedInput:
        """Analyze user command and return parsed input."""
        try:
            # Clean and preprocess input
            cleaned_text = self._preprocess_text(user_input)
            
            # Classify intent
            intent = self.intent_classifier.classify_intent(cleaned_text)
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(cleaned_text)
            
            # Calculate overall confidence
            overall_confidence = self._calculate_confidence(intent, entities, cleaned_text)
            
            # Create parsed input
            parsed_input = ParsedInput(
                original_text=user_input,
                cleaned_text=cleaned_text,
                intent=intent,
                entities=entities,
                confidence=overall_confidence,
                metadata={
                    "context": context,
                    "processing_method": "spacy_nlp",
                    "model_version": "1.0.0"
                }
            )
            
            self.logger.debug(f"Analyzed command: intent={intent.type.value}, confidence={overall_confidence:.2f}")
            return parsed_input
            
        except Exception as e:
            self.logger.error(f"Command analysis failed: {e}")
            # Return fallback parsed input
            return self._create_fallback_parsed_input(user_input, str(e))
    
    def _preprocess_text(self, text: str) -> str:
        """Clean and preprocess user input text."""
        # Remove extra whitespace
        cleaned = " ".join(text.strip().split())
        
        # Handle common contractions and abbreviations
        replacements = {
            "i'm": "i am",
            "i'd": "i would",
            "i'll": "i will",
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "doesn't": "does not",
            "isn't": "is not",
            "aren't": "are not",
            "wasn't": "was not",
            "weren't": "were not",
            "hasn't": "has not",
            "haven't": "have not",
            "hadn't": "had not",
            "wouldn't": "would not",
            "shouldn't": "should not",
            "couldn't": "could not",
            "mustn't": "must not",
        }
        
        for contraction, expansion in replacements.items():
            cleaned = cleaned.replace(contraction, expansion)
        
        return cleaned
    
    def _calculate_confidence(self, intent: Intent, entities: Dict[str, List[str]], text: str) -> float:
        """Calculate overall confidence score for the parsed input."""
        # Start with intent confidence
        confidence = intent.confidence
        
        # Boost confidence if we found relevant entities
        entity_boost = 0.0
        total_entities = sum(len(entity_list) for entity_list in entities.values())
        
        if total_entities > 0:
            entity_boost = min(0.1 * total_entities, 0.2)  # Max 0.2 boost
        
        # Consider text length and complexity
        length_factor = 1.0
        if len(text.split()) <= 2:
            length_factor = 0.9  # Short commands might be ambiguous
        elif len(text.split()) > 20:
            length_factor = 0.95  # Very long commands might be complex
        
        # Check for specific indicators
        specific_indicators = 0
        if any(entities["files"]):
            specific_indicators += 1
        if any(entities["functions"]) or any(entities["classes"]):
            specific_indicators += 1
        if any(entities["languages"]):
            specific_indicators += 1
        
        indicator_boost = min(0.05 * specific_indicators, 0.15)
        
        # Calculate final confidence
        final_confidence = min((confidence + entity_boost + indicator_boost) * length_factor, 1.0)
        
        return max(final_confidence, 0.1)  # Minimum confidence of 0.1
    
    def _create_fallback_parsed_input(self, original_text: str, error_msg: str) -> ParsedInput:
        """Create a fallback parsed input when analysis fails."""
        from ..core.commands.types import IntentType, Intent
        
        return ParsedInput(
            original_text=original_text,
            cleaned_text=original_text.strip(),
            intent=Intent(
                type=IntentType.CONVERSATION,
                confidence=0.3,
                reasoning=f"Analysis failed: {error_msg}"
            ),
            entities={
                "files": [],
                "functions": [],
                "classes": [],
                "languages": [],
                "create_targets": [],
                "directories": [],
            },
            confidence=0.3,
            metadata={
                "error": error_msg,
                "fallback": True
            }
        )
    
    def get_analysis_stats(self) -> Dict[str, Any]:
        """Get analysis statistics and model information."""
        return {
            "model_name": self.nlp.meta.get("name", "unknown"),
            "model_version": self.nlp.meta.get("version", "unknown"),
            "pipeline_components": self.nlp.pipe_names,
            "vocab_size": len(self.nlp.vocab),
            "intent_classifier_loaded": self.intent_classifier is not None,
            "entity_extractor_loaded": self.entity_extractor is not None,
        }