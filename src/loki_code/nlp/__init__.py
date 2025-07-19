"""
Natural Language Processing module for Loki Code.

Provides spaCy-based intent classification and entity extraction
for robust command understanding.
"""

from .intent_classifier import IntentClassifier
from .entity_extractor import EntityExtractor  
from .command_analyzer import CommandAnalyzer
from .pipeline_factory import PipelineFactory