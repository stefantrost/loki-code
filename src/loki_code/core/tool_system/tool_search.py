"""
Tool search and filtering functionality.

Extracted from tool_registry.py for better organization.
"""

from typing import List, Set
import re

from ...tools.types import ToolSchema, ToolCapability, SecurityLevel
from .tool_types import ToolFilter
from ...utils.logging import get_logger


class ToolSearch:
    """Provides search and filtering capabilities for tools."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    @staticmethod
    def filter_tools(tools: List[ToolSchema], filter_obj: ToolFilter) -> List[ToolSchema]:
        """Filter tools based on filter criteria."""
        filtered = tools
        
        # Filter by capabilities
        if filter_obj.capabilities:
            filtered = [
                tool for tool in filtered
                if any(cap in tool.capabilities for cap in filter_obj.capabilities)
            ]
        
        # Filter by security levels
        if filter_obj.security_levels:
            filtered = [
                tool for tool in filtered
                if tool.security_level in filter_obj.security_levels
            ]
        
        # Filter by MCP compatibility
        if filter_obj.mcp_compatible is not None:
            filtered = [
                tool for tool in filtered
                if tool.mcp_compatible == filter_obj.mcp_compatible
            ]
        
        # Filter by keywords
        if filter_obj.keywords:
            keyword_set = set(keyword.lower() for keyword in filter_obj.keywords)
            filtered = [
                tool for tool in filtered
                if ToolSearch._matches_keywords(tool, keyword_set)
            ]
        
        return filtered
    
    @staticmethod
    def _matches_keywords(tool: ToolSchema, keywords: Set[str]) -> bool:
        """Check if tool matches any of the keywords."""
        searchable_text = (
            tool.name + " " + 
            tool.description + " " + 
            " ".join(tool.tags)
        ).lower()
        
        return any(keyword in searchable_text for keyword in keywords)
    
    @staticmethod
    def search_by_description(tools: List[ToolSchema], query: str) -> List[ToolSchema]:
        """Search tools by description content."""
        if not query:
            return tools
        
        query_lower = query.lower()
        query_words = query_lower.split()
        
        results = []
        for tool in tools:
            score = ToolSearch._calculate_relevance_score(tool, query_words)
            if score > 0:
                results.append((tool, score))
        
        # Sort by relevance score (descending)
        results.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, _ in results]
    
    @staticmethod
    def _calculate_relevance_score(tool: ToolSchema, query_words: List[str]) -> float:
        """Calculate relevance score for a tool against query words."""
        score = 0.0
        
        # Check name (highest weight)
        name_lower = tool.name.lower()
        for word in query_words:
            if word in name_lower:
                score += 3.0
                if name_lower.startswith(word):
                    score += 2.0  # Bonus for prefix match
        
        # Check description (medium weight)
        desc_lower = tool.description.lower()
        for word in query_words:
            if word in desc_lower:
                score += 1.0
        
        # Check tags (lower weight)
        tags_text = " ".join(tool.tags).lower()
        for word in query_words:
            if word in tags_text:
                score += 0.5
        
        # Check capabilities (lower weight)
        capabilities_text = " ".join(cap.value for cap in tool.capabilities).lower()
        for word in query_words:
            if word in capabilities_text:
                score += 0.3
        
        return score
    
    @staticmethod
    def rank_tools_by_relevance(tools: List[ToolSchema], task_description: str) -> List[ToolSchema]:
        """Rank tools by relevance to a task description."""
        if not task_description:
            return tools
        
        # Extract key terms from task description
        task_lower = task_description.lower()
        
        # Simple keyword extraction (could be enhanced with NLP)
        important_words = [
            "read", "write", "create", "delete", "analyze", "search", 
            "file", "code", "text", "data", "format", "convert"
        ]
        
        found_keywords = [word for word in important_words if word in task_lower]
        
        # Use existing search functionality
        return ToolSearch.search_by_description(tools, " ".join(found_keywords))
    
    @staticmethod
    def find_tools_by_capability(tools: List[ToolSchema], capability: ToolCapability) -> List[ToolSchema]:
        """Find all tools that have a specific capability."""
        return [tool for tool in tools if capability in tool.capabilities]
    
    @staticmethod
    def find_tools_by_security_level(tools: List[ToolSchema], max_level: SecurityLevel) -> List[ToolSchema]:
        """Find tools with security level at or below the specified level."""
        security_order = {
            SecurityLevel.SAFE: 0,
            SecurityLevel.CAUTION: 1,
            SecurityLevel.MODERATE: 2,
            SecurityLevel.DANGEROUS: 3,
            SecurityLevel.CRITICAL: 4
        }
        
        max_level_value = security_order.get(max_level, 0)
        
        return [
            tool for tool in tools
            if security_order.get(tool.security_level, 5) <= max_level_value
        ]
    
    @staticmethod
    def group_tools_by_category(tools: List[ToolSchema]) -> dict[str, List[ToolSchema]]:
        """Group tools by their primary capability."""
        categories = {}
        
        for tool in tools:
            # Use the first capability as primary category
            if tool.capabilities:
                category = tool.capabilities[0].value
            else:
                category = "uncategorized"
            
            if category not in categories:
                categories[category] = []
            categories[category].append(tool)
        
        return categories