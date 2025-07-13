"""
Execution planning component for the Loki Code agent.

Handles creating and managing execution plans for user requests.
"""

from typing import Dict, Any, List

from .types import RequestUnderstanding, ExecutionPlan, RequestContext
from .permission_manager import PermissionLevel
from ...utils.logging import get_logger


class ExecutionPlanner:
    """Creates and manages execution plans for agent requests."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = get_logger(__name__)
    
    async def create_execution_plan(
        self, 
        understanding: RequestUnderstanding, 
        context: RequestContext
    ) -> ExecutionPlan:
        """Create a detailed execution plan based on request understanding."""
        self.logger.debug(f"Creating execution plan for intent: {understanding.user_intent}")
        
        # Generate steps based on intent
        steps = self._generate_steps(understanding, context)
        
        # Estimate duration
        estimated_duration = self._estimate_duration(steps)
        
        # Identify required permissions
        required_permissions = self._identify_permissions(steps, understanding)
        
        # Assess safety considerations
        safety_considerations = self._assess_safety(steps, understanding)
        
        # Generate alternative approaches
        alternatives = self._generate_alternatives(understanding, context)
        
        plan = ExecutionPlan(
            steps=steps,
            estimated_duration=estimated_duration,
            required_permissions=required_permissions,
            safety_considerations=safety_considerations,
            alternative_approaches=alternatives
        )
        
        self.logger.info(f"Execution plan created with {len(steps)} steps")
        return plan
    
    def _generate_steps(self, understanding: RequestUnderstanding, context: RequestContext) -> List[Dict[str, Any]]:
        """Generate execution steps based on the request understanding."""
        steps = []
        intent = understanding.user_intent
        entities = understanding.extracted_entities
        
        if intent == "read_file":
            steps.extend(self._create_file_read_steps(entities))
        elif intent == "write_file":
            steps.extend(self._create_file_write_steps(entities, understanding))
        elif intent == "analyze_code":
            steps.extend(self._create_code_analysis_steps(entities))
        elif intent == "search":
            steps.extend(self._create_search_steps(entities, understanding))
        elif intent == "question":
            steps.extend(self._create_question_steps(understanding))
        else:
            steps.extend(self._create_general_steps(understanding, context))
        
        # Add common final steps
        steps.append({
            "type": "generate_response",
            "description": "Generate response based on results",
            "tool": "llm_generator",
            "estimated_time": 2.0,
            "permission_level": "none"
        })
        
        return steps
    
    def _create_file_read_steps(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create steps for file reading operations."""
        steps = []
        files = entities.get("files", [])
        
        if not files:
            steps.append({
                "type": "clarification",
                "description": "Ask user to specify which file to read",
                "tool": "user_interaction",
                "estimated_time": 0.5,
                "permission_level": "none"
            })
        else:
            for file_path in files:
                steps.append({
                    "type": "file_operation",
                    "description": f"Read file: {file_path}",
                    "tool": "file_reader",
                    "input": {"file_path": file_path, "analysis_level": "standard"},
                    "estimated_time": 1.5,
                    "permission_level": "read"
                })
        
        return steps
    
    def _create_file_write_steps(self, entities: Dict[str, Any], understanding: RequestUnderstanding) -> List[Dict[str, Any]]:
        """Create steps for file writing operations."""
        steps = []
        files = entities.get("files", [])
        
        # First, read existing files if specified
        for file_path in files:
            steps.append({
                "type": "file_operation",
                "description": f"Read existing file: {file_path}",
                "tool": "file_reader",
                "input": {"file_path": file_path, "analysis_level": "minimal"},
                "estimated_time": 1.0,
                "permission_level": "read"
            })
        
        # Then plan the write operation
        steps.append({
            "type": "content_generation",
            "description": "Generate content for file modification",
            "tool": "llm_generator",
            "estimated_time": 3.0,
            "permission_level": "none"
        })
        
        # Finally, write the files
        for file_path in files:
            steps.append({
                "type": "file_operation",
                "description": f"Write to file: {file_path}",
                "tool": "file_writer",
                "input": {"file_path": file_path},
                "estimated_time": 1.0,
                "permission_level": "write"
            })
        
        return steps
    
    def _create_code_analysis_steps(self, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create steps for code analysis operations."""
        steps = []
        files = entities.get("files", [])
        
        if files:
            for file_path in files:
                steps.append({
                    "type": "file_operation",
                    "description": f"Analyze code structure: {file_path}",
                    "tool": "file_reader",
                    "input": {
                        "file_path": file_path, 
                        "analysis_level": "comprehensive",
                        "include_context": True
                    },
                    "estimated_time": 2.5,
                    "permission_level": "read"
                })
        else:
            # Analyze current project or directory
            steps.append({
                "type": "project_analysis",
                "description": "Analyze project structure",
                "tool": "project_analyzer",
                "estimated_time": 5.0,
                "permission_level": "read"
            })
        
        steps.append({
            "type": "analysis_summary",
            "description": "Generate analysis summary and insights",
            "tool": "llm_generator",
            "estimated_time": 2.0,
            "permission_level": "none"
        })
        
        return steps
    
    def _create_search_steps(self, entities: Dict[str, Any], understanding: RequestUnderstanding) -> List[Dict[str, Any]]:
        """Create steps for search operations."""
        steps = []
        
        # Determine search scope
        if "project_path" in entities:
            steps.append({
                "type": "project_search",
                "description": "Search within project",
                "tool": "project_search",
                "input": {"scope": "project"},
                "estimated_time": 3.0,
                "permission_level": "read"
            })
        else:
            steps.append({
                "type": "file_search",
                "description": "Search in current directory",
                "tool": "file_search",
                "input": {"scope": "directory"},
                "estimated_time": 2.0,
                "permission_level": "read"
            })
        
        return steps
    
    def _create_question_steps(self, understanding: RequestUnderstanding) -> List[Dict[str, Any]]:
        """Create steps for question/explanation requests."""
        steps = []
        
        # Check if we need to read any code for context
        if understanding.extracted_entities.get("files"):
            steps.extend(self._create_file_read_steps(understanding.extracted_entities))
        
        steps.append({
            "type": "explanation",
            "description": "Generate explanation or answer",
            "tool": "llm_generator", 
            "estimated_time": 2.0,
            "permission_level": "none"
        })
        
        return steps
    
    def _create_general_steps(self, understanding: RequestUnderstanding, context: RequestContext) -> List[Dict[str, Any]]:
        """Create steps for general assistance requests."""
        return [{
            "type": "general_assistance",
            "description": "Provide general assistance",
            "tool": "llm_generator",
            "estimated_time": 2.0,
            "permission_level": "none"
        }]
    
    def _estimate_duration(self, steps: List[Dict[str, Any]]) -> float:
        """Estimate total execution duration for the plan."""
        total_time = sum(step.get("estimated_time", 1.0) for step in steps)
        
        # Add overhead for coordination and error handling
        overhead = len(steps) * 0.5
        
        return total_time + overhead
    
    def _identify_permissions(self, steps: List[Dict[str, Any]], understanding: RequestUnderstanding) -> List[str]:
        """Identify required permissions for the execution plan."""
        permissions = set()
        
        for step in steps:
            perm_level = step.get("permission_level", "none")
            if perm_level != "none":
                permissions.add(perm_level)
        
        # Add special permissions based on risk assessment
        if understanding.risk_assessment == "high":
            permissions.add("admin_approval")
        elif understanding.risk_assessment == "medium":
            permissions.add("user_confirmation")
        
        return list(permissions)
    
    def _assess_safety(self, steps: List[Dict[str, Any]], understanding: RequestUnderstanding) -> List[str]:
        """Assess safety considerations for the execution plan."""
        considerations = []
        
        # Check for write operations
        write_steps = [s for s in steps if s.get("permission_level") == "write"]
        if write_steps:
            considerations.append("File modification operations require user approval")
        
        # Check for high-risk operations
        if understanding.risk_assessment == "high":
            considerations.append("High-risk operation detected - extra caution required")
        
        # Check for multiple file operations
        file_ops = [s for s in steps if s.get("type") == "file_operation"]
        if len(file_ops) > 3:
            considerations.append("Multiple file operations - review scope carefully")
        
        # Check for external tool usage
        external_tools = [s for s in steps if s.get("tool", "").startswith("external_")]
        if external_tools:
            considerations.append("External tool usage - verify tool safety")
        
        return considerations
    
    def _generate_alternatives(self, understanding: RequestUnderstanding, context: RequestContext) -> List[str]:
        """Generate alternative approaches for the request."""
        alternatives = []
        intent = understanding.user_intent
        
        if intent == "read_file":
            alternatives.extend([
                "Use a simpler text viewer for basic content",
                "Read file in chunks if it's very large",
                "Use grep/search to find specific content"
            ])
        elif intent == "write_file":
            alternatives.extend([
                "Create a backup before modification",
                "Use a dry-run mode to preview changes",
                "Apply changes incrementally with review"
            ])
        elif intent == "analyze_code":
            alternatives.extend([
                "Focus analysis on specific functions/classes",
                "Use lightweight analysis for quick overview",
                "Generate documentation instead of analysis"
            ])
        
        return alternatives