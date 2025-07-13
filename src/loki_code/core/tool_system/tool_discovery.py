"""
Tool discovery functionality.

Handles finding and loading tools from various sources.
"""

import inspect
import importlib
import pkgutil
from pathlib import Path
from typing import List, Type, Dict, Any

from ...tools.base import BaseTool
from ...utils.logging import get_logger


class ToolDiscovery:
    """Discovers and loads tools from various sources."""
    
    def __init__(self, registry: 'ToolRegistry'):
        self.registry = registry
        self.logger = get_logger(__name__)
    
    def discover_builtin_tools(self) -> List[Type[BaseTool]]:
        """Discover built-in tools from the tools package."""
        tools = []
        
        try:
            # Import tools package
            import loki_code.tools as tools_package
            tools_path = Path(tools_package.__file__).parent
            
            # Scan for tool modules
            for module_info in pkgutil.iter_modules([str(tools_path)]):
                if module_info.name.startswith('_'):
                    continue
                
                try:
                    module = importlib.import_module(f'loki_code.tools.{module_info.name}')
                    tool_classes = self._extract_tool_classes_from_module(module)
                    tools.extend(tool_classes)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to load tools from {module_info.name}: {e}")
            
            self.logger.info(f"Discovered {len(tools)} built-in tools")
            
        except Exception as e:
            self.logger.error(f"Failed to discover built-in tools: {e}")
        
        return tools
    
    def _extract_tool_classes_from_module(self, module) -> List[Type[BaseTool]]:
        """Extract tool classes from a module."""
        tool_classes = []
        
        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Check if it's a tool class (subclass of BaseTool, not BaseTool itself)
            if (issubclass(obj, BaseTool) and 
                obj is not BaseTool and 
                not inspect.isabstract(obj)):
                tool_classes.append(obj)
        
        return tool_classes
    
    def load_plugin_tools(self, plugin_dirs: List[str]) -> List[Type[BaseTool]]:
        """Load tools from plugin directories."""
        tools = []
        
        for plugin_dir in plugin_dirs:
            try:
                plugin_path = Path(plugin_dir)
                if not plugin_path.exists():
                    self.logger.warning(f"Plugin directory does not exist: {plugin_dir}")
                    continue
                
                # Scan for Python files in plugin directory
                for py_file in plugin_path.glob("*.py"):
                    if py_file.name.startswith('_'):
                        continue
                    
                    try:
                        tool_classes = self._load_tools_from_file(py_file)
                        tools.extend(tool_classes)
                        
                    except Exception as e:
                        self.logger.warning(f"Failed to load tools from {py_file}: {e}")
                
            except Exception as e:
                self.logger.error(f"Failed to scan plugin directory {plugin_dir}: {e}")
        
        self.logger.info(f"Loaded {len(tools)} plugin tools")
        return tools
    
    def _load_tools_from_file(self, file_path: Path) -> List[Type[BaseTool]]:
        """Load tool classes from a Python file."""
        import importlib.util
        
        # Load module from file
        spec = importlib.util.spec_from_file_location(file_path.stem, file_path)
        if not spec or not spec.loader:
            return []
        
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Extract tool classes
        return self._extract_tool_classes_from_module(module)
    
    def auto_discover_and_register(self, include_plugins: bool = False, plugin_dirs: List[str] = None) -> int:
        """Automatically discover and register all available tools."""
        registered_count = 0
        
        # Discover built-in tools
        builtin_tools = self.discover_builtin_tools()
        for tool_class in builtin_tools:
            try:
                self.registry.register_tool_class(tool_class, source="builtin")
                registered_count += 1
            except Exception as e:
                self.logger.warning(f"Failed to register built-in tool {tool_class.__name__}: {e}")
        
        # Discover plugin tools if requested
        if include_plugins and plugin_dirs:
            plugin_tools = self.load_plugin_tools(plugin_dirs)
            for tool_class in plugin_tools:
                try:
                    self.registry.register_tool_class(tool_class, source="plugin")
                    registered_count += 1
                except Exception as e:
                    self.logger.warning(f"Failed to register plugin tool {tool_class.__name__}: {e}")
        
        self.logger.info(f"Auto-discovery completed: {registered_count} tools registered")
        return registered_count
    
    def validate_tool_class(self, tool_class: Type[BaseTool]) -> List[str]:
        """Validate a tool class and return any issues found."""
        issues = []
        
        # Check if it's a proper subclass
        if not issubclass(tool_class, BaseTool):
            issues.append("Not a subclass of BaseTool")
        
        # Check if it's not abstract
        if inspect.isabstract(tool_class):
            issues.append("Tool class is abstract")
        
        # Check for required methods
        required_methods = ['get_schema', 'execute']
        for method_name in required_methods:
            if not hasattr(tool_class, method_name):
                issues.append(f"Missing required method: {method_name}")
        
        # Try to instantiate to check for obvious issues
        try:
            instance = tool_class()
            schema = instance.get_schema()
            if not schema:
                issues.append("get_schema() returns None")
            elif not hasattr(schema, 'name') or not schema.name:
                issues.append("Schema missing or has no name")
        except Exception as e:
            issues.append(f"Failed to instantiate or get schema: {e}")
        
        return issues
    
    def scan_for_mcp_servers(self, config: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Scan for available MCP servers (placeholder for future implementation)."""
        # This would implement MCP server discovery
        # For now, return empty list
        self.logger.debug("MCP server discovery not yet implemented")
        return []