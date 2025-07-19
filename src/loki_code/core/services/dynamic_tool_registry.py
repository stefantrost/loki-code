"""
Dynamic Tool Registry Service - Hot-reload and watch capabilities.

Extends the core tool registry with dynamic registration features:
- Hot-reload of tool modules
- File system watching for automatic updates
- Runtime tool registration/unregistration
- Plugin management
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Callable

# Optional watchdog import
try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler, FileModifiedEvent
    WATCHDOG_AVAILABLE = True
except ImportError:
    Observer = None
    FileSystemEventHandler = None
    FileModifiedEvent = None
    WATCHDOG_AVAILABLE = False

from ..tool_system.tool_registry_core import get_global_registry, ToolRegistry
from ...tools.base import BaseTool
from ...utils.logging import get_logger


class ToolModuleWatcher(FileSystemEventHandler if WATCHDOG_AVAILABLE else object):
    """File system watcher for tool modules."""
    
    def __init__(self, registry_service: 'DynamicToolRegistry'):
        """Initialize the watcher."""
        self.registry_service = registry_service
        self.logger = get_logger(__name__)
        self._debounce_delay = 1.0  # Seconds to wait before processing changes
        self._pending_changes: Dict[str, float] = {}
        self._processing_lock = asyncio.Lock()
    
    def on_modified(self, event):
        """Handle file modification events."""
        if event.is_directory:
            return
        
        # Only watch Python files
        if not event.src_path.endswith('.py'):
            return
        
        # Debounce rapid changes
        current_time = time.time()
        self._pending_changes[event.src_path] = current_time
        
        # Schedule processing after debounce delay
        asyncio.create_task(self._process_change_after_delay(event.src_path, current_time))
    
    async def _process_change_after_delay(self, file_path: str, change_time: float):
        """Process file change after debounce delay."""
        await asyncio.sleep(self._debounce_delay)
        
        # Check if this is still the latest change for this file
        if self._pending_changes.get(file_path) != change_time:
            return  # Newer change detected, skip this one
        
        async with self._processing_lock:
            try:
                await self.registry_service.reload_tool_module(file_path)
                self._pending_changes.pop(file_path, None)
            except Exception as e:
                self.logger.error(f"Failed to reload tool module {file_path}: {e}")


class DynamicToolRegistry:
    """
    Enhanced tool registry with dynamic capabilities.
    
    Features:
    - Hot-reload of tool modules
    - File system watching
    - Runtime registration/unregistration
    - Plugin management
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dynamic tool registry."""
        self.config = config or {}
        self.logger = get_logger(__name__)
        self.registry = get_global_registry()
        
        # File watching
        self._observer: Optional[Observer] = None
        self._watched_directories: Set[str] = set()
        self._watcher = ToolModuleWatcher(self)
        
        # Module tracking
        self._module_to_tools: Dict[str, List[str]] = {}
        self._tool_to_module: Dict[str, str] = {}
        
        # Callbacks
        self._registration_callbacks: List[Callable[[str, bool], None]] = []
        
        self.logger.info("DynamicToolRegistry initialized")
    
    def add_registration_callback(self, callback: Callable[[str, bool], None]):
        """
        Add callback for tool registration events.
        
        Args:
            callback: Function that takes (tool_name, is_registered) arguments
        """
        self._registration_callbacks.append(callback)
    
    def _notify_registration_change(self, tool_name: str, is_registered: bool):
        """Notify callbacks of registration changes."""
        for callback in self._registration_callbacks:
            try:
                callback(tool_name, is_registered)
            except Exception as e:
                self.logger.error(f"Registration callback failed: {e}")
    
    async def register_tool_from_module(
        self,
        module_path: str,
        tool_class_name: Optional[str] = None
    ) -> List[str]:
        """
        Register tool(s) from a Python module.
        
        Args:
            module_path: Path to Python module file
            tool_class_name: Specific tool class name (if None, registers all found)
            
        Returns:
            List of registered tool names
        """
        import importlib.util
        import inspect
        
        registered_tools = []
        
        try:
            # Load module from file
            spec = importlib.util.spec_from_file_location("dynamic_tool", module_path)
            if not spec or not spec.loader:
                raise ValueError(f"Cannot load module from {module_path}")
            
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            
            # Find tool classes in module
            tool_classes = []
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (issubclass(obj, BaseTool) and 
                    obj is not BaseTool and 
                    not inspect.isabstract(obj)):
                    
                    if tool_class_name is None or name == tool_class_name:
                        tool_classes.append((name, obj))
            
            # Register found tool classes
            for class_name, tool_class in tool_classes:
                try:
                    self.registry.register_tool_class(tool_class, source="dynamic")
                    
                    # Track module association
                    tool_name = tool_class().get_schema().name
                    self._tool_to_module[tool_name] = module_path
                    
                    if module_path not in self._module_to_tools:
                        self._module_to_tools[module_path] = []
                    self._module_to_tools[module_path].append(tool_name)
                    
                    registered_tools.append(tool_name)
                    self._notify_registration_change(tool_name, True)
                    
                    self.logger.info(f"Registered dynamic tool: {tool_name} from {module_path}")
                    
                except Exception as e:
                    self.logger.error(f"Failed to register tool {class_name}: {e}")
            
            if not registered_tools:
                self.logger.warning(f"No valid tool classes found in {module_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to load tools from {module_path}: {e}")
            raise
        
        return registered_tools
    
    async def unregister_tool_by_name(self, tool_name: str) -> bool:
        """
        Unregister a tool by name.
        
        Args:
            tool_name: Name of the tool to unregister
            
        Returns:
            True if tool was unregistered, False if not found
        """
        if self.registry.unregister_tool(tool_name):
            # Clean up tracking
            module_path = self._tool_to_module.pop(tool_name, None)
            if module_path and module_path in self._module_to_tools:
                self._module_to_tools[module_path].remove(tool_name)
                if not self._module_to_tools[module_path]:
                    del self._module_to_tools[module_path]
            
            self._notify_registration_change(tool_name, False)
            self.logger.info(f"Unregistered tool: {tool_name}")
            return True
        
        return False
    
    async def reload_tool_module(self, module_path: str) -> List[str]:
        """
        Reload tools from a modified module.
        
        Args:
            module_path: Path to the modified module
            
        Returns:
            List of reloaded tool names
        """
        self.logger.info(f"Reloading tool module: {module_path}")
        
        # Unregister existing tools from this module
        existing_tools = self._module_to_tools.get(module_path, [])
        for tool_name in existing_tools[:]:  # Copy list to avoid modification during iteration
            await self.unregister_tool_by_name(tool_name)
        
        # Re-register tools from module
        try:
            reloaded_tools = await self.register_tool_from_module(module_path)
            self.logger.info(f"Reloaded {len(reloaded_tools)} tools from {module_path}")
            return reloaded_tools
        except Exception as e:
            self.logger.error(f"Failed to reload module {module_path}: {e}")
            return []
    
    def start_watching(self, directories: List[str]) -> bool:
        """
        Start watching directories for tool module changes.
        
        Args:
            directories: List of directory paths to watch
            
        Returns:
            True if watching started successfully
        """
        if not WATCHDOG_AVAILABLE:
            self.logger.warning("Watchdog not available - file watching disabled")
            return False
            
        try:
            if self._observer is None:
                self._observer = Observer()
            
            for directory in directories:
                dir_path = Path(directory)
                if dir_path.exists() and dir_path.is_dir():
                    self._observer.schedule(self._watcher, str(dir_path), recursive=True)
                    self._watched_directories.add(directory)
                    self.logger.info(f"Started watching directory: {directory}")
                else:
                    self.logger.warning(f"Directory does not exist: {directory}")
            
            if not self._observer.is_alive():
                self._observer.start()
                self.logger.info("File system watcher started")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start file watching: {e}")
            return False
    
    def stop_watching(self):
        """Stop watching for file changes."""
        if self._observer and self._observer.is_alive():
            self._observer.stop()
            self._observer.join()
            self._watched_directories.clear()
            self.logger.info("File system watcher stopped")
    
    def install_plugin(self, plugin_path: str) -> List[str]:
        """
        Install a plugin from a directory or file.
        
        Args:
            plugin_path: Path to plugin directory or Python file
            
        Returns:
            List of installed tool names
        """
        plugin_path_obj = Path(plugin_path)
        installed_tools = []
        
        if plugin_path_obj.is_file() and plugin_path_obj.suffix == '.py':
            # Single Python file
            tools = asyncio.run(self.register_tool_from_module(str(plugin_path_obj)))
            installed_tools.extend(tools)
        
        elif plugin_path_obj.is_dir():
            # Plugin directory
            for py_file in plugin_path_obj.glob("*.py"):
                if not py_file.name.startswith('_'):
                    tools = asyncio.run(self.register_tool_from_module(str(py_file)))
                    installed_tools.extend(tools)
        
        else:
            raise ValueError(f"Invalid plugin path: {plugin_path}")
        
        self.logger.info(f"Installed plugin with {len(installed_tools)} tools: {installed_tools}")
        return installed_tools
    
    def uninstall_plugin(self, plugin_identifier: str) -> List[str]:
        """
        Uninstall a plugin by path or tool name pattern.
        
        Args:
            plugin_identifier: Plugin path or tool name pattern
            
        Returns:
            List of uninstalled tool names
        """
        uninstalled_tools = []
        
        # Check if it's a module path
        if plugin_identifier in self._module_to_tools:
            tools_to_remove = self._module_to_tools[plugin_identifier][:]
            for tool_name in tools_to_remove:
                if asyncio.run(self.unregister_tool_by_name(tool_name)):
                    uninstalled_tools.append(tool_name)
        
        # Check if it's a tool name pattern
        else:
            matching_tools = [
                name for name in self.registry.list_tool_names()
                if plugin_identifier in name
            ]
            for tool_name in matching_tools:
                if asyncio.run(self.unregister_tool_by_name(tool_name)):
                    uninstalled_tools.append(tool_name)
        
        self.logger.info(f"Uninstalled {len(uninstalled_tools)} tools: {uninstalled_tools}")
        return uninstalled_tools
    
    def get_module_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about loaded modules and their tools."""
        module_info = {}
        
        for module_path, tool_names in self._module_to_tools.items():
            module_info[module_path] = {
                "tool_count": len(tool_names),
                "tools": tool_names,
                "path": module_path,
                "watched": any(module_path.startswith(watched_dir) 
                              for watched_dir in self._watched_directories)
            }
        
        return module_info
    
    def get_status(self) -> Dict[str, Any]:
        """Get status of the dynamic registry."""
        return {
            "watching": self._observer is not None and self._observer.is_alive(),
            "watched_directories": list(self._watched_directories),
            "loaded_modules": len(self._module_to_tools),
            "dynamic_tools": len(self._tool_to_module),
            "total_tools": len(self.registry.list_tool_names()),
            "callbacks_registered": len(self._registration_callbacks)
        }
    
    def cleanup(self):
        """Clean up resources."""
        self.stop_watching()
        self._registration_callbacks.clear()
        self._module_to_tools.clear()
        self._tool_to_module.clear()


# Global dynamic registry instance
_global_dynamic_registry: Optional[DynamicToolRegistry] = None


def get_global_dynamic_registry() -> DynamicToolRegistry:
    """Get the global dynamic tool registry instance."""
    global _global_dynamic_registry
    if _global_dynamic_registry is None:
        _global_dynamic_registry = DynamicToolRegistry()
    return _global_dynamic_registry