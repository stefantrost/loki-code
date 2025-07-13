"""
Integration of progress displays with agent and tool systems.

This module provides seamless integration of beautiful progress indicators
with Loki Code's agent processing and tool execution systems.
"""

import asyncio
import time
from typing import Any, Callable, AsyncIterator, Optional, Dict
from functools import wraps

try:
    from rich.console import Console
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .managers import (
    LLMProgressManager, ToolProgressManager, 
    ProjectProgressManager, CodeProgressManager
)
from .animations import ProgressEffects, AnimationSequence
from ...utils.logging import get_logger


class ProgressIntegration:
    """Main integration class for progress displays with Loki Code systems."""
    
    def __init__(self, console: Console = None):
        if not RICH_AVAILABLE:
            self.console = None
            self.enabled = False
            return
            
        # Use provided console or create a new one
        self.console = console or Console()
        self.enabled = True
        self.logger = get_logger(__name__)
        
        # Initialize progress managers
        self.llm_progress = LLMProgressManager(self.console)
        self.tool_progress = ToolProgressManager(self.console)
        self.project_progress = ProjectProgressManager(self.console)
        self.code_progress = CodeProgressManager(self.console)
        self.effects = ProgressEffects(self.console)
        self.animations = AnimationSequence(self.console)
        
    async def wrap_agent_call(self, agent_func: Callable, 
                            *args, **kwargs) -> Any:
        """Wrap agent call with beautiful progress indication."""
        
        if not self.enabled:
            return await agent_func(*args, **kwargs)
        
        # Extract request for better progress display
        request_text = "Processing request"
        if args and isinstance(args[0], str):
            request_text = args[0][:50] + ("..." if len(args[0]) > 50 else "")
        
        async with self.llm_progress.show_llm_thinking(request_text) as progress:
            start_time = time.time()
            
            try:
                # Show thinking stages
                progress.update_stage("🧠 Understanding request...")
                await asyncio.sleep(0.1)  # Brief pause for visual effect
                
                progress.update_stage("📖 Analyzing context...")
                await asyncio.sleep(0.1)
                
                progress.update_stage("🎯 Planning response...")
                await asyncio.sleep(0.1)
                
                progress.update_stage("✨ Generating answer...")
                
                # Execute the actual agent call
                result = await agent_func(*args, **kwargs)
                
                # Show completion
                duration = time.time() - start_time
                progress.complete(success=True)
                
                if duration > 2.0:  # Show celebration for longer operations
                    self.effects.show_completion_celebration("Agent processing", duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                progress.complete(success=False, message=str(e))
                self.effects.show_error_effect(f"Agent processing failed: {str(e)}")
                raise
    
    async def wrap_tool_call(self, tool_name: str, tool_func: Callable,
                           *args, **kwargs) -> Any:
        """Wrap tool call with progress indication."""
        
        if not self.enabled:
            return await tool_func(*args, **kwargs)
        
        # Determine operation type from tool name and args
        operation = self._determine_tool_operation(tool_name, args, kwargs)
        
        async with self.tool_progress.show_tool_execution(tool_name, operation) as progress:
            start_time = time.time()
            
            try:
                result = await tool_func(*args, **kwargs)
                
                duration = time.time() - start_time
                progress.complete(success=True)
                
                # Show success effect for longer operations
                if duration > 1.0:
                    self.effects.show_success_effect(f"{tool_name} completed", duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                progress.complete(success=False, message=str(e))
                self.effects.show_error_effect(f"{tool_name} failed: {str(e)}")
                raise
    
    async def wrap_file_analysis(self, file_path: str, analysis_func: Callable,
                               *args, **kwargs) -> Any:
        """Wrap file analysis with detailed progress."""
        
        if not self.enabled:
            return await analysis_func(*args, **kwargs)
        
        # Get file size for progress estimation
        try:
            from pathlib import Path
            file_size = Path(file_path).stat().st_size
        except:
            file_size = 0
        
        async with self.tool_progress.show_file_analysis(file_path, file_size) as progress:
            start_time = time.time()
            
            try:
                # Progress through analysis stages
                stages = [
                    "📖 Reading file...",
                    "🔍 Parsing syntax...", 
                    "🏗️ Analyzing structure...",
                    "⚡ Extracting functions...",
                    "📊 Calculating metrics..."
                ]
                
                for stage in stages:
                    progress.update_stage(stage)
                    await asyncio.sleep(0.1)  # Visual pause
                
                # Execute analysis
                result = await analysis_func(*args, **kwargs)
                
                duration = time.time() - start_time
                progress.complete(success=True)
                
                self.effects.show_success_effect(f"Analysis of {Path(file_path).name} complete", duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                progress.complete(success=False, message=str(e))
                self.effects.show_error_effect(f"Analysis failed: {str(e)}")
                raise
    
    async def show_streaming_response(self, response_stream: AsyncIterator[str],
                                    callback: Optional[Callable[[str], None]] = None,
                                    estimated_tokens: int = None) -> str:
        """Show streaming response with progress."""
        
        if not self.enabled:
            # Fallback without progress
            accumulated = ""
            async for chunk in response_stream:
                accumulated += chunk
                if callback:
                    callback(chunk)
            return accumulated
        
        streaming_progress = await self.llm_progress.show_streaming_response(estimated_tokens)
        accumulated = ""
        token_count = 0
        
        try:
            async for chunk in response_stream:
                accumulated += chunk
                token_count += len(chunk.split())
                
                # Update progress
                streaming_progress.update(token_count)
                
                # Call user callback
                if callback:
                    callback(chunk)
            
            # Complete streaming
            streaming_progress.complete()
            
            return accumulated
            
        except Exception as e:
            self.effects.show_error_effect(f"Streaming failed: {str(e)}")
            raise
        finally:
            if hasattr(streaming_progress, 'progress'):
                streaming_progress.progress.stop()
    
    async def show_project_analysis(self, project_path: str, 
                                  analysis_func: Callable,
                                  *args, **kwargs) -> Any:
        """Show project-wide analysis with progress."""
        
        if not self.enabled:
            return await analysis_func(*args, **kwargs)
        
        # Get file count for progress
        try:
            from pathlib import Path
            code_files = list(Path(project_path).rglob("*.py")) + \
                        list(Path(project_path).rglob("*.js")) + \
                        list(Path(project_path).rglob("*.ts"))
            file_count = len(code_files)
        except:
            file_count = 10  # Default estimate
        
        async with self.project_progress.show_project_analysis(project_path, file_count) as progress:
            start_time = time.time()
            
            try:
                # Execute analysis with progress callback
                def progress_callback(completed_files):
                    progress.update_progress(completed_files)
                
                # Add progress callback to kwargs if the function supports it
                if 'progress_callback' in analysis_func.__code__.co_varnames:
                    kwargs['progress_callback'] = progress_callback
                
                result = await analysis_func(*args, **kwargs)
                
                duration = time.time() - start_time
                progress.complete(success=True)
                
                self.effects.show_completion_celebration("Project analysis", duration)
                
                return result
                
            except Exception as e:
                duration = time.time() - start_time
                progress.complete(success=False, message=str(e))
                self.effects.show_error_effect(f"Project analysis failed: {str(e)}")
                raise
    
    def _determine_tool_operation(self, tool_name: str, args: tuple, kwargs: dict) -> str:
        """Determine the operation description for a tool call."""
        
        operation_map = {
            "file_reader": "Reading file",
            "file_writer": "Writing file",
            "directory_lister": "Listing directory",
            "code_analyzer": "Analyzing code",
            "file_searcher": "Searching files",
            "debugger": "Debugging code",
            "refactoring_tool": "Refactoring code",
            "code_generator": "Generating code"
        }
        
        base_operation = operation_map.get(tool_name, f"Executing {tool_name}")
        
        # Try to get more specific info from args
        if args:
            if isinstance(args[0], dict) and 'file_path' in args[0]:
                from pathlib import Path
                filename = Path(args[0]['file_path']).name
                return f"{base_operation}: {filename}"
            elif isinstance(args[0], str) and '.' in args[0]:
                return f"{base_operation}: {args[0]}"
        
        return base_operation
    
    def show_operation_summary(self, operations: list) -> None:
        """Show summary of completed operations."""
        
        if not self.enabled:
            print("\nOperation Summary:")
            for op in operations:
                status = "✅" if op.get('success', True) else "❌"
                print(f"  {status} {op.get('name', 'Operation')}")
            return
        
        self.effects.show_progress_summary(operations)


class CancellableProgress:
    """Progress indicator with cancellation support."""
    
    def __init__(self, console: Console = None):
        if not RICH_AVAILABLE:
            self.console = None
            self.enabled = False
            return
            
        self.console = console or Console()
        self.enabled = True
        self.is_cancelled = False
        self.cancel_event = asyncio.Event()
        self.logger = get_logger(__name__)
        
    async def run_cancellable_task(self, task_func: Callable, 
                                 task_name: str) -> Any:
        """Run task with cancellation support."""
        
        if not self.enabled:
            return await task_func()
        
        from rich.progress import Progress, SpinnerColumn, TextColumn
        from rich.text import Text
        
        # Show progress with cancel instruction
        progress_text = Text.assemble(
            (f"⚡ {task_name}", "blue"),
            ("\nPress ", "dim"),
            ("Ctrl+C", "yellow"),
            (" to cancel", "dim")
        )
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[blue]{task.description}"),
            console=self.console
        ) as progress:
            
            task_id = progress.add_task(task_name)
            
            try:
                # Create cancellable task
                task = asyncio.create_task(task_func())
                
                # Wait for completion or cancellation
                done, pending = await asyncio.wait(
                    [task, asyncio.create_task(self.cancel_event.wait())],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                if self.cancel_event.is_set():
                    # Cancelled
                    task.cancel()
                    progress.update(task_id, description=f"⚠️ {task_name} - Cancelled")
                    raise asyncio.CancelledError("Operation cancelled by user")
                else:
                    # Completed
                    result = await task
                    progress.update(task_id, description=f"✅ {task_name} - Complete")
                    return result
                    
            except asyncio.CancelledError:
                progress.update(task_id, description=f"⚠️ {task_name} - Cancelled")
                raise
            except Exception as e:
                progress.update(task_id, description=f"❌ {task_name} - Error")
                raise
    
    def cancel(self):
        """Cancel the current operation."""
        self.is_cancelled = True
        self.cancel_event.set()


def with_progress(console: Console = None):
    """Decorator to add progress indication to async functions."""
    
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            if not RICH_AVAILABLE:
                return await func(*args, **kwargs)
            
            integration = ProgressIntegration(console)
            
            # Determine progress type based on function name
            func_name = func.__name__
            
            if 'agent' in func_name.lower():
                return await integration.wrap_agent_call(func, *args, **kwargs)
            elif 'tool' in func_name.lower():
                tool_name = kwargs.get('tool_name', 'unknown_tool')
                return await integration.wrap_tool_call(tool_name, func, *args, **kwargs)
            else:
                # Generic progress
                async with integration.llm_progress.show_llm_thinking(f"Executing {func_name}") as progress:
                    try:
                        result = await func(*args, **kwargs)
                        progress.complete(success=True)
                        return result
                    except Exception as e:
                        progress.complete(success=False, message=str(e))
                        raise
        
        return wrapper
    return decorator