"""
Task-specific progress managers for Loki Code.

This module provides specialized progress managers for different types of operations
including LLM processing, tool execution, and project analysis.
"""

import time
from pathlib import Path
from typing import Optional, List, Dict, Any

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn,
        MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
    )
    from rich.text import Text
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from .indicators import ProgressContext, StreamingProgress
from ...utils.logging import get_logger


class LLMProgressManager:
    """Progress manager for LLM operations."""
    
    def __init__(self, console: Console):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required for progress displays")
            
        self.console = console
        self.logger = get_logger(__name__)
        
    def show_llm_thinking(self, prompt: str = "Processing request") -> ProgressContext:
        """Show LLM thinking progress with stages."""
        
        thinking_stages = [
            "🧠 Understanding request...",
            "📖 Analyzing context...", 
            "🎯 Planning response...",
            "✨ Generating answer..."
        ]
        
        progress = Progress(
            SpinnerColumn("dots12"),
            TextColumn("[cyan]{task.description}"),
            console=self.console,
            transient=True
        )
        
        task = progress.add_task("🤔 Thinking...", total=len(thinking_stages))
        
        return ProgressContext(
            progress=progress,
            task_id=task,
            stages=thinking_stages,
            manager=self
        )
    
    def show_streaming_response(self, estimated_tokens: int = None) -> StreamingProgress:
        """Show progress for streaming LLM response."""
        
        if estimated_tokens:
            progress = Progress(
                TextColumn("[cyan]🤖 Responding:"),
                BarColumn(bar_width=30),
                TextColumn("[dim]{task.completed}/{task.total} tokens"),
                console=self.console,
                transient=True
            )
            task = progress.add_task("Generating...", total=estimated_tokens)
        else:
            progress = Progress(
                SpinnerColumn("arrow3"),
                TextColumn("[cyan]🤖 Responding..."),
                TextColumn("[dim]{task.completed} tokens"),
                console=self.console,
                transient=True
            )
            task = progress.add_task("Generating...", total=None)
        
        progress.start()
        return StreamingProgress(progress, task)
    
    def show_model_loading(self, model_name: str) -> ProgressContext:
        """Show model loading progress."""
        
        loading_stages = [
            f"📥 Loading {model_name}...",
            "⚙️ Initializing model...",
            "🔧 Setting up context...",
            "✅ Model ready"
        ]
        
        progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("[yellow]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        )
        
        task = progress.add_task("Loading model...", total=len(loading_stages))
        
        return ProgressContext(
            progress=progress,
            task_id=task,
            stages=loading_stages,
            manager=self
        )


class ToolProgressManager:
    """Progress manager for tool execution."""
    
    def __init__(self, console: Console):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required for progress displays")
            
        self.console = console
        self.logger = get_logger(__name__)
        
        # Tool icons for better visual feedback
        self.tool_icons = {
            "file_reader": "📖",
            "file_writer": "✏️", 
            "directory_lister": "📁",
            "code_analyzer": "🔍",
            "file_searcher": "🔎",
            "debugger": "🐛",
            "refactoring_tool": "🔧",
            "code_generator": "⚡"
        }
        
    def show_tool_execution(self, tool_name: str, 
                                operation: str) -> ProgressContext:
        """Show tool execution progress."""
        
        icon = self.tool_icons.get(tool_name, "⚡")
        
        progress = Progress(
            SpinnerColumn("simpleDotsScrolling"),
            TextColumn(f"[yellow]{icon}"),
            TextColumn("[blue]{task.description}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        )
        
        task = progress.add_task(f"{operation}...")
        
        return ProgressContext(
            progress=progress,
            task_id=task,
            manager=self
        )
    
    def show_file_analysis(self, file_path: str, 
                               file_size: int = 0) -> ProgressContext:
        """Show file analysis progress with stages."""
        
        # Estimate steps based on file size
        if file_size > 0:
            estimated_steps = min(max(file_size // 1000, 3), 8)
        else:
            estimated_steps = 5
        
        analysis_stages = [
            "📖 Reading file...",
            "🔍 Parsing syntax...",
            "🏗️ Analyzing structure...",
            "⚡ Extracting functions...",
            "📊 Calculating metrics...",
            "✨ Generating insights...",
            "📝 Creating summary...",
            "✅ Analysis complete"
        ]
        
        # Use appropriate number of stages
        stages_to_use = analysis_stages[:estimated_steps]
        
        progress = Progress(
            TextColumn("[green]🔍 Analyzing:"),
            TextColumn(f"[blue]{Path(file_path).name}"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        )
        
        task = progress.add_task("Analyzing...", total=len(stages_to_use))
        
        return ProgressContext(
            progress=progress,
            task_id=task,
            stages=stages_to_use,
            manager=self
        )
    
    def show_file_writing(self, file_path: str, 
                              estimated_lines: int = 0) -> ProgressContext:
        """Show file writing progress."""
        
        writing_stages = [
            "📝 Preparing content...",
            "💾 Writing to file...",
            "🔧 Formatting code...",
            "✅ File saved"
        ]
        
        progress = Progress(
            TextColumn("[green]✏️ Writing:"),
            TextColumn(f"[blue]{Path(file_path).name}"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        )
        
        task = progress.add_task("Writing...", total=len(writing_stages))
        
        return ProgressContext(
            progress=progress,
            task_id=task,
            stages=writing_stages,
            manager=self
        )


class ProjectProgressManager:
    """Progress manager for project-wide operations."""
    
    def __init__(self, console: Console):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required for progress displays")
            
        self.console = console
        self.logger = get_logger(__name__)
        
    def show_project_analysis(self, project_path: str,
                                  file_count: int) -> ProgressContext:
        """Show project-wide analysis progress."""
        
        progress = Progress(
            TextColumn("[magenta]📊 Analyzing project:"),
            BarColumn(bar_width=40),
            TextColumn("[dim]{task.completed}/{task.total} files"),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False
        )
        
        task = progress.add_task(
            f"Scanning {Path(project_path).name}...", 
            total=file_count
        )
        
        return ProgressContext(
            progress=progress,
            task_id=task,
            manager=self
        )
    
    def show_project_scanning(self, project_path: str) -> ProgressContext:
        """Show project directory scanning."""
        
        scanning_stages = [
            "📁 Discovering files...",
            "🔍 Filtering code files...",
            "📋 Building file list...",
            "✅ Scan complete"
        ]
        
        progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("[magenta]📁 Scanning:"),
            TextColumn(f"[blue]{Path(project_path).name}"),
            TimeElapsedColumn(),
            console=self.console,
            transient=True
        )
        
        task = progress.add_task("Scanning...", total=len(scanning_stages))
        
        return ProgressContext(
            progress=progress,
            task_id=task,
            stages=scanning_stages,
            manager=self
        )
    
    def show_dependency_analysis(self, project_path: str) -> ProgressContext:
        """Show dependency analysis progress."""
        
        dependency_stages = [
            "📦 Finding package files...",
            "🔗 Analyzing dependencies...",
            "🌐 Checking versions...",
            "⚠️ Identifying issues...",
            "📊 Creating report..."
        ]
        
        progress = Progress(
            SpinnerColumn("dots12"),
            TextColumn("[cyan]📦 Dependencies:"),
            TextColumn(f"[blue]{Path(project_path).name}"),
            BarColumn(bar_width=20),
            MofNCompleteColumn(),
            console=self.console,
            transient=True
        )
        
        task = progress.add_task("Analyzing...", total=len(dependency_stages))
        
        return ProgressContext(
            progress=progress,
            task_id=task,
            stages=dependency_stages,
            manager=self
        )


class CodeProgressManager:
    """Progress manager for code-specific operations."""
    
    def __init__(self, console: Console):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required for progress displays")
            
        self.console = console
        self.logger = get_logger(__name__)
    
    def show_code_generation(self, task_description: str) -> ProgressContext:
        """Show code generation progress."""
        
        generation_stages = [
            "🧠 Understanding requirements...",
            "📋 Planning implementation...",
            "⚡ Generating code...",
            "🔧 Adding documentation...",
            "✅ Code ready"
        ]
        
        progress = Progress(
            SpinnerColumn("dots"),
            TextColumn("[green]⚡ Generating:"),
            TextColumn(f"[blue]{task_description}"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            console=self.console,
            transient=True
        )
        
        task = progress.add_task("Generating...", total=len(generation_stages))
        
        return ProgressContext(
            progress=progress,
            task_id=task,
            stages=generation_stages,
            manager=self
        )
    
    def show_code_refactoring(self, target: str) -> ProgressContext:
        """Show code refactoring progress."""
        
        refactoring_stages = [
            "🔍 Analyzing current code...",
            "🎯 Identifying improvements...",
            "🔧 Applying changes...",
            "🧪 Validating syntax...",
            "✅ Refactoring complete"
        ]
        
        progress = Progress(
            SpinnerColumn("simpleDotsScrolling"),
            TextColumn("[yellow]🔧 Refactoring:"),
            TextColumn(f"[blue]{target}"),
            BarColumn(bar_width=25),
            MofNCompleteColumn(),
            console=self.console,
            transient=True
        )
        
        task = progress.add_task("Refactoring...", total=len(refactoring_stages))
        
        return ProgressContext(
            progress=progress,
            task_id=task,
            stages=refactoring_stages,
            manager=self
        )