"""
Core progress indicator components for Loki Code.

This module provides the fundamental progress display components including
dynamic indicators, streaming progress, and completion status displays.
"""

import time
import asyncio
from typing import Optional, Union, Dict, Any
from dataclasses import dataclass, field
from contextlib import asynccontextmanager

try:
    from rich.console import Console
    from rich.progress import (
        Progress, SpinnerColumn, TextColumn, BarColumn, 
        MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
    )
    from rich.text import Text
    from rich.panel import Panel
    from rich.align import Align
    from rich.spinner import Spinner
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ...utils.logging import get_logger


@dataclass
class ProgressState:
    """Current state of a progress operation."""
    current_step: int = 0
    total_steps: Optional[int] = None
    start_time: float = field(default_factory=time.time)
    is_cancelled: bool = False
    description: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProgressIndicator:
    """Core progress indicator for various operations."""
    
    def __init__(self, console: Console, task_name: str, 
                 total_steps: Optional[int] = None):
        if not RICH_AVAILABLE:
            raise ImportError("Rich library required for progress indicators")
            
        self.console = console
        self.task_name = task_name
        self.total_steps = total_steps
        self.state = ProgressState(total_steps=total_steps)
        self.logger = get_logger(__name__)
        
    def create_progress_display(self) -> Union[Progress, Spinner]:
        """Create appropriate progress display based on type."""
        
        if self.total_steps and self.total_steps > 0:
            return self._create_determinate_progress()
        else:
            return self._create_indeterminate_progress()
    
    def _create_determinate_progress(self) -> Progress:
        """Progress bar with known total steps."""
        
        return Progress(
            SpinnerColumn("dots"),
            TextColumn("[blue]{task.description}"),
            BarColumn(bar_width=40),
            MofNCompleteColumn(),
            TextColumn("•"),
            TimeElapsedColumn(),
            TextColumn("•"),
            TimeRemainingColumn(),
            console=self.console,
            transient=False
        )
    
    def _create_indeterminate_progress(self) -> Spinner:
        """Spinner for unknown duration tasks."""
        
        return Spinner(
            "dots", 
            text=Text(self.task_name, style="blue"),
            style="cyan"
        )
    
    def update_progress(self, step: int, description: str = None) -> None:
        """Update progress indicator state."""
        
        self.state.current_step = step
        
        if description:
            self.state.description = description
            self.task_name = description
    
    def complete(self, success: bool = True, message: str = None) -> Panel:
        """Show completion status with timing."""
        
        elapsed = time.time() - self.state.start_time
        
        if success:
            icon = "✅"
            style = "green"
            status = "Complete"
        else:
            icon = "❌" 
            style = "red"
            status = "Failed"
        
        content = Text.assemble(
            (f"{icon} ", style),
            (f"{status}: ", style),
            (self.task_name, "white"),
            ("\n", ""),
            (f"Time: {elapsed:.1f}s", "dim")
        )
        
        if message:
            content.append("\n")
            content.append(message, "white")
        
        return Panel(
            content,
            border_style=style,
            padding=(0, 1)
        )


class ProgressContext:
    """Context manager for progress operations."""
    
    def __init__(self, progress: Progress, task_id: Any, 
                 stages: Optional[list] = None, manager=None):
        self.progress = progress
        self.task_id = task_id
        self.stages = stages or []
        self.manager = manager
        self.current_stage = 0
        self.start_time = time.time()
        
    async def __aenter__(self):
        """Enter progress context."""
        self.progress.start()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit progress context."""
        self.progress.stop()
        
        # Show completion
        if exc_type is None:
            self._show_success()
        else:
            self._show_error(str(exc_val))
    
    def update_stage(self, stage_name: str = None):
        """Update to next stage."""
        if self.stages and self.current_stage < len(self.stages):
            if stage_name is None:
                stage_name = self.stages[self.current_stage]
            
            self.progress.update(self.task_id, 
                               completed=self.current_stage + 1,
                               description=stage_name)
            self.current_stage += 1
    
    def update_progress(self, completed: int, description: str = None):
        """Update progress manually."""
        if description:
            self.progress.update(self.task_id, 
                               completed=completed,
                               description=description)
        else:
            self.progress.update(self.task_id, completed=completed)
    
    def complete(self, success: bool = True, message: str = None):
        """Mark operation as complete."""
        elapsed = time.time() - self.start_time
        
        if success:
            status = f"✅ Complete ({elapsed:.1f}s)"
            style = "green"
        else:
            status = f"❌ Failed ({elapsed:.1f}s)"
            style = "red"
        
        self.progress.update(self.task_id, description=status)
    
    def _show_success(self):
        """Show success completion."""
        elapsed = time.time() - self.start_time
        self.progress.update(self.task_id, 
                           description=f"✅ Complete ({elapsed:.1f}s)")
    
    def _show_error(self, error_msg: str):
        """Show error completion."""
        elapsed = time.time() - self.start_time
        self.progress.update(self.task_id, 
                           description=f"❌ Failed ({elapsed:.1f}s)")


class StreamingProgress:
    """Progress indicator for streaming operations."""
    
    def __init__(self, progress: Progress, task_id: Any):
        self.progress = progress
        self.task_id = task_id
        self.token_count = 0
        self.start_time = time.time()
        
    def update(self, tokens_received: int):
        """Update streaming progress."""
        self.token_count = tokens_received
        
        # Update progress display
        self.progress.update(self.task_id, completed=tokens_received)
    
    def add_tokens(self, token_chunk: str):
        """Add tokens from a chunk."""
        # Simple token counting (could be improved)
        new_tokens = len(token_chunk.split())
        self.token_count += new_tokens
        self.update(self.token_count)
    
    def complete(self):
        """Mark streaming as complete."""
        elapsed = time.time() - self.start_time
        self.progress.update(self.task_id, 
                           description=f"✅ Response complete ({self.token_count} tokens, {elapsed:.1f}s)")


@asynccontextmanager
async def progress_indicator(console: Console, task_name: str, 
                           total_steps: Optional[int] = None):
    """Async context manager for simple progress indication."""
    
    if not RICH_AVAILABLE:
        # Fallback for no Rich
        print(f"Starting: {task_name}")
        yield None
        print(f"Completed: {task_name}")
        return
    
    indicator = ProgressIndicator(console, task_name, total_steps)
    display = indicator.create_progress_display()
    
    if isinstance(display, Progress):
        task_id = display.add_task(task_name, total=total_steps)
        
        with display:
            try:
                yield ProgressContext(display, task_id)
            except Exception as e:
                display.update(task_id, description=f"❌ Failed: {str(e)}")
                raise
            else:
                display.update(task_id, description="✅ Complete")
    else:
        # Spinner case
        with console.status(display):
            yield indicator