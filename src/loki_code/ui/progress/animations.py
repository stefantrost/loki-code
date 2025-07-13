"""
Progress animations and visual effects for Loki Code.

This module provides beautiful animations, completion effects, and visual feedback
for various operations to enhance the user experience.
"""

import time
import asyncio
from typing import Optional, List

try:
    from rich.console import Console
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.text import Text
    from rich.panel import Panel
    from rich.align import Align
    from rich.spinner import Spinner
    from rich import box
    from rich.table import Table
    from rich.columns import Columns
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

from ...utils.logging import get_logger


class ProgressAnimations:
    """Collection of progress animations for different operations."""
    
    @staticmethod
    def create_thinking_animation() -> Spinner:
        """Create thinking animation for LLM processing."""
        
        if not RICH_AVAILABLE:
            return None
            
        return Spinner(
            "dots12",
            text=Text("🤔 Thinking...", style="cyan"),
            style="blue"
        )
    
    @staticmethod
    def create_working_animation(task: str) -> Spinner:
        """Create working animation for general tasks."""
        
        if not RICH_AVAILABLE:
            return None
            
        return Spinner(
            "simpleDotsScrolling", 
            text=Text(f"⚡ {task}", style="yellow"),
            style="blue"
        )
    
    @staticmethod
    def create_analyzing_animation(target: str) -> Spinner:
        """Create analyzing animation for code analysis."""
        
        if not RICH_AVAILABLE:
            return None
            
        return Spinner(
            "dots",
            text=Text(f"🔍 Analyzing {target}...", style="green"),
            style="cyan"
        )
    
    @staticmethod
    def create_generating_animation(item: str) -> Spinner:
        """Create generating animation for code generation."""
        
        if not RICH_AVAILABLE:
            return None
            
        return Spinner(
            "arrow3",
            text=Text(f"⚡ Generating {item}...", style="yellow"),
            style="green"
        )
    
    @staticmethod
    def create_streaming_animation() -> Spinner:
        """Create streaming animation for real-time responses."""
        
        if not RICH_AVAILABLE:
            return None
            
        return Spinner(
            "arrow3",
            text=Text("🤖 Streaming response...", style="cyan"),
            style="blue"
        )
    
    @staticmethod
    def create_loading_animation(item: str) -> Spinner:
        """Create loading animation for initialization."""
        
        if not RICH_AVAILABLE:
            return None
            
        return Spinner(
            "dots2",
            text=Text(f"📥 Loading {item}...", style="yellow"),
            style="magenta"
        )


class ProgressEffects:
    """Visual effects for progress completion and status."""
    
    def __init__(self, console: Console):
        if not RICH_AVAILABLE:
            self.console = None
            return
            
        self.console = console
        self.logger = get_logger(__name__)
        
    def show_success_effect(self, message: str, duration: float = 0) -> None:
        """Show success completion effect."""
        
        if not self.console:
            print(f"✅ Success: {message}")
            return
            
        if duration > 0:
            timing_text = f" in {duration:.1f}s"
        else:
            timing_text = ""
            
        celebration = Text.assemble(
            ("🎉 ", "bold green"),
            ("Success! ", "bold green"),
            (message, "green"),
            (timing_text, "dim")
        )
        
        panel = Panel(
            Align.center(celebration),
            style="bold green",
            box=box.ROUNDED,
            padding=(0, 2)
        )
        
        self.console.print(panel)
        
    def show_completion_celebration(self, task_name: str, 
                                  duration: float) -> None:
        """Show celebration effect for completed tasks."""
        
        if not self.console:
            print(f"🎊 Completed: {task_name} in {duration:.1f}s")
            return
            
        celebration = Text.assemble(
            ("🎊 ", "bold yellow"),
            ("Completed: ", "bold green"),
            (task_name, "blue"),
            (f" in {duration:.1f}s", "dim")
        )
        
        panel = Panel(
            Align.center(celebration),
            style="bold green",
            box=box.DOUBLE,
            padding=(0, 2)
        )
        
        self.console.print(panel)
        time.sleep(0.5)  # Brief pause for effect
    
    def show_error_effect(self, error_message: str) -> None:
        """Show error effect with helpful styling."""
        
        if not self.console:
            print(f"❌ Error: {error_message}")
            return
            
        error_panel = Panel(
            Text.assemble(
                ("⚠️ ", "bold red"),
                ("Error: ", "bold red"),
                (error_message, "red")
            ),
            style="bold red",
            box=box.ROUNDED,
            padding=(0, 1)
        )
        
        self.console.print(error_panel)
    
    def show_warning_effect(self, warning_message: str) -> None:
        """Show warning effect."""
        
        if not self.console:
            print(f"⚠️ Warning: {warning_message}")
            return
            
        warning_panel = Panel(
            Text.assemble(
                ("⚠️ ", "bold yellow"),
                ("Warning: ", "bold yellow"),
                (warning_message, "yellow")
            ),
            style="bold yellow",
            box=box.ROUNDED,
            padding=(0, 1)
        )
        
        self.console.print(warning_panel)
    
    def show_info_effect(self, info_message: str) -> None:
        """Show informational effect."""
        
        if not self.console:
            print(f"ℹ️ Info: {info_message}")
            return
            
        info_panel = Panel(
            Text.assemble(
                ("ℹ️ ", "bold blue"),
                ("Info: ", "bold blue"),
                (info_message, "blue")
            ),
            style="bold blue",
            box=box.ROUNDED,
            padding=(0, 1)
        )
        
        self.console.print(info_panel)
    
    def create_status_table(self, title: str, items: List[tuple]) -> Table:
        """Create a beautiful status table."""
        
        if not self.console:
            # Fallback for no Rich
            print(f"\n{title}:")
            for name, status in items:
                print(f"  {name}: {status}")
            return None
            
        table = Table(title=title, box=box.ROUNDED, show_header=True)
        table.add_column("Component", style="cyan", no_wrap=True)
        table.add_column("Status", style="magenta")
        
        for name, status in items:
            # Color-code status
            if "✅" in status or "ready" in status.lower():
                status_style = "green"
            elif "❌" in status or "error" in status.lower():
                status_style = "red"
            elif "⚠️" in status or "warning" in status.lower():
                status_style = "yellow"
            else:
                status_style = "blue"
                
            table.add_row(name, Text(status, style=status_style))
        
        return table
    
    def show_progress_summary(self, operations: List[dict]) -> None:
        """Show summary of multiple operations."""
        
        if not self.console:
            print("\nProgress Summary:")
            for op in operations:
                status = "✅" if op.get('success', True) else "❌"
                print(f"  {status} {op.get('name', 'Operation')}: {op.get('duration', 0):.1f}s")
            return
            
        # Create summary table
        table = Table(
            title="🚀 Operation Summary",
            box=box.ROUNDED,
            show_header=True
        )
        table.add_column("Operation", style="cyan")
        table.add_column("Status", justify="center")
        table.add_column("Duration", justify="right", style="dim")
        
        total_time = 0
        success_count = 0
        
        for op in operations:
            name = op.get('name', 'Unknown Operation')
            success = op.get('success', True)
            duration = op.get('duration', 0)
            
            if success:
                status = Text("✅ Success", style="green")
                success_count += 1
            else:
                status = Text("❌ Failed", style="red")
            
            total_time += duration
            table.add_row(name, status, f"{duration:.1f}s")
        
        # Add summary row
        table.add_section()
        success_rate = (success_count / len(operations)) * 100 if operations else 0
        table.add_row(
            Text("Total", style="bold"),
            Text(f"{success_count}/{len(operations)} ({success_rate:.0f}%)", 
                 style="bold green" if success_rate == 100 else "bold yellow"),
            Text(f"{total_time:.1f}s", style="bold")
        )
        
        self.console.print(table)


class AnimationSequence:
    """Sequence of animations for complex operations."""
    
    def __init__(self, console: Console):
        if not RICH_AVAILABLE:
            self.console = None
            return
            
        self.console = console
        self.effects = ProgressEffects(console)
        
    async def run_startup_sequence(self) -> None:
        """Run beautiful startup animation sequence."""
        
        if not self.console:
            print("🚀 Starting Loki Code...")
            return
            
        startup_steps = [
            ("🧠", "Initializing AI engine...", 0.5),
            ("🔧", "Loading tools...", 0.3),
            ("📚", "Preparing knowledge base...", 0.4),
            ("✨", "Ready to code!", 0.2)
        ]
        
        for icon, message, delay in startup_steps:
            with self.console.status(
                Spinner("dots", text=Text(f"{icon} {message}", style="cyan"))
            ):
                await asyncio.sleep(delay)
        
        # Final success message
        self.effects.show_success_effect("Loki Code is ready! 🚀")
    
    async def run_shutdown_sequence(self) -> None:
        """Run beautiful shutdown animation sequence."""
        
        if not self.console:
            print("👋 Shutting down Loki Code...")
            return
            
        shutdown_steps = [
            ("💾", "Saving session...", 0.2),
            ("🧹", "Cleaning up...", 0.3),
            ("👋", "Goodbye!", 0.2)
        ]
        
        for icon, message, delay in shutdown_steps:
            with self.console.status(
                Spinner("dots", text=Text(f"{icon} {message}", style="yellow"))
            ):
                await asyncio.sleep(delay)
        
        # Final message
        self.console.print(
            Panel(
                Align.center(Text("Thank you for using Loki Code! 👋", style="bold blue")),
                style="blue",
                box=box.ROUNDED
            )
        )