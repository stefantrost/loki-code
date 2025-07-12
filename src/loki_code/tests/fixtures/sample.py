"""
Sample Python file for testing code analysis functionality.

This file contains various Python constructs to test the completeness
of the Tree-sitter parsing and analysis capabilities.
"""

import os
import sys
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum


class Priority(Enum):
    """Priority levels for tasks."""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class Task:
    """Represents a task with priority and status."""
    id: int
    title: str
    description: Optional[str] = None
    priority: Priority = Priority.MEDIUM
    completed: bool = False
    
    def mark_complete(self) -> None:
        """Mark this task as completed."""
        self.completed = True
    
    def update_priority(self, new_priority: Priority) -> None:
        """Update the task priority."""
        self.priority = new_priority


def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number using recursion.
    
    Args:
        n: The position in the Fibonacci sequence
        
    Returns:
        The nth Fibonacci number
        
    Raises:
        ValueError: If n is negative
    """
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    elif n <= 1:
        return n
    else:
        return fibonacci(n - 1) + fibonacci(n - 2)


def fibonacci_iterative(n: int) -> int:
    """Calculate Fibonacci number iteratively for better performance."""
    if n < 0:
        raise ValueError("Fibonacci is not defined for negative numbers")
    elif n <= 1:
        return n
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b


class TaskManager:
    """Manages a collection of tasks with various operations."""
    
    def __init__(self):
        """Initialize an empty task manager."""
        self.tasks: Dict[int, Task] = {}
        self.next_id = 1
    
    def add_task(self, title: str, description: Optional[str] = None, 
                 priority: Priority = Priority.MEDIUM) -> int:
        """
        Add a new task to the manager.
        
        Args:
            title: Task title
            description: Optional task description
            priority: Task priority level
            
        Returns:
            The ID of the newly created task
        """
        task_id = self.next_id
        self.next_id += 1
        
        task = Task(
            id=task_id,
            title=title,
            description=description,
            priority=priority
        )
        
        self.tasks[task_id] = task
        return task_id
    
    def get_task(self, task_id: int) -> Optional[Task]:
        """Get a task by ID."""
        return self.tasks.get(task_id)
    
    def complete_task(self, task_id: int) -> bool:
        """Mark a task as completed."""
        task = self.get_task(task_id)
        if task:
            task.mark_complete()
            return True
        return False
    
    def get_tasks_by_priority(self, priority: Priority) -> List[Task]:
        """Get all tasks with the specified priority."""
        return [task for task in self.tasks.values() if task.priority == priority]
    
    def get_pending_tasks(self) -> List[Task]:
        """Get all tasks that are not completed."""
        return [task for task in self.tasks.values() if not task.completed]
    
    def get_completed_tasks(self) -> List[Task]:
        """Get all completed tasks."""
        return [task for task in self.tasks.values() if task.completed]
    
    def delete_task(self, task_id: int) -> bool:
        """Delete a task by ID."""
        if task_id in self.tasks:
            del self.tasks[task_id]
            return True
        return False
    
    def update_task_priority(self, task_id: int, priority: Priority) -> bool:
        """Update the priority of a task."""
        task = self.get_task(task_id)
        if task:
            task.update_priority(priority)
            return True
        return False
    
    def get_statistics(self) -> Dict[str, Union[int, float]]:
        """Get statistics about the tasks."""
        total_tasks = len(self.tasks)
        completed_tasks = len(self.get_completed_tasks())
        pending_tasks = len(self.get_pending_tasks())
        
        completion_rate = completed_tasks / total_tasks if total_tasks > 0 else 0.0
        
        priority_counts = {
            priority.name: len(self.get_tasks_by_priority(priority))
            for priority in Priority
        }
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "completion_rate": completion_rate,
            **priority_counts
        }


def process_data(data: List[Union[int, float]], 
                transform_func: Optional[callable] = None) -> List[Union[int, float]]:
    """
    Process a list of numeric data with optional transformation.
    
    Args:
        data: List of numbers to process
        transform_func: Optional function to transform each element
        
    Returns:
        Processed data list
    """
    if not data:
        return []
    
    processed = data.copy()
    
    if transform_func:
        processed = [transform_func(x) for x in processed]
    
    # Filter out any non-numeric results
    processed = [x for x in processed if isinstance(x, (int, float))]
    
    return processed


def analyze_numbers(numbers: List[Union[int, float]]) -> Dict[str, float]:
    """
    Analyze a list of numbers and return statistical information.
    
    Args:
        numbers: List of numbers to analyze
        
    Returns:
        Dictionary with statistical measures
    """
    if not numbers:
        return {"count": 0}
    
    count = len(numbers)
    total = sum(numbers)
    mean = total / count
    
    # Calculate variance
    variance = sum((x - mean) ** 2 for x in numbers) / count
    std_dev = variance ** 0.5
    
    sorted_numbers = sorted(numbers)
    
    # Calculate median
    mid = count // 2
    if count % 2 == 0:
        median = (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    else:
        median = sorted_numbers[mid]
    
    return {
        "count": count,
        "sum": total,
        "mean": mean,
        "median": median,
        "min": min(numbers),
        "max": max(numbers),
        "variance": variance,
        "std_dev": std_dev
    }


def complex_nested_function():
    """A function with complex nested structure for testing parsing."""
    
    def inner_function_1(x):
        """Inner function with closure."""
        def deeply_nested(y):
            """Deeply nested function."""
            if y > 0:
                return x + y
            else:
                return x - y
        
        return deeply_nested
    
    def inner_function_2():
        """Another inner function."""
        local_var = "local"
        
        for i in range(3):
            if i % 2 == 0:
                print(f"Even: {i}")
            else:
                print(f"Odd: {i}")
        
        return local_var
    
    # Complex control flow
    result = []
    
    try:
        func = inner_function_1(10)
        result.append(func(5))
        result.append(func(-3))
        
        inner_result = inner_function_2()
        result.append(len(inner_result))
        
    except Exception as e:
        print(f"Error in complex function: {e}")
        result = [0]
    
    finally:
        print("Complex function completed")
    
    return result


def main():
    """Main function demonstrating the task manager functionality."""
    print("Task Manager Demo")
    print("=" * 40)
    
    # Create task manager
    manager = TaskManager()
    
    # Add some tasks
    task1_id = manager.add_task("Learn Python", "Study Python programming", Priority.HIGH)
    task2_id = manager.add_task("Buy groceries", "Get milk, bread, and eggs", Priority.MEDIUM)
    task3_id = manager.add_task("Exercise", "Go for a 30-minute run", Priority.LOW)
    task4_id = manager.add_task("Fix bug", "Resolve critical production issue", Priority.CRITICAL)
    
    print(f"Added tasks with IDs: {task1_id}, {task2_id}, {task3_id}, {task4_id}")
    
    # Complete some tasks
    manager.complete_task(task2_id)
    manager.complete_task(task3_id)
    
    # Display statistics
    stats = manager.get_statistics()
    print(f"\nTask Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test fibonacci
    print(f"\nFibonacci(10) = {fibonacci(10)}")
    print(f"Fibonacci iterative(10) = {fibonacci_iterative(10)}")
    
    # Test data analysis
    test_numbers = [1, 5, 3, 9, 2, 7, 4, 8, 6]
    analysis = analyze_numbers(test_numbers)
    print(f"\nNumber analysis for {test_numbers}:")
    for key, value in analysis.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    # Test complex function
    complex_result = complex_nested_function()
    print(f"\nComplex function result: {complex_result}")


if __name__ == "__main__":
    main()