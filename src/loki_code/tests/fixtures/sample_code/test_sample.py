"""
Sample Python code for testing Tree-sitter analysis and tool execution.
"""

def fibonacci(n):
    """Calculate fibonacci number recursively."""
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)


class Calculator:
    """Simple calculator with basic operations."""
    
    def __init__(self):
        self.history = []
    
    def add(self, a, b):
        """Add two numbers."""
        result = a + b
        self.history.append(f"{a} + {b} = {result}")
        return result
    
    def multiply(self, a, b):
        """Multiply two numbers."""
        result = a * b
        self.history.append(f"{a} * {b} = {result}")
        return result
    
    def get_history(self):
        """Get calculation history."""
        return self.history


def complex_function(data, options=None):
    """
    A more complex function for testing analysis.
    
    Args:
        data: Input data dictionary
        options: Optional configuration
        
    Returns:
        Processed result dictionary
    """
    if options is None:
        options = {"verbose": False, "timeout": 30}
    
    result = {
        "status": "processing",
        "input_size": len(data) if hasattr(data, '__len__') else 0,
        "timestamp": None
    }
    
    # Process data
    for key, value in data.items():
        if isinstance(value, str):
            result[key] = value.upper()
        elif isinstance(value, (int, float)):
            result[key] = value * 2
        else:
            result[key] = str(value)
    
    result["status"] = "completed"
    return result


# Test classes and functions for analysis
class DataProcessor:
    """Processes various types of data."""
    
    def __init__(self, config=None):
        self.config = config or {}
        self.processed_count = 0
    
    def process_item(self, item):
        """Process a single item."""
        self.processed_count += 1
        return {"id": self.processed_count, "data": item}
    
    @staticmethod
    def validate_data(data):
        """Validate input data."""
        if not isinstance(data, dict):
            raise ValueError("Data must be a dictionary")
        return True


def main():
    """Main function for testing."""
    calc = Calculator()
    print(f"5 + 3 = {calc.add(5, 3)}")
    print(f"4 * 7 = {calc.multiply(4, 7)}")
    
    processor = DataProcessor()
    result = processor.process_item({"test": "data"})
    print(f"Processed: {result}")


if __name__ == "__main__":
    main()