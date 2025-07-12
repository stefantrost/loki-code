"""
Simple benchmarks that don't require pytest - for basic testing.
"""

import tempfile
import time
import os
from pathlib import Path


def simple_benchmark_file_analysis():
    """Simple benchmark without pytest dependency."""
    
    # Create temporary files of different sizes
    temp_dir = tempfile.mkdtemp(prefix="loki_benchmark_")
    results = {}
    
    try:
        test_cases = [
            ("small", 50),
            ("medium", 200), 
            ("large", 500)
        ]
        
        for size_name, line_count in test_cases:
            # Create test file
            content_lines = []
            for i in range(line_count):
                content_lines.append(f"def function_{i}(param):")
                content_lines.append(f"    '''Function {i} for testing.'''")
                content_lines.append(f"    return param * {i}")
                content_lines.append("")
            
            content = "\n".join(content_lines)
            
            # Write to file
            test_file = os.path.join(temp_dir, f"test_{size_name}.py")
            with open(test_file, 'w') as f:
                f.write(content)
            
            # Measure file reading performance
            start_time = time.time()
            
            # Simulate file analysis (basic reading)
            with open(test_file, 'r') as f:
                file_content = f.read()
                lines = len(file_content.split('\n'))
                chars = len(file_content)
            
            end_time = time.time()
            duration_ms = (end_time - start_time) * 1000
            
            chars_per_second = chars / (duration_ms / 1000) if duration_ms > 0 else 0
            
            results[size_name] = {
                "duration_ms": duration_ms,
                "lines": lines,
                "chars": chars,
                "chars_per_second": chars_per_second
            }
        
        return results
        
    finally:
        # Clean up temp files
        import shutil
        try:
            shutil.rmtree(temp_dir)
        except:
            pass


if __name__ == "__main__":
    print("Running simple file analysis benchmark...")
    results = simple_benchmark_file_analysis()
    
    print("\nBenchmark Results:")
    print("-" * 30)
    
    for size, metrics in results.items():
        print(f"{size.capitalize()} file:")
        print(f"  Duration: {metrics['duration_ms']:.2f}ms")
        print(f"  Lines: {metrics['lines']}")
        print(f"  Chars/sec: {metrics['chars_per_second']:,.0f}")
        print()