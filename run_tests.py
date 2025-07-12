#!/usr/bin/env python3
"""
Test runner script for Loki Code integration tests.

This script provides various ways to run the integration test suite
with different configurations and reporting options.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_tests(args):
    """Run the test suite with specified options."""
    
    # Base pytest command
    cmd = ["python", "-m", "pytest"]
    
    # Add test path
    test_path = Path("src/loki_code/tests/test_tool_integration.py")
    cmd.append(str(test_path))
    
    # Add verbosity
    if args.verbose:
        cmd.extend(["-v", "-s", "--capture=no"])
    else:
        cmd.append("-v")
    
    # Add specific test categories
    if args.unit:
        cmd.extend(["-m", "unit"])
    elif args.integration:
        cmd.extend(["-m", "integration"])
    elif args.performance:
        cmd.extend(["-m", "performance"])
    elif args.security:
        cmd.extend(["-m", "security"])
    elif args.cli:
        cmd.extend(["-m", "cli"])
    elif args.multi_language:
        cmd.extend(["-m", "multi_language"])
    
    # Add coverage reporting
    if args.coverage:
        cmd.extend([
            "--cov=src/loki_code",
            "--cov-report=html",
            "--cov-report=term-missing"
        ])
    
    # Add parallel execution
    if args.parallel:
        cmd.extend(["-n", "auto"])
    
    # Add HTML reporting
    if args.html_report:
        cmd.extend(["--html=test_report.html", "--self-contained-html"])
    
    # Add benchmark mode
    if args.benchmark:
        cmd.append("--benchmark-only")
    
    # Exclude slow tests unless specifically requested
    if not args.include_slow:
        cmd.extend(["-m", "not slow"])
    
    # Add any additional pytest args
    if args.pytest_args:
        cmd.extend(args.pytest_args.split())
    
    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)
    
    # Run the tests
    try:
        result = subprocess.run(cmd, cwd=Path(__file__).parent)
        return result.returncode
    except KeyboardInterrupt:
        print("\nTest run interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running tests: {e}")
        return 1


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run Loki Code integration tests",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_tests.py                    # Run all tests
  python run_tests.py --verbose          # Run with verbose output
  python run_tests.py --integration      # Run only integration tests
  python run_tests.py --performance      # Run only performance tests
  python run_tests.py --coverage         # Run with coverage reporting
  python run_tests.py --html-report      # Generate HTML test report
  python run_tests.py --benchmark        # Run only benchmark tests
  python run_tests.py --parallel         # Run tests in parallel
        """
    )
    
    # Test categories
    parser.add_argument(
        "--unit", 
        action="store_true",
        help="Run only unit tests"
    )
    parser.add_argument(
        "--integration", 
        action="store_true",
        help="Run only integration tests"
    )
    parser.add_argument(
        "--performance", 
        action="store_true",
        help="Run only performance tests"
    )
    parser.add_argument(
        "--security", 
        action="store_true",
        help="Run only security tests"
    )
    parser.add_argument(
        "--cli", 
        action="store_true",
        help="Run only CLI integration tests"
    )
    parser.add_argument(
        "--multi-language", 
        action="store_true",
        dest="multi_language",
        help="Run only multi-language tests"
    )
    
    # Test options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Run tests with verbose output"
    )
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report"
    )
    parser.add_argument(
        "--html-report",
        action="store_true",
        help="Generate HTML test report"
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Run tests in parallel"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run benchmark tests only"
    )
    parser.add_argument(
        "--include-slow",
        action="store_true",
        help="Include slow tests in the run"
    )
    parser.add_argument(
        "--pytest-args",
        type=str,
        help="Additional arguments to pass to pytest"
    )
    
    args = parser.parse_args()
    
    # Check if pytest is available
    try:
        subprocess.run(
            ["python", "-m", "pytest", "--version"], 
            capture_output=True, 
            check=True
        )
    except subprocess.CalledProcessError:
        print("❌ pytest is not installed or not available")
        print("💡 Install with: pip install -r requirements-test.txt")
        return 1
    
    # Run the tests
    return run_tests(args)


if __name__ == "__main__":
    sys.exit(main())