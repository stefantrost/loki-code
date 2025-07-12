"""
Main entry point for Loki Code CLI application.

This module provides the main entry point that can be called from the
installed console script or directly from the command line.
"""

import sys
from pathlib import Path

# Add the project root to Python path if running directly
if __name__ == "__main__":
    # This allows running the module directly during development
    project_root = Path(__file__).parent.parent.parent
    main_py = project_root / "main.py"
    
    if main_py.exists():
        # Import and run the main function from the root main.py
        sys.path.insert(0, str(project_root))
        from main import main as root_main
        sys.exit(root_main())
    else:
        print("Error: Could not find main.py in project root", file=sys.stderr)
        sys.exit(1)


def main():
    """
    Main entry point for the Loki Code application.
    
    This function serves as the entry point when the package is installed
    and called via the 'loki-code' console script.
    """
    try:
        # Import the main function from the root main.py
        # We do this import here to avoid circular imports
        project_root = Path(__file__).parent.parent.parent
        main_py = project_root / "main.py"
        
        if main_py.exists():
            sys.path.insert(0, str(project_root))
            from main import main as root_main
            return root_main()
        else:
            # If we can't find the root main.py, implement basic functionality here
            print("Error: Could not find main.py in project root", file=sys.stderr)
            print("This typically means the package was not installed correctly.", file=sys.stderr)
            return 1
            
    except Exception as e:
        print(f"Error starting Loki Code: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())