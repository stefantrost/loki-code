#!/bin/bash

# Script to run Loki Code TUI with logging in a separate terminal

echo "🚀 Starting Loki Code TUI with Developer Console"
echo ""
echo "This will open two windows:"
echo "1. Main TUI application"
echo "2. Developer console for logs and debugging"
echo ""
echo "First, let's start the developer console..."

# Check if we're on macOS to use the right terminal command
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS - open developer console in new terminal window
    osascript -e 'tell application "Terminal" to do script "textual console"'
    echo "✅ Developer console opened in new Terminal window"
else
    # Linux/other - show instructions
    echo "Please run 'textual console' in a separate terminal window"
    echo "Press Enter when ready..."
    read
fi

echo ""
echo "Now starting the TUI application with --dev flag..."
echo "Press Ctrl+C to quit"
echo ""

# Run the main app with textual run --dev to connect to the console
textual run --dev main.py --tui