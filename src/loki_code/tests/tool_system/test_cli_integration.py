"""
CLI integration tests.

Tests the integration between tools and CLI commands:
- CLI command execution
- Tool discovery and listing
- Command-line tool execution
- Configuration and setup
"""

import pytest
import subprocess
import sys
import json
from pathlib import Path
from unittest.mock import Mock, patch

from ...cli.commands import create_parser, parse_args
from ...cli.handlers import handle_cli_command
from ...config import load_config
from ..fixtures.tool_test_fixtures import (
    test_config, temp_workspace, ToolTestHelpers,
    get_sample_code_path
)


class TestCLIIntegration:
    """Test CLI integration with tool system."""
    
    def test_cli_parser_creation(self):
        """Test CLI argument parser creation."""
        parser = create_parser()
        assert parser is not None
        
        # Test basic arguments
        args = parser.parse_args(["--version"])
        assert hasattr(args, 'version')
        
        # Test tool-related arguments
        args = parser.parse_args(["--list-tools"])
        assert args.list_tools is True
        
        args = parser.parse_args(["--tool-info", "file_reader"])
        assert args.tool_info == "file_reader"
    
    def test_list_tools_command(self, test_config):
        """Test --list-tools CLI command."""
        args = parse_args(["--list-tools"])
        
        with patch('loki_code.cli.handlers.load_config', return_value=test_config):
            result = handle_cli_command(args)
        
        # Should complete successfully
        assert result == 0
    
    def test_tool_info_command(self, test_config):
        """Test --tool-info CLI command."""
        args = parse_args(["--tool-info", "file_reader"])
        
        with patch('loki_code.cli.handlers.load_config', return_value=test_config):
            result = handle_cli_command(args)
        
        # Should complete successfully
        assert result == 0
    
    def test_read_file_command(self, test_config, temp_workspace):
        """Test --read-file CLI command."""
        # Create test file
        test_file = temp_workspace / "cli_test.py"
        test_content = "print('CLI test file')\n"
        ToolTestHelpers.create_test_file(test_file, test_content)
        
        args = parse_args(["--read-file", str(test_file)])
        
        with patch('loki_code.cli.handlers.load_config', return_value=test_config):
            result = handle_cli_command(args)
        
        # Should complete successfully
        assert result == 0
    
    def test_analyze_file_command(self, test_config):
        """Test --analyze-file CLI command."""
        sample_file = get_sample_code_path("python")
        args = parse_args(["--analyze-file", str(sample_file)])
        
        with patch('loki_code.cli.handlers.load_config', return_value=test_config):
            result = handle_cli_command(args)
        
        # Should complete successfully
        assert result == 0
    
    def test_test_llm_command(self, test_config):
        """Test --test-llm CLI command."""
        args = parse_args(["--test-llm"])
        
        # Mock the provider creation to avoid actual LLM calls
        with patch('loki_code.cli.handlers.create_llm_provider') as mock_provider:
            mock_provider.return_value = Mock()
            
            with patch('loki_code.cli.handlers.test_ollama_connection') as mock_test:
                mock_test.return_value = Mock(success=True)
                
                with patch('loki_code.cli.handlers.load_config', return_value=test_config):
                    result = handle_cli_command(args)
        
        # Should complete successfully
        assert result == 0
    
    def test_list_providers_command(self, test_config):
        """Test --list-providers CLI command."""
        args = parse_args(["--list-providers"])
        
        # Mock provider listing
        with patch('loki_code.cli.handlers.list_available_providers') as mock_list:
            mock_list.return_value = {
                "ollama": {"available": True, "models": ["llama2"]},
                "openai": {"available": False, "models": []}
            }
            
            with patch('loki_code.cli.handlers.load_config', return_value=test_config):
                result = handle_cli_command(args)
        
        # Should complete successfully
        assert result == 0
    
    def test_chat_mode_single_prompt(self, test_config):
        """Test chat mode with single prompt."""
        args = parse_args(["--chat", "--prompt", "Hello, world!"])
        
        # Mock provider and response
        mock_response = Mock()
        mock_response.content = "Hello! How can I help you?"
        
        mock_provider = Mock()
        mock_provider.generate_sync.return_value = mock_response
        
        with patch('loki_code.cli.handlers.create_llm_provider', return_value=mock_provider):
            with patch('loki_code.cli.handlers.load_config', return_value=test_config):
                result = handle_cli_command(args)
        
        # Should complete successfully
        assert result == 0
        mock_provider.generate_sync.assert_called_once()
    
    def test_invalid_file_handling(self, test_config):
        """Test CLI handling of invalid file paths."""
        args = parse_args(["--read-file", "/nonexistent/file.py"])
        
        with patch('loki_code.cli.handlers.load_config', return_value=test_config):
            result = handle_cli_command(args)
        
        # Should fail gracefully
        assert result == 1
    
    def test_configuration_error_handling(self):
        """Test CLI handling of configuration errors."""
        args = parse_args(["--list-tools"])
        
        # Mock configuration error
        with patch('loki_code.cli.handlers.load_config') as mock_config:
            mock_config.side_effect = Exception("Configuration error")
            result = handle_cli_command(args)
        
        # Should handle error gracefully
        assert result == 1
    
    def test_cli_with_verbose_flag(self, test_config):
        """Test CLI commands with verbose flag."""
        args = parse_args(["--verbose", "--list-tools"])
        assert args.verbose is True
        
        with patch('loki_code.cli.handlers.load_config', return_value=test_config):
            result = handle_cli_command(args)
        
        # Should complete successfully with verbose output
        assert result == 0
    
    def test_mutually_exclusive_modes(self):
        """Test mutually exclusive CLI modes."""
        parser = create_parser()
        
        # These should not conflict (chat is in mode group)
        args1 = parser.parse_args(["--chat"])
        assert args1.chat is True
        
        args2 = parser.parse_args(["--tui"])
        assert args2.tui is True
        
        # Test that we can specify non-conflicting options
        args3 = parser.parse_args(["--list-tools", "--verbose"])
        assert args3.list_tools is True
        assert args3.verbose is True
    
    @pytest.mark.slow
    def test_full_cli_integration(self, temp_workspace):
        """Test full CLI integration by running actual commands."""
        # This test runs the actual CLI as a subprocess
        # Skip if we can't find the main script
        main_script = Path(__file__).parent.parent.parent.parent.parent / "main.py"
        if not main_script.exists():
            pytest.skip("Main script not found for integration test")
        
        # Create test file
        test_file = temp_workspace / "integration_test.py"
        ToolTestHelpers.create_test_file(test_file, "def test(): return 42")
        
        # Test help command
        result = subprocess.run([
            sys.executable, str(main_script), "--help"
        ], capture_output=True, text=True, timeout=10)
        
        assert result.returncode == 0
        assert "Loki Code" in result.stdout or "usage:" in result.stdout
    
    def test_default_tui_mode(self, test_config):
        """Test default TUI mode activation."""
        # When no specific mode is specified, should default to TUI
        args = parse_args([])
        
        # Mock the TUI creation to avoid actual UI
        with patch('loki_code.cli.handlers.create_loki_app') as mock_tui:
            mock_app = Mock()
            mock_tui.return_value = mock_app
            
            with patch('loki_code.cli.handlers.load_config', return_value=test_config):
                result = handle_cli_command(args)
        
        # Should attempt to create and run TUI
        mock_tui.assert_called_once()
        mock_app.run.assert_called_once()
    
    def test_tui_unavailable_fallback(self, test_config):
        """Test fallback when TUI is unavailable."""
        args = parse_args([])  # Default to TUI mode
        
        # Mock TUI unavailable
        with patch('loki_code.cli.handlers.create_loki_app', return_value=None):
            with patch('loki_code.cli.handlers.load_config', return_value=test_config):
                result = handle_cli_command(args)
        
        # Should fail gracefully
        assert result == 1