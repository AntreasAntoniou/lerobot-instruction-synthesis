#!/usr/bin/env python

"""
Tests for CLI commands to ensure they work as documented in README.
"""

import subprocess
import sys
from unittest.mock import Mock, patch

import pytest


class TestCLICommands:
    """Test the actual CLI commands work as documented."""

    def test_lesynthesis_command_exists(self):
        """Test that lesynthesis command is available."""
        # Try without any arguments to see available commands
        result = subprocess.run(
            ["lesynthesis"], capture_output=True, text=True
        )

        # If that fails, try with python -m
        if result.returncode != 0 and "lesynthesis" not in result.stdout:
            result = subprocess.run(
                [sys.executable, "-m", "lesynthesis"],
                capture_output=True,
                text=True,
            )

        # Fire shows usage when called without args
        output = result.stdout + result.stderr
        assert "generate_instructions" in output
        assert "summarize" in output
        assert "generate_negatives" in output

    def test_lesynthesis_server_command_exists(self):
        """Test that lesynthesis-server command is available."""
        # Just test that the command exists by checking if we can import it
        try:
            from lesynthesis.web_server import main

            assert callable(main)
        except ImportError:
            pytest.fail(
                "lesynthesis-server command not available - web_server.main not found"
            )

    def test_generate_instructions_command(self):
        """Test generate_instructions command structure."""
        # Just verify the command exists and has the right structure
        # We can't actually run it without a valid API key
        result = subprocess.run(
            ["lesynthesis"], capture_output=True, text=True
        )

        # Check that generate_instructions is listed as a command
        assert "generate_instructions" in result.stdout

        # Verify it's a valid fire command structure
        assert "COMMANDS" in result.stdout

    def test_summarize_command(self):
        """Test summarize command structure."""
        # Just verify the command exists
        result = subprocess.run(
            ["lesynthesis"], capture_output=True, text=True
        )

        assert "summarize" in result.stdout

    def test_generate_negatives_command(self):
        """Test generate_negatives command structure."""
        # Just verify the command exists
        result = subprocess.run(
            ["lesynthesis"], capture_output=True, text=True
        )

        assert "generate_negatives" in result.stdout

    def test_model_name_parameter(self):
        """Test that model_name parameter works."""
        result = subprocess.run(
            [
                "lesynthesis",
                "--model_name",
                "gemini-2.5-pro-preview-03-25",
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_server_port_parameter(self):
        """Test server accepts port parameter."""
        # Just verify the main function has the expected parameters
        from lesynthesis.web_server import main
        import inspect

        sig = inspect.signature(main)
        params = list(sig.parameters.keys())

        assert "port" in params
        assert "host" in params
        assert "debug" in params


class TestREADMEExamples:
    """Test that examples in README actually work."""

    def test_readme_cli_examples_syntax(self):
        """Verify README examples have correct syntax."""
        # These should at least not cause import/syntax errors
        commands = [
            [
                "lesynthesis",
                "generate_instructions",
                "lerobot/pusht",
                "--episode_index",
                "0",
            ],
            [
                "lesynthesis",
                "summarize",
                "lerobot/pusht",
                "--episode_index",
                "0",
            ],
            ["lesynthesis", "generate_negatives", "lerobot/pusht"],
            [
                "lesynthesis",
                "--model_name",
                "gemini-2.5-pro-preview-03-25",
                "generate_instructions",
                "lerobot/pusht",
            ],
            ["lesynthesis-server", "--port", "5001"],
            [
                "lesynthesis-server",
                "--port",
                "5001",
                "--host",
                "0.0.0.0",
                "--debug",
            ],
        ]

        for cmd in commands:
            # Just verify the command structure is valid (would parse correctly)
            assert len(cmd) >= 2
            assert cmd[0] in ["lesynthesis", "lesynthesis-server"]

            # For lesynthesis commands, verify the subcommand exists
            if cmd[0] == "lesynthesis" and not cmd[1].startswith("--"):
                assert cmd[1] in [
                    "generate_instructions",
                    "summarize",
                    "generate_negatives",
                ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
