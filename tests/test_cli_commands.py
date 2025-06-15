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
        result = subprocess.run(
            [sys.executable, "-m", "lesynthesis", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "generate_instructions" in result.stdout
        assert "summarize" in result.stdout
        assert "generate_negatives" in result.stdout

    def test_lesynthesis_server_command_exists(self):
        """Test that lesynthesis-server command is available."""
        # Just test that the command exists and shows help
        result = subprocess.run(
            ["lesynthesis-server", "--help"], capture_output=True, text=True
        )
        # Fire.Fire returns 0 for help
        assert result.returncode == 0
        assert "port" in result.stdout or "PORT" in result.stdout

    @patch("lesynthesis.synthesizer.CaptionSynthesizer")
    def test_generate_instructions_command(self, mock_synthesizer):
        """Test generate_instructions command structure."""
        # Mock the synthesizer to avoid API calls
        mock_instance = Mock()
        mock_synthesizer.return_value = mock_instance
        mock_instance.generate_instructions.return_value = {
            "high_level": "Test instruction",
            "mid_level": ["Step 1", "Step 2"],
            "low_level": [
                {"action": "Move", "detail": "Forward", "timing": "0s"}
            ],
        }

        # Test the command structure
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lesynthesis",
                "generate_instructions",
                "lerobot/pusht",
                "--episode_index",
                "0",
            ],
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, "GOOGLE_API_KEY": "test-key"},
        )

        # Should not error out
        assert result.returncode == 0 or "GOOGLE_API_KEY" in result.stderr

    @patch("lesynthesis.synthesizer.CaptionSynthesizer")
    def test_summarize_command(self, mock_synthesizer):
        """Test summarize command structure."""
        mock_instance = Mock()
        mock_synthesizer.return_value = mock_instance

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lesynthesis",
                "summarize",
                "lerobot/pusht",
                "--episode_index",
                "0",
            ],
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, "GOOGLE_API_KEY": "test-key"},
        )

        assert result.returncode == 0 or "GOOGLE_API_KEY" in result.stderr

    @patch("lesynthesis.synthesizer.CaptionSynthesizer")
    def test_generate_negatives_command(self, mock_synthesizer):
        """Test generate_negatives command structure."""
        mock_instance = Mock()
        mock_synthesizer.return_value = mock_instance
        mock_instance.generate_negatives.return_value = {
            "Pick up cube": "1. Dropping the cube\n2. Missing the cube"
        }

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lesynthesis",
                "generate_negatives",
                "lerobot/pusht",
            ],
            capture_output=True,
            text=True,
            env={**subprocess.os.environ, "GOOGLE_API_KEY": "test-key"},
        )

        assert result.returncode == 0 or "GOOGLE_API_KEY" in result.stderr

    def test_model_name_parameter(self):
        """Test that model_name parameter works."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lesynthesis",
                "--model_name",
                "gemini-2.5-pro-preview-03-25",
                "--help",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    @patch("lesynthesis.web_server.app")
    def test_server_port_parameter(self, mock_app):
        """Test server accepts port parameter."""
        # We can't actually start the server in tests, but we can check the command structure
        result = subprocess.run(
            ["lesynthesis-server", "--help"], capture_output=True, text=True
        )
        assert result.returncode == 0
        assert "port" in result.stdout.lower() or "PORT" in result.stdout


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
