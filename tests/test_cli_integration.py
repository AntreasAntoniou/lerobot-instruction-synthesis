#!/usr/bin/env python

"""
Integration tests for CLI commands.
These tests actually run the commands and verify they work correctly.
Run with: INTEGRATION_TESTS=1 pytest tests/test_cli_integration.py
"""

import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pytest


@pytest.mark.skipif(
    not os.environ.get("INTEGRATION_TESTS"),
    reason="Integration tests require INTEGRATION_TESTS env var",
)
class TestCLIIntegration:
    """Integration tests that actually run the CLI commands."""

    @pytest.fixture
    def api_key(self):
        """Ensure API key is available for tests."""
        key = os.environ.get("GOOGLE_API_KEY")
        if not key:
            pytest.skip("GOOGLE_API_KEY required for integration tests")
        return key

    def test_generate_instructions_integration(self, api_key):
        """Test generate_instructions command actually works."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lesynthesis",
                "generate_instructions",
                "lerobot/aloha_sim_transfer_cube_human",
                "--episode_index",
                "0",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "GOOGLE_API_KEY": api_key},
        )

        assert result.returncode == 0
        assert "HIGH-LEVEL:" in result.stdout
        assert "MID-LEVEL PHASES:" in result.stdout
        assert "LOW-LEVEL ACTIONS:" in result.stdout

    def test_summarize_integration(self, api_key):
        """Test summarize command actually works."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lesynthesis",
                "summarize",
                "lerobot/aloha_sim_transfer_cube_human",
                "--episode_index",
                "0",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "GOOGLE_API_KEY": api_key},
        )

        assert result.returncode == 0
        assert (
            "Trajectory Summary" in result.stdout
            or "trajectory" in result.stdout.lower()
        )

    def test_generate_negatives_integration(self, api_key):
        """Test generate_negatives command actually works."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lesynthesis",
                "generate_negatives",
                "lerobot/aloha_sim_transfer_cube_human",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "GOOGLE_API_KEY": api_key},
        )

        assert result.returncode == 0
        assert "Pitfall" in result.stdout or "pitfall" in result.stdout.lower()

    def test_custom_model_integration(self, api_key):
        """Test using custom model name."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lesynthesis",
                "--model_name",
                "gemini-2.5-flash-preview-05-20",
                "generate_instructions",
                "lerobot/aloha_sim_transfer_cube_human",
                "--episode_index",
                "0",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "GOOGLE_API_KEY": api_key},
        )

        assert result.returncode == 0
        assert "HIGH-LEVEL:" in result.stdout

    def test_server_starts_and_stops(self, api_key):
        """Test that server can start and be stopped."""
        # Start server in background
        proc = subprocess.Popen(
            ["lesynthesis-server", "--port", "7778"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "GOOGLE_API_KEY": api_key},
        )

        # Give it time to start
        time.sleep(2)

        try:
            # Check if it's running
            assert proc.poll() is None, "Server failed to start"

            # Try to connect
            import requests

            try:
                response = requests.get("http://localhost:7778", timeout=5)
                assert response.status_code == 200
            except requests.exceptions.ConnectionError:
                pytest.fail("Could not connect to server")

        finally:
            # Clean up
            proc.terminate()
            proc.wait(timeout=5)


@pytest.mark.skipif(
    not os.environ.get("INTEGRATION_TESTS"),
    reason="Integration tests require INTEGRATION_TESTS env var",
)
class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_missing_api_key_error(self):
        """Test helpful error when API key is missing."""
        # Remove API key
        env = os.environ.copy()
        env.pop("GOOGLE_API_KEY", None)

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
            env=env,
        )

        # Should fail with helpful message
        assert result.returncode != 0
        assert "API key" in result.stderr or "api key" in result.stderr.lower()

    def test_invalid_dataset_error(self):
        """Test error handling for invalid dataset."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lesynthesis",
                "generate_instructions",
                "invalid/dataset",
                "--episode_index",
                "0",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "GOOGLE_API_KEY": "test-key"},
        )

        assert result.returncode != 0

    def test_invalid_episode_index(self):
        """Test error handling for invalid episode index."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "lesynthesis",
                "generate_instructions",
                "lerobot/pusht",
                "--episode_index",
                "99999",
            ],
            capture_output=True,
            text=True,
            env={**os.environ, "GOOGLE_API_KEY": "test-key"},
        )

        assert result.returncode != 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
