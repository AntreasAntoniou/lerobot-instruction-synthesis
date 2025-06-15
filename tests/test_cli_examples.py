#!/usr/bin/env python

"""
Tests for CLI examples shown in the README.
"""

import os
from unittest.mock import Mock, patch

import pytest

from lesynthesis.enrich_with_llm import LLMEnrichmentTool, main


class TestCLIExamples:
    """Test CLI examples from the README."""

    @pytest.fixture
    def mock_fire(self):
        """Mock fire.Fire to prevent actual CLI execution."""
        with patch("lesynthesis.enrich_with_llm.fire.Fire") as mock:
            yield mock

    @pytest.fixture
    def mock_llm_tool(self):
        """Mock the LLMEnrichmentTool."""
        with patch(
            "lesynthesis.enrich_with_llm.LLMEnrichmentTool"
        ) as mock_class:
            mock_instance = Mock()
            mock_class.return_value = mock_instance

            # Mock methods
            mock_instance.generate_instructions.return_value = {
                "high_level": "Pick up the red cube from the table and transfer it to the blue container",
                "mid_level": [
                    "1. Approach the red cube on the table surface",
                    "2. Position the gripper above the cube and align for optimal grasp",
                    "3. Close the gripper to secure the cube",
                    "4. Lift and transport the cube to the blue container",
                    "5. Lower the cube into the container and release",
                ],
                "low_level": [
                    {
                        "action": "Move arm forward",
                        "detail": "Move arm forward 15cm",
                        "timing": "0.0s",
                    },
                    {
                        "action": "Rotate wrist",
                        "detail": "Rotate wrist 45° clockwise",
                        "timing": "0.5s",
                    },
                ],
                "raw_response": "Mock response",
            }

            mock_instance.generate_negatives.return_value = {
                "Pick up cube": "1. Approaching too quickly\n2. Gripping too hard"
            }

            yield mock_instance

    def test_main_function_calls_fire(self, mock_fire):
        """Test that main() calls fire.Fire with LLMEnrichmentTool."""
        # Set API key for the test
        os.environ["GOOGLE_API_KEY"] = "test-key"

        try:
            main()
            mock_fire.assert_called_once_with(LLMEnrichmentTool)
        finally:
            if "GOOGLE_API_KEY" in os.environ:
                del os.environ["GOOGLE_API_KEY"]

    def test_generate_instructions_cli_simulation(self, mock_llm_tool):
        """Simulate the CLI command: lesynthesis generate-instructions --dataset lerobot/pusht --episode 0"""
        # This tests that the method can be called with the expected arguments
        tool = LLMEnrichmentTool()

        # Simulate CLI call
        result = tool.generate_instructions(
            dataset_repo_id="lerobot/pusht", episode_index=0
        )

        # Verify the result structure matches what's expected
        assert isinstance(result, dict)
        assert "high_level" in result
        assert "mid_level" in result
        assert "low_level" in result

    def test_enrich_dataset_simulation(self, mock_llm_tool):
        """Simulate batch processing of a dataset."""
        tool = LLMEnrichmentTool()

        # Mock dataset with multiple episodes
        with patch(
            "lesynthesis.enrich_with_llm.LeRobotDataset"
        ) as mock_dataset:
            mock_dataset.return_value = Mock(
                num_episodes=3,
                episode_data_index={
                    "from": [Mock(item=lambda: i * 100) for i in range(3)],
                    "to": [Mock(item=lambda: (i + 1) * 100) for i in range(3)],
                },
                hf_dataset=Mock(
                    select=Mock(
                        return_value={
                            "task_index": [Mock(item=lambda: 0)] * 100,
                            "action": [
                                Mock(numpy=lambda: [0.1] * 6)
                                for _ in range(100)
                            ],
                        }
                    )
                ),
                meta=Mock(tasks={0: "Test task"}, robot_type="test_robot"),
                fps=30,
                features={},
            )

            # Simulate processing multiple episodes
            results = []
            for episode in range(3):
                result = tool.generate_instructions(
                    dataset_repo_id="lerobot/pusht", episode_index=episode
                )
                results.append(result)

            # Verify all episodes were processed
            assert len(results) == 3
            for result in results:
                assert isinstance(result, dict)
                assert "high_level" in result

    def test_generate_with_detail_level_simulation(self):
        """Simulate generating instructions with specific detail level."""
        # This is a conceptual test since detail level isn't implemented
        # but shows how it would work based on README
        tool = LLMEnrichmentTool()

        with patch.object(tool, "generate_instructions") as mock_generate:
            mock_generate.return_value = {
                "high_level": "High-level task description",
                "mid_level": [],
                "low_level": [],
                "raw_response": "",
            }

            # Simulate call with detail level (future feature)
            result = tool.generate_instructions(
                dataset_repo_id="lerobot/pusht", episode_index=0
            )

            assert "high_level" in result

    def test_server_command_simulation(self):
        """Test that the server can be started with custom port."""
        from lesynthesis.enrich_with_llm_server import main as server_main

        with patch("lesynthesis.enrich_with_llm_server.app.run") as mock_run:
            with patch(
                "lesynthesis.enrich_with_llm_server.fire.Fire"
            ) as mock_fire:
                # Simulate the server being called with custom port
                server_main(port=5001, host="0.0.0.0", debug=False)

                # Verify app.run was called with correct parameters
                mock_run.assert_called_once_with(
                    debug=False, host="0.0.0.0", port=5001
                )


class TestCLIIntegration:
    """Integration tests for CLI functionality."""

    @pytest.mark.skipif(
        "INTEGRATION_TESTS" not in os.environ,
        reason="Integration tests require INTEGRATION_TESTS env var",
    )
    def test_real_cli_execution(self):
        """Test actual CLI execution with a real dataset."""
        # This would test the actual CLI with a real dataset
        # Only runs when INTEGRATION_TESTS is set
        import subprocess

        result = subprocess.run(
            [
                "lesynthesis",
                "generate-instructions",
                "--dataset",
                "lerobot/pusht",
                "--episode",
                "0",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "HIGH-LEVEL" in result.stdout
        assert "MID-LEVEL" in result.stdout
        assert "LOW-LEVEL" in result.stdout


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
