#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Tests for the LLM enrichment tool.
"""

import os
from unittest.mock import MagicMock, patch

import pytest

from lerobot.cli.enrich_with_llm import LLMEnrichmentTool

# Skip all tests in this file if the GOOGLE_API_KEY is not set.
pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY is not set"
)


class TestLiveLLMEnrichment:
    """Test the LLM enrichment tool with real API calls."""

    def test_summarize_trajectory_live(self):
        """
        Tests the full workflow of summarizing a real trajectory,
        verifying that the output is a non-empty string.
        """
        # Use a small dataset that's likely to be cached
        dataset_repo_id = "lerobot/aloha_sim_transfer_cube_human"

        tool = LLMEnrichmentTool()

        # Mock the console to capture output
        with patch.object(tool, "_console") as mock_console:
            # Run the summarization
            tool.summarize(dataset_repo_id=dataset_repo_id, episode_index=0)

            # Verify that print was called
            assert mock_console.print.called

            # Check that we printed the task header
            task_header_calls = [
                call
                for call in mock_console.print.call_args_list
                if "[bold]Task: Trajectory Summarization" in str(call.args[0])
            ]
            assert len(task_header_calls) > 0

            # Check that we printed a Panel with the summary
            panel_calls = [
                call
                for call in mock_console.print.call_args_list
                if hasattr(call.args[0], "__class__")
                and call.args[0].__class__.__name__ == "Panel"
            ]
            assert len(panel_calls) > 0

            # Verify the panel contains green text (the summary)
            panel = panel_calls[0].args[0]
            assert "[green]" in panel.renderable
            assert (
                len(panel.renderable) > 50
            )  # Should have substantial content

    def test_generate_negatives_live(self):
        """
        Tests the full workflow of generating negative examples,
        verifying the structure of the output.
        """
        # Use a small dataset that's likely to be cached
        dataset_repo_id = "lerobot/aloha_sim_transfer_cube_human"

        tool = LLMEnrichmentTool()

        # Mock the console to capture output
        with patch.object(tool, "_console") as mock_console:
            # Run the negative generation
            result = tool.generate_negatives(dataset_repo_id=dataset_repo_id)

            # Verify that print was called
            assert mock_console.print.called

            # Check that we printed the task header
            task_header_calls = [
                call
                for call in mock_console.print.call_args_list
                if "[bold cyan]Task: Negative Data Generation"
                in str(call.args[0])
            ]
            assert len(task_header_calls) > 0

            # Check that we found tasks
            found_tasks_calls = [
                call
                for call in mock_console.print.call_args_list
                if "Found" in str(call.args[0])
                and "unique task(s)" in str(call.args[0])
            ]
            assert len(found_tasks_calls) > 0

            # Check that we printed panels with negative examples
            panel_calls = [
                call
                for call in mock_console.print.call_args_list
                if hasattr(call.args[0], "__class__")
                and call.args[0].__class__.__name__ == "Panel"
            ]
            assert len(panel_calls) > 0

            # Verify the panel contains red text (the negative examples)
            panel = panel_calls[0].args[0]
            assert "[red]" in panel.renderable
            assert (
                len(panel.renderable) > 50
            )  # Should have substantial content

            # Verify the function returns a dictionary with results
            assert isinstance(result, dict)
            assert len(result) > 0
            for task, negatives in result.items():
                assert isinstance(task, str)
                assert isinstance(negatives, str)
                assert len(negatives) > 50  # Should have substantial content
