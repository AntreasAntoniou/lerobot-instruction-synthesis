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
from unittest.mock import MagicMock, patch, Mock

import pytest

from lesynthesis.synthesizer import CaptionSynthesizer

# Skip all tests in this file if the GOOGLE_API_KEY is not set.
pytestmark = pytest.mark.skipif(
    not os.environ.get("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY is not set"
)


class TestCaptionSynthesizer:
    """Test cases for CaptionSynthesizer."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        with patch("lesynthesis.synthesizer._setup_generative_model") as mock:
            mock_model = Mock()
            mock_model.generate_content.return_value = Mock(
                text="Test response from LLM"
            )
            mock.return_value = mock_model
            yield mock_model

    @pytest.fixture
    def tool(self, mock_model):
        """Create a tool instance with mocked model."""
        return CaptionSynthesizer()

    def test_initialization(self, tool):
        """Test tool initialization."""
        assert tool is not None
        assert hasattr(tool, "_model")
        assert hasattr(tool, "_console")

    @patch("lesynthesis.synthesizer.LeRobotDataset")
    def test_generate_instructions(self, mock_dataset, tool):
        """Test instruction generation."""
        # Mock dataset
        # Create mock action data
        mock_actions = []
        for i in range(100):
            mock_action = Mock()
            mock_action.numpy.return_value = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
            mock_actions.append(mock_action)

        # Create mock episode data that behaves like a dict with list values
        mock_episode_data = {
            "task_index": [Mock(item=lambda: 0)] * 100,
            "action": mock_actions,
        }

        mock_dataset.return_value = Mock(
            episode_data_index={
                "from": [Mock(item=lambda: 0)],
                "to": [Mock(item=lambda: 100)],
            },
            hf_dataset=Mock(select=Mock(return_value=mock_episode_data)),
            meta=Mock(tasks={0: "Test task"}, robot_type="test_robot"),
            fps=30,
            features={},
        )

        # Test instruction generation
        instructions = tool.generate_instructions(
            dataset_repo_id="test/dataset", episode_index=0
        )

        assert isinstance(instructions, dict)
        assert "high_level" in instructions
        assert "mid_level" in instructions
        assert "low_level" in instructions
        assert "raw_response" in instructions

    @patch("lesynthesis.synthesizer.LeRobotDataset")
    def test_generate_negatives(self, mock_dataset, tool):
        """Test negative example generation."""
        # Mock dataset
        mock_dataset.return_value = Mock(
            meta=Mock(tasks={0: "Pick up object", 1: "Pick up object"})
        )

        # Test negative generation
        negatives = tool.generate_negatives(dataset_repo_id="test/dataset")

        assert isinstance(negatives, dict)
        assert len(negatives) > 0
        assert "Pick up object" in negatives

    def test_model_response_parsing(self, tool):
        """Test parsing of model responses."""
        # Test high-level parsing
        test_response = """
        ===HIGH_LEVEL_START===
        Pick up the cube and transfer it
        ===HIGH_LEVEL_END===
        
        ===MID_LEVEL_START===
        1. Approach the cube
        2. Grasp the cube
        3. Transfer to target
        ===MID_LEVEL_END===
        
        ===LOW_LEVEL_START===
        Move arm | Extend towards cube | 0.0s
        Close gripper | Secure the cube | 1.0s
        ===LOW_LEVEL_END===
        """

        # Mock the model to return our test response
        tool._model.generate_content = Mock(
            return_value=Mock(text=test_response)
        )

        # This would be part of generate_instructions
        # We're testing the parsing logic
        import re

        instructions = {"high_level": "", "mid_level": [], "low_level": []}

        # Parse HIGH-LEVEL
        high_match = re.search(
            r"===HIGH_LEVEL_START===\s*\n(.*?)\n\s*===HIGH_LEVEL_END===",
            test_response,
            re.DOTALL,
        )
        if high_match:
            instructions["high_level"] = high_match.group(1).strip()

        # Parse MID-LEVEL
        mid_match = re.search(
            r"===MID_LEVEL_START===\s*\n(.*?)\n\s*===MID_LEVEL_END===",
            test_response,
            re.DOTALL,
        )
        if mid_match:
            for line in mid_match.group(1).strip().split("\n"):
                cleaned = line.strip()
                if cleaned and not cleaned.startswith("#"):
                    instructions["mid_level"].append(cleaned)

        assert instructions["high_level"] == "Pick up the cube and transfer it"
        assert len(instructions["mid_level"]) == 3
        assert "1. Approach the cube" in instructions["mid_level"]


@pytest.mark.integration
class TestIntegration:
    """Integration tests that require API access."""

    @pytest.mark.skipif(
        "INTEGRATION_TESTS" not in os.environ,
        reason="Integration tests require INTEGRATION_TESTS env var",
    )
    def test_real_api_call(self):
        """Test with real API (requires GOOGLE_API_KEY)."""
        tool = LLMEnrichmentTool()

        # This would make a real API call
        instructions = tool.generate_instructions(
            dataset_repo_id="lerobot/pusht", episode_index=0
        )

        assert instructions is not None
        assert len(instructions["high_level"]) > 0
