#!/usr/bin/env python

"""
Tests for all examples shown in the README.
"""

import os
from unittest.mock import Mock, patch

import pytest

from lesynthesis import CaptionSynthesizer


class TestReadmeExamples:
    """Test all examples from the README."""

    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        with patch("lesynthesis.synthesizer._setup_generative_model") as mock:
            mock_model = Mock()

            # Mock response for high-level instruction
            mock_model.generate_content.return_value = Mock(
                text="""
===HIGH_LEVEL_START===
Pick up the red cube from the table and transfer it to the blue container
===HIGH_LEVEL_END===

===MID_LEVEL_START===
1. Approach the red cube on the table surface
2. Position the gripper above the cube and align for optimal grasp
3. Close the gripper to secure the cube
4. Lift and transport the cube to the blue container
5. Lower the cube into the container and release
===MID_LEVEL_END===

===LOW_LEVEL_START===
Move arm forward | Move arm forward 15cm while maintaining 10cm height above table | 0.0s
Rotate wrist | Rotate wrist 45° clockwise for perpendicular approach angle | 0.5s
Lower gripper | Lower gripper 8cm until 2cm above cube surface | 1.0s
Close gripper | Actuate gripper to close around cube | 1.5s
Lift cube | Raise arm 20cm vertically | 2.0s
Move to container | Translate arm 30cm horizontally to blue container | 2.5s
Lower into container | Lower arm 15cm into container | 3.5s
Release cube | Open gripper to release cube | 4.0s
===LOW_LEVEL_END===
"""
            )
            mock.return_value = mock_model
            yield mock_model

    @pytest.fixture
    def enrichment_tool(self, mock_model):
        """Create an enrichment tool with mocked model."""
        # Set a dummy API key for testing
        os.environ["GOOGLE_API_KEY"] = "test-api-key"
        tool = CaptionSynthesizer()
        yield tool
        # Clean up
        if "GOOGLE_API_KEY" in os.environ:
            del os.environ["GOOGLE_API_KEY"]

    def test_basic_usage_example(self, enrichment_tool):
        """Test the basic usage example from README."""
        # Mock the dataset
        with patch("lesynthesis.synthesizer.LeRobotDataset") as mock_dataset:
            # Setup mock dataset
            mock_dataset.return_value = Mock(
                episode_data_index={
                    "from": [Mock(item=lambda: 0)],
                    "to": [Mock(item=lambda: 100)],
                },
                hf_dataset=Mock(
                    select=Mock(
                        return_value={
                            "task_index": [Mock(item=lambda: 0)] * 100,
                            "action": [
                                Mock(
                                    numpy=lambda: [
                                        0.1,
                                        0.2,
                                        0.3,
                                        0.4,
                                        0.5,
                                        0.6,
                                    ]
                                )
                                for _ in range(100)
                            ],
                        }
                    )
                ),
                meta=Mock(tasks={0: "Pick up cube"}, robot_type="aloha"),
                fps=30,
                features={},
            )

            # Test generate_instructions
            instructions = enrichment_tool.generate_instructions(
                dataset_repo_id="lerobot/pusht", episode_index=0
            )

            # Verify the structure matches README example
            assert isinstance(instructions, dict)
            assert "high_level" in instructions
            assert "mid_level" in instructions
            assert "low_level" in instructions

            # Check high-level instruction
            assert (
                instructions["high_level"]
                == "Pick up the red cube from the table and transfer it to the blue container"
            )

            # Check mid-level instructions
            assert len(instructions["mid_level"]) == 5
            assert (
                instructions["mid_level"][0]
                == "1. Approach the red cube on the table surface"
            )
            assert (
                instructions["mid_level"][1]
                == "2. Position the gripper above the cube and align for optimal grasp"
            )

            # Check low-level instructions
            assert len(instructions["low_level"]) == 8
            assert instructions["low_level"][0]["action"] == "Move arm forward"
            assert (
                instructions["low_level"][0]["detail"]
                == "Move arm forward 15cm while maintaining 10cm height above table"
            )
            assert instructions["low_level"][0]["timing"] == "0.0s"

    def test_python_api_example(self, enrichment_tool):
        """Test the Python API example from README."""
        with patch("lesynthesis.synthesizer.LeRobotDataset") as mock_dataset:
            # Setup mock dataset
            mock_dataset.return_value = Mock(
                episode_data_index={
                    "from": [Mock(item=lambda: 0)],
                    "to": [Mock(item=lambda: 100)],
                },
                hf_dataset=Mock(
                    select=Mock(
                        return_value={
                            "task_index": [Mock(item=lambda: 0)] * 100,
                            "action": [
                                Mock(
                                    numpy=lambda: [
                                        0.1,
                                        0.2,
                                        0.3,
                                        0.4,
                                        0.5,
                                        0.6,
                                    ]
                                )
                                for _ in range(100)
                            ],
                        }
                    )
                ),
                meta=Mock(tasks={0: "Pick up cube"}, robot_type="aloha"),
                fps=30,
                features={},
            )

            # Generate rich captions for a robot trajectory
            instructions = enrichment_tool.generate_instructions(
                dataset_repo_id="lerobot/pusht", episode_index=0
            )

            # Access multi-level captions as shown in README
            goal = instructions["high_level"]
            steps = instructions["mid_level"]
            actions = instructions["low_level"]

            # Verify we can access the data as shown
            assert isinstance(goal, str)
            assert isinstance(steps, list)
            assert isinstance(actions, list)
            assert len(goal) > 0
            assert len(steps) > 0
            assert len(actions) > 0

    def test_transformation_example(self, enrichment_tool):
        """Test the transformation example from README."""
        with patch("lesynthesis.synthesizer.LeRobotDataset") as mock_dataset:
            # Setup mock dataset with simple task
            mock_dataset.return_value = Mock(
                episode_data_index={
                    "from": [Mock(item=lambda: 0)],
                    "to": [Mock(item=lambda: 100)],
                },
                hf_dataset=Mock(
                    select=Mock(
                        return_value={
                            "task_index": [Mock(item=lambda: 0)] * 100,
                            "action": [
                                Mock(
                                    numpy=lambda: [
                                        0.1,
                                        0.2,
                                        0.3,
                                        0.4,
                                        0.5,
                                        0.6,
                                    ]
                                )
                                for _ in range(100)
                            ],
                        }
                    )
                ),
                meta=Mock(tasks={0: "Pick up cube"}, robot_type="aloha"),
                fps=30,
                features={},
            )

            # Generate instructions
            instructions = enrichment_tool.generate_instructions(
                dataset_repo_id="lerobot/pusht", episode_index=0
            )

            # Verify transformation from simple to rich
            original_task = "Pick up cube"

            # High-level should be more detailed than original
            assert len(instructions["high_level"]) > len(original_task)
            assert "cube" in instructions["high_level"].lower()

            # Mid-level should have multiple phases
            assert len(instructions["mid_level"]) >= 3

            # Low-level should have detailed actions
            assert len(instructions["low_level"]) >= 5
            for action in instructions["low_level"]:
                if isinstance(action, dict):
                    assert "action" in action
                    assert "detail" in action or "timing" in action

    def test_negative_examples_generation(self, enrichment_tool):
        """Test negative example generation."""
        with patch("lesynthesis.synthesizer.LeRobotDataset") as mock_dataset:
            # Setup mock dataset
            mock_dataset.return_value = Mock(
                meta=Mock(tasks={0: "Pick up cube", 1: "Open drawer"})
            )

            # Mock the model response for negatives
            enrichment_tool._model.generate_content.return_value = Mock(
                text="""1. Approaching the cube too quickly without proper alignment
2. Gripping too hard and potentially damaging the cube
3. Moving the arm without checking for obstacles
4. Releasing the cube from too high above the container
5. Failing to verify the cube is securely grasped before lifting"""
            )

            # Generate negatives
            negatives = enrichment_tool.generate_negatives(
                dataset_repo_id="lerobot/pusht"
            )

            # Verify structure
            assert isinstance(negatives, dict)
            assert "Pick up cube" in negatives
            assert "Open drawer" in negatives

            # Verify content
            assert "too quickly" in negatives["Pick up cube"]
            assert "Gripping too hard" in negatives["Pick up cube"]

    def test_trajectory_summarization(self, enrichment_tool):
        """Test trajectory summarization."""
        with patch("lesynthesis.synthesizer.LeRobotDataset") as mock_dataset:
            # Setup mock dataset
            mock_dataset.return_value = Mock(
                episode_data_index={
                    "from": [Mock(item=lambda: 0)],
                    "to": [Mock(item=lambda: 100)],
                },
                hf_dataset=Mock(
                    select=Mock(
                        return_value=Mock(
                            __getitem__=lambda self, key: {
                                "task_index": [Mock(item=lambda: 0)] * 100,
                                "action": [
                                    Mock(
                                        numpy=lambda: [
                                            0.1,
                                            0.2,
                                            0.3,
                                            0.4,
                                            0.5,
                                            0.6,
                                        ]
                                    )
                                    for _ in range(100)
                                ],
                                "observation.state": [
                                    Mock(numpy=lambda: [0.1] * 10)
                                    for _ in range(100)
                                ],
                            }[key],
                            column_names=[
                                "task_index",
                                "action",
                                "observation.state",
                            ],
                        )
                    ),
                ),
                meta=Mock(tasks={0: "Pick up cube"}, robot_type="aloha"),
                fps=30,
                features={
                    "action": {"shape": [6], "dtype": "float32"},
                    "observation.state": {"shape": [10], "dtype": "float32"},
                },
            )

            # Mock the model response for summary
            enrichment_tool._model.generate_content.return_value = Mock(
                text="""The robot executes a pick-and-place task over 3.33 seconds. 
The trajectory begins with the arm approaching the cube location, 
followed by a precise grasping motion. The robot then lifts the cube 
and transfers it to the target location with smooth, controlled movements."""
            )

            # Capture console output
            captured_output = []
            original_print = enrichment_tool._console.print
            enrichment_tool._console.print = (
                lambda *args, **kwargs: captured_output.append(
                    str(args[0]) if args else ""
                )
            )

            # Generate summary
            enrichment_tool.summarize(
                dataset_repo_id="lerobot/pusht", episode_index=0
            )

            # Restore original print
            enrichment_tool._console.print = original_print

            # Verify summary was generated
            assert len(captured_output) > 0

            # We should have captured a Panel object (as a string representation)
            panel_found = False
            for output in captured_output:
                if "Panel object" in str(output) or "panel.Panel" in str(
                    output
                ):
                    panel_found = True
                    break

            assert (
                panel_found
            ), f"No Panel object found in outputs: {captured_output}"

    def test_error_handling_no_api_key(self):
        """Test error handling when no API key is provided."""
        # Remove API key if present
        api_key = os.environ.pop("GOOGLE_API_KEY", None)

        try:
            # Mock the setup function to raise ValueError
            with patch(
                "lesynthesis.synthesizer._setup_generative_model"
            ) as mock_setup:
                mock_setup.side_effect = ValueError(
                    "Google API key is required"
                )

                with pytest.raises(
                    ValueError, match="Google API key is required"
                ):
                    CaptionSynthesizer()
        finally:
            # Restore API key if it was present
            if api_key:
                os.environ["GOOGLE_API_KEY"] = api_key

    def test_model_initialization_with_custom_model(self):
        """Test initialization with custom model name."""
        os.environ["GOOGLE_API_KEY"] = "test-api-key"

        with patch(
            "lesynthesis.synthesizer._setup_generative_model"
        ) as mock_setup:
            mock_setup.return_value = Mock()

            # Test with custom model - pass api_key explicitly
            tool = CaptionSynthesizer(
                model_name="gemini-2.5-pro-preview-03-25",
                api_key="test-api-key",
            )

            # Verify the model name was passed correctly
            mock_setup.assert_called_with(
                "gemini-2.5-pro-preview-03-25", "test-api-key"
            )
            assert tool.model_name == "gemini-2.5-pro-preview-03-25"

    def test_instruction_structure_validation(self, enrichment_tool):
        """Test that generated instructions have the correct structure."""
        with patch("lesynthesis.synthesizer.LeRobotDataset") as mock_dataset:
            # Setup mock dataset
            mock_dataset.return_value = Mock(
                episode_data_index={
                    "from": [Mock(item=lambda: 0)],
                    "to": [Mock(item=lambda: 50)],
                },
                hf_dataset=Mock(
                    select=Mock(
                        return_value={
                            "task_index": [Mock(item=lambda: 0)] * 50,
                            "action": [
                                Mock(
                                    numpy=lambda: [
                                        0.1,
                                        0.2,
                                        0.3,
                                        0.4,
                                        0.5,
                                        0.6,
                                    ]
                                )
                                for _ in range(50)
                            ],
                        }
                    )
                ),
                meta=Mock(tasks={0: "Test task"}, robot_type="test_robot"),
                fps=30,
                features={},
            )

            # Generate instructions
            instructions = enrichment_tool.generate_instructions(
                dataset_repo_id="test/dataset", episode_index=0
            )

            # Validate structure
            assert isinstance(instructions, dict)
            assert set(instructions.keys()) == {
                "high_level",
                "mid_level",
                "low_level",
                "raw_response",
            }

            # Validate high-level
            assert isinstance(instructions["high_level"], str)
            assert len(instructions["high_level"]) > 0

            # Validate mid-level
            assert isinstance(instructions["mid_level"], list)
            assert all(
                isinstance(item, str) for item in instructions["mid_level"]
            )

            # Validate low-level
            assert isinstance(instructions["low_level"], list)
            for item in instructions["low_level"]:
                assert isinstance(item, dict)
                assert "action" in item
                assert isinstance(item["action"], str)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
