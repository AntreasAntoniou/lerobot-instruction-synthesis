"""
A script to enrich LeRobot datasets using a Large Language Model (LLM).

This tool provides two main functionalities:
1.  Automated Trajectory Summarization: It takes raw robot trajectory data from a dataset
    and uses an LLM to generate a natural language summary of the robot's actions.
2.  Generating "What Not To Do" Data (Negative Mining): It uses an LLM's world
    knowledge to generate textual descriptions of incorrect, unsafe, or inefficient ways
    to perform a given task.

This helps in transforming text-poor datasets into more descriptive and richer ones,
which is valuable for training more robust and context-aware robot policies.

Example usage:

# To summarize a trajectory from a specific episode
python -m lerobot.cli.enrich_with_llm summarize --dataset_repo_id="lerobot/aloha_sim_transfer_cube_human" --episode_index=0

# To generate negative data for a task
python -m lerobot.cli.enrich_with_llm generate_negatives --dataset_repo_id="lerobot/aloha_sim_transfer_cube_human"
"""

import os

import fire
import numpy as np
from rich.console import Console
from rich.panel import Panel

# Import genai and types with specific aliases as per guidelines
try:
    from google import genai
    from google.genai import types
except ImportError as e:
    raise ImportError(
        "Google Generative AI SDK not found. Please install it with: pip install google-generativeai"
    ) from e

# Try to import dotenv for .env file support
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.utils.utils import init_logging


def _setup_generative_model(
    model_name: str, api_key: str | None = None
) -> "genai.GenerativeModel":
    """Configures and returns a Gemini generative model instance."""
    if load_dotenv:
        load_dotenv()

    resolved_api_key = api_key or os.environ.get("GOOGLE_API_KEY")
    if not resolved_api_key:
        raise ValueError(
            "Google API key is required. Please pass it via the --api_key argument, "
            "set the GOOGLE_API_KEY environment variable, or place it in a .env file."
        )

    safety_settings = [
        types.SafetySetting(category=f"HARM_CATEGORY_{harm}", threshold="OFF")
        for harm in [
            "HATE_SPEECH",
            "DANGEROUS_CONTENT",
            "SEXUALLY_EXPLICIT",
            "HARASSMENT",
        ]
    ]

    # Create a client with the API key
    client = genai.Client(api_key=resolved_api_key)

    # Return a wrapper that includes the model name and safety settings
    class ModelWrapper:
        def __init__(self, client, model_name, safety_settings):
            self.client = client
            self.model_name = model_name
            self.safety_settings = safety_settings

        def generate_content(self, prompt):
            return self.client.models.generate_content(
                model=self.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    safety_settings=self.safety_settings
                ),
            )

    return ModelWrapper(client, model_name, safety_settings)


class CaptionSynthesizer:
    """Synthesizes rich, multi-level captions for robot trajectories using LLMs."""

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash-preview-05-20",
        api_key: str | None = None,
    ):
        self._model = _setup_generative_model(model_name, api_key)
        self._console = Console()

    def summarize(self, dataset_repo_id: str, episode_index: int = 0) -> None:
        """Summarize a trajectory from a dataset."""
        self._console.print(
            f"[bold]Task: Trajectory Summarization for episode {episode_index}[/bold]"
        )

        dataset = LeRobotDataset(dataset_repo_id)

        # Get episode data using episode_data_index
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()

        # Extract episode data
        episode_data = dataset.hf_dataset.select(range(from_idx, to_idx))

        # Get task information
        task_idx = episode_data["task_index"][0].item()
        task = dataset.meta.tasks[task_idx]

        # Get episode length and fps
        episode_length = to_idx - from_idx
        fps = dataset.fps

        # Create trajectory description
        trajectory_description = f"""
Task: {task}
Episode Index: {episode_index}
Episode Length: {episode_length} frames
Duration: {episode_length / fps:.2f} seconds
FPS: {fps}

Available modalities:
"""

        # Add information about available features
        for key, feature in dataset.features.items():
            if key not in [
                "index",
                "episode_index",
                "task_index",
                "timestamp",
                "frame_index",
            ]:
                shape = feature.get("shape", "N/A")
                dtype = feature.get("dtype", "N/A")
                trajectory_description += (
                    f"- {key}: shape={shape}, dtype={dtype}\n"
                )

        # Add some sample data points
        trajectory_description += "\nSample data from first and last frames:\n"

        # First frame
        trajectory_description += "\nFirst frame (t=0):\n"
        for key in ["observation.state", "action"]:
            if key in episode_data.column_names:
                value = episode_data[key][0]
                if hasattr(value, "numpy"):
                    value = value.numpy()
                trajectory_description += f"  {key}: {np.array(value)[:5]}... (showing first 5 values)\n"

        # Last frame
        trajectory_description += f"\nLast frame (t={episode_length-1}):\n"
        for key in ["observation.state", "action"]:
            if key in episode_data.column_names:
                value = episode_data[key][-1]
                if hasattr(value, "numpy"):
                    value = value.numpy()
                trajectory_description += f"  {key}: {np.array(value)[:5]}... (showing first 5 values)\n"

        prompt = f"""
Please provide a comprehensive summary of this robotic trajectory:

{trajectory_description}

Your summary should include:
1. A high-level description of what the robot is doing
2. Key phases or stages of the task
3. Notable patterns in the robot's movements
4. Any interesting observations about the execution

Please be concise but informative.
"""

        response = self._model.generate_content(prompt)
        summary = response.text

        # Display the summary
        self._console.print(
            Panel(
                f"[green]{summary}[/green]",
                title="Trajectory Summary",
                border_style="green",
            )
        )

    def generate_negatives(self, dataset_repo_id: str):
        """
        Uses the task description from a dataset to prompt the LLM for negative examples (pitfalls).

        Args:
            dataset_repo_id: The ID of the LeRobot dataset on the Hugging Face Hub.
        """
        self._console.print(
            "[bold cyan]Task: Pitfall Generation (Negative Examples)[/bold cyan]"
        )
        dataset = LeRobotDataset(dataset_repo_id)

        # Get all unique tasks from the dataset
        unique_tasks = list(set(dataset.meta.tasks.values()))

        self._console.print(
            f"Found {len(unique_tasks)} unique task(s) in the dataset:"
        )
        for i, task in enumerate(unique_tasks):
            self._console.print(f"  {i+1}. {task}")

        # For now, we'll generate negatives for all tasks
        all_negatives = {}

        for task_description in unique_tasks:
            self._console.print(
                f"\nGenerating pitfalls for task: '[italic]{task_description}[/italic]'"
            )

            prompt = f"""Generate a numbered list of 3-5 common pitfalls (negative examples) for the following robot task:
"{task_description}"

These pitfalls should describe incorrect, unsafe, or inefficient ways to perform the task. Focus on plausible mistakes that a robot might make.

IMPORTANT: Start your response directly with the numbered list. Do not include any preamble, introduction, or meta-commentary.

Format your response EXACTLY like this example:
1. [First pitfall description]
2. [Second pitfall description]
3. [Third pitfall description]

Now generate the pitfalls for the given task:"""

            self._console.print(
                "\n[yellow]Sending prompt to Gemini...[/yellow]"
            )
            response = self._model.generate_content(prompt)

            all_negatives[task_description] = response.text.strip()

            self._console.print(
                Panel(
                    f"[red]{response.text.strip()}[/red]",
                    title=f"Generated Pitfalls for: {task_description}",
                    border_style="red",
                )
            )

        return all_negatives

    def generate_instructions(
        self, dataset_repo_id: str, episode_index: int = 0
    ) -> dict:
        """
        Generate multi-level instructional descriptions for a robot trajectory.

        This creates instructions at various levels of detail:
        - High-level (global task)
        - Mid-level (subtasks/phases)
        - Low-level (specific actions)

        Args:
            dataset_repo_id: The ID of the LeRobot dataset
            episode_index: The episode to analyze

        Returns:
            Dictionary containing instructions at different levels
        """
        self._console.print(
            f"[bold magenta]Task: Multi-Level Instruction Generation for episode {episode_index}[/bold magenta]"
        )

        dataset = LeRobotDataset(dataset_repo_id)

        # Get episode data
        from_idx = dataset.episode_data_index["from"][episode_index].item()
        to_idx = dataset.episode_data_index["to"][episode_index].item()
        episode_data = dataset.hf_dataset.select(range(from_idx, to_idx))

        # Get task information
        task_idx = episode_data["task_index"][0].item()
        task = dataset.meta.tasks[task_idx]

        # Get episode details
        episode_length = to_idx - from_idx
        fps = dataset.fps
        duration = episode_length / fps

        # Extract action data for analysis
        actions = []
        for i in range(len(episode_data)):
            action = episode_data["action"][i]
            if hasattr(action, "numpy"):
                action = action.numpy()
            actions.append(action)
        actions = np.array(actions)

        # Analyze action patterns
        action_stats = {
            "mean": np.mean(actions, axis=0),
            "std": np.std(actions, axis=0),
            "max": np.max(actions, axis=0),
            "min": np.min(actions, axis=0),
        }

        # Create detailed trajectory information
        trajectory_info = f"""
Task: {task}
Episode: {episode_index}
Duration: {duration:.2f} seconds ({episode_length} frames at {fps} FPS)
Robot Type: {dataset.meta.robot_type or 'Unknown'}

Action Statistics:
- Dimensions: {actions.shape[1]}
- Mean values: {action_stats['mean'][:5]}... (first 5 shown)
- Std deviation: {action_stats['std'][:5]}... (first 5 shown)
- Range: [{action_stats['min'][:5]}...] to [{action_stats['max'][:5]}...]

Trajectory Phases (based on action analysis):
"""

        # Simple phase detection based on action changes
        phase_changes = []
        window_size = max(1, episode_length // 10)  # 10% of episode
        for i in range(window_size, episode_length - window_size, window_size):
            prev_mean = np.mean(actions[i - window_size : i], axis=0)
            curr_mean = np.mean(actions[i : i + window_size], axis=0)
            change = np.linalg.norm(curr_mean - prev_mean)
            if change > np.mean(action_stats["std"]):
                phase_changes.append((i, i / fps, change))

        for i, (frame, time, change) in enumerate(
            phase_changes[:5]
        ):  # Show up to 5 phases
            trajectory_info += f"- Phase {i+1}: Around {time:.1f}s (frame {frame}), significant change detected (magnitude: {change:.3f})\n"

        # Create the prompt for multi-level instructions
        prompt = f"""Based on this trajectory analysis, generate instructions at three levels of detail.

{trajectory_info}

Generate instructions in the following format with explicit separators:

===HIGH_LEVEL_START===
[Single sentence describing the overall goal]
===HIGH_LEVEL_END===

===MID_LEVEL_START===
[3-5 major phases or subtasks, one per line]
===MID_LEVEL_END===

===LOW_LEVEL_START===
[Detailed step-by-step actions, one per line in format: Action | Detail | Timing]
===LOW_LEVEL_END===

Requirements:
- Use ONLY the separators shown above
- Each instruction level must be between its START and END markers
- For low-level actions, use pipe (|) to separate Action, Detail, and Timing
- Be specific and technical
- Focus on what the robot should do, not meta-commentary"""

        self._console.print(
            "\n[yellow]Generating multi-level instructions...[/yellow]"
        )
        response = self._model.generate_content(prompt)

        instructions_text = response.text.strip()

        # Parse the response into structured format
        instructions = {
            "high_level": "",
            "mid_level": [],
            "low_level": [],
            "raw_response": instructions_text,
        }

        # Debug: print first part of response to see format
        self._console.print(
            "\n[dim]Debug - First 500 chars of response:[/dim]"
        )
        self._console.print(Panel(instructions_text[:500], border_style="dim"))

        # More robust parsing with regex
        import re

        # Parse HIGH-LEVEL instructions
        # Look for the section and extract the actual task description
        high_level_match = re.search(
            r"===HIGH_LEVEL_START===\s*\n(.*?)\n\s*===HIGH_LEVEL_END===",
            instructions_text,
            re.DOTALL,
        )
        if high_level_match:
            instructions["high_level"] = high_level_match.group(1).strip()
        else:
            # Fallback parsing
            high_level_section = re.search(
                r"(?:### )?1\.\s*HIGH-LEVEL.*?\n+(.+?)(?=\n\n|### 2|---)",
                instructions_text,
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            if high_level_section:
                # Extract the content and clean it
                content = high_level_section.group(1).strip()
                # Remove any sub-headers or formatting
                lines = content.split("\n")
                for line in lines:
                    cleaned = line.strip()
                    # Skip empty lines and formatting
                    if cleaned and not cleaned.startswith(("*", "-", "#")):
                        # Remove any leading "The robot" or similar
                        cleaned = re.sub(
                            r"^The robot['']s\s+",
                            "",
                            cleaned,
                            flags=re.IGNORECASE,
                        )
                        cleaned = re.sub(
                            r"^overall goal is to\s+",
                            "",
                            cleaned,
                            flags=re.IGNORECASE,
                        )
                        instructions["high_level"] = cleaned
                        break

        # Parse MID-LEVEL instructions
        mid_level_match = re.search(
            r"===MID_LEVEL_START===\s*\n(.*?)\n\s*===MID_LEVEL_END===",
            instructions_text,
            re.DOTALL,
        )
        if mid_level_match:
            content = mid_level_match.group(1).strip()
            # Split by lines and clean each
            for line in content.split("\n"):
                cleaned = line.strip()
                if cleaned and not cleaned.startswith("#"):
                    instructions["mid_level"].append(cleaned)
        else:
            # Fallback parsing
            mid_level_section = re.search(
                r"(?:### )?2\.\s*MID-LEVEL.*?\n+(.+?)(?=\n### 3|---|\n\n[A-Z])",
                instructions_text,
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            if mid_level_section:
                content = mid_level_section.group(1).strip()
                # Extract phase descriptions
                phase_pattern = r"\*?\s*\*?\*?Phase\s*\d+[:\s]([^*\n]+(?:\n[^*\n]+)*?)(?=\*?\s*\*?\*?Phase|\n\n|$)"
                phases = re.findall(
                    phase_pattern, content, re.MULTILINE | re.IGNORECASE
                )

                for phase in phases:
                    # Clean up the phase description
                    phase_text = phase.strip()
                    # Remove timing info if it's at the beginning
                    phase_text = re.sub(r"^\([^)]+\)\s*", "", phase_text)
                    # Remove any ** markers
                    phase_text = re.sub(r"\*\*", "", phase_text)
                    if phase_text:
                        instructions["mid_level"].append(phase_text)

        # Parse LOW-LEVEL instructions
        low_level_match = re.search(
            r"===LOW_LEVEL_START===\s*\n(.*?)\n\s*===LOW_LEVEL_END===",
            instructions_text,
            re.DOTALL,
        )
        if low_level_match:
            content = low_level_match.group(1).strip()
            # Parse pipe-separated format
            for line in content.split("\n"):
                line = line.strip()
                if line and "|" in line:
                    parts = [p.strip() for p in line.split("|")]
                    if len(parts) >= 3:
                        instructions["low_level"].append(
                            {
                                "action": parts[0],
                                "detail": parts[1],
                                "timing": parts[2],
                            }
                        )
        else:
            # Fallback parsing
            low_level_section = re.search(
                r"(?:### )?3\.\s*LOW-LEVEL.*?\n+(.+?)(?=$)",
                instructions_text,
                re.IGNORECASE | re.MULTILINE | re.DOTALL,
            )
            if low_level_section:
                content = low_level_section.group(1).strip()

                # First try to extract steps with the structured format
                step_pattern = r"\*?\s*\*?\*?Step\s*\d+[:\s].*?\n\s*Action:\s*(.+?)\n\s*Detail:\s*(.+?)\n\s*Timing:\s*(.+?)(?=\n\s*\*?\s*\*?\*?Step|\n\n|$)"
                steps = re.findall(
                    step_pattern,
                    content,
                    re.MULTILINE | re.IGNORECASE | re.DOTALL,
                )

                if steps:
                    for action, detail, timing in steps:
                        instructions["low_level"].append(
                            {
                                "action": action.strip(),
                                "detail": detail.strip(),
                                "timing": timing.strip(),
                            }
                        )
                else:
                    # Try alternative format with just step descriptions
                    step_pattern = r"\*?\s*\*?\*?Step\s*\d+[:\s]([^*\n]+(?:\n[^*\n]+)*?)(?=\*?\s*\*?\*?Step|\n\n|$)"
                    steps = re.findall(
                        step_pattern, content, re.MULTILINE | re.IGNORECASE
                    )
                    for step in steps:
                        step_text = step.strip()
                        # Remove timing info if present
                        step_text = re.sub(r"^\([^)]+\)\s*", "", step_text)
                        if step_text:
                            instructions["low_level"].append(
                                {
                                    "action": step_text,
                                    "detail": "",
                                    "timing": "",
                                }
                            )

        # Final cleanup for all instructions
        if instructions["high_level"]:
            instructions["high_level"] = (
                instructions["high_level"].strip().rstrip(".")
            )

        # Remove duplicates and clean up lists
        instructions["mid_level"] = [
            item for item in instructions["mid_level"] if item
        ][
            :5
        ]  # Max 5 phases
        instructions["low_level"] = [
            item for item in instructions["low_level"] if item
        ][
            :15
        ]  # Max 15 steps

        # Display the instructions
        self._console.print(
            Panel(
                f"[bold cyan]HIGH-LEVEL:[/bold cyan]\n{instructions['high_level']}",
                title="Global Task Instruction",
                border_style="cyan",
            )
        )

        mid_level_text = "\n".join(
            f"• {inst}" for inst in instructions["mid_level"]
        )
        self._console.print(
            Panel(
                f"[bold yellow]MID-LEVEL PHASES:[/bold yellow]\n{mid_level_text}",
                title="Subtask Instructions",
                border_style="yellow",
            )
        )

        low_level_text = "\n".join(
            f"• {inst}" for inst in instructions["low_level"][:10]
        )  # Show first 10
        if len(instructions["low_level"]) > 10:
            low_level_text += (
                f"\n... and {len(instructions['low_level']) - 10} more steps"
            )

        self._console.print(
            Panel(
                f"[bold green]LOW-LEVEL ACTIONS:[/bold green]\n{low_level_text}",
                title="Detailed Action Instructions",
                border_style="green",
            )
        )

        return instructions


def main():
    """Main entry point for the script."""
    init_logging()
    fire.Fire(CaptionSynthesizer)


if __name__ == "__main__":
    main()
