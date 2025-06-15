# LeRobot LLM Enrichment Tool

The LLM enrichment tool enhances LeRobot datasets by using Large Language Models (LLMs) to generate natural language descriptions and insights about robot trajectories.

## Features

### 1. Trajectory Summarization
Analyzes robot trajectory data and generates comprehensive natural language summaries that describe:
- What the robot is doing at a high level
- Key phases or stages of the task
- Notable patterns in the robot's movements
- Interesting observations about the execution

### 2. Negative Example Generation
Uses the LLM's world knowledge to generate textual descriptions of incorrect, unsafe, or inefficient ways to perform a given task. This helps in:
- Understanding common failure modes
- Training more robust policies
- Creating safety-aware systems

## Installation

The tool requires the Google Generative AI SDK:

```bash
pip install google-generativeai
```

## Setup

You need a Google API key to use this tool. You can set it up in one of three ways:

1. **Environment variable:**
   ```bash
   export GOOGLE_API_KEY="your-api-key-here"
   ```

2. **`.env` file:** Create a `.env` file in your project root:
   ```
   GOOGLE_API_KEY=your-api-key-here
   ```

3. **Command line argument:** Pass it directly when running the tool

## Usage

### Command Line Interface

#### Trajectory Summarization
```bash
python -m lerobot.cli.enrich_with_llm summarize \
    --dataset_repo_id="lerobot/aloha_sim_transfer_cube_human" \
    --episode_index=0
```

#### Negative Example Generation
```bash
python -m lerobot.cli.enrich_with_llm generate_negatives \
    --dataset_repo_id="lerobot/aloha_sim_transfer_cube_human"
```

### Python API

```python
from lerobot.cli.enrich_with_llm import LLMEnrichmentTool

# Initialize the tool
tool = LLMEnrichmentTool()

# Or use a different model
# tool = LLMEnrichmentTool(model_name="gemini-2.5-flash-preview-05-20")

# Summarize a trajectory
tool.summarize(
    dataset_repo_id="lerobot/aloha_sim_transfer_cube_human",
    episode_index=0
)

# Generate negative examples
negatives = tool.generate_negatives(
    dataset_repo_id="lerobot/aloha_sim_transfer_cube_human"
)
```

## Available Models

The default model is `gemini-2.5-flash-preview-05-20`, which provides a good balance of speed and quality. You can also use:
- `gemini-1.5-flash` - Faster, good for quick iterations
- `gemini-1.5-pro` - Higher quality, slower
- Other Gemini models as they become available

## Example Output

### Trajectory Summary
```
Task: Pick up the cube with the right arm and transfer it to the left arm.
Episode Length: 400 frames
Duration: 8.00 seconds

The robot successfully executes a bimanual manipulation task over 8 seconds, 
involving the precise acquisition of a cube with its right arm, followed by 
a coordinated transfer of the cube to its left arm...
```

### Negative Examples
```
1. Grasping Error: The right arm's end-effector approaches the cube but 
   misjudges its exact 3D position, causing the gripper to close on empty 
   space adjacent to the cube.

2. Insufficient Grasp Force: The right arm successfully closes its gripper 
   around the cube, but the applied force is insufficient, leading to the 
   cube slipping from the gripper...
```

## Testing

Run the tests with:
```bash
pytest tests/test_enrich_with_llm.py -v
```

Note: Tests require a valid GOOGLE_API_KEY to be set.

## Limitations

- Requires an internet connection and valid API key
- API usage is subject to rate limits and quotas
- Quality of summaries depends on the information available in the dataset
- Currently only supports Google's Gemini models

## Future Enhancements

- Support for other LLM providers (OpenAI, Anthropic, etc.)
- Batch processing for multiple episodes
- Integration with dataset creation pipeline
- Custom prompts for domain-specific insights
- Export summaries to various formats (JSON, Markdown, etc.) 