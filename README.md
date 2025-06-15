# 🤖 LeRobot Instruction Synthesis

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## 🎯 Rich Caption Generation for Robot Learning

Transform simple robot action datasets into richly annotated training data using state-of-the-art multi-modal LLMs. This project takes existing robot trajectories with basic single-line descriptions and generates detailed, multi-level instructions that enable training more sophisticated robot control models.

### 🌟 Key Innovation

Most lerobot datasets come with minimal descriptions like "pick up the cube" or "open the drawer". This severely limits the ability to train robots that can understand complex, narrative-style commands. **LeRobot Instruction Synthesis** solves this by:

- 📹 **Analyzing robot trajectory videos** using multi-modal LLMs (Gemini, Claude 3.5, GPT-4o)
- 📝 **Generating rich, hierarchical captions** at multiple levels of detail
- 🎭 **Creating narrative-style instructions** that describe not just what to do, but how and why
- 🔄 **Enhancing existing datasets** without requiring new robot demonstrations

### 📊 Example Transformation

**Before (Original Dataset):**
```
Task: "Pick up cube"
```

**After (Enhanced with our tool):**
```
High-Level: "Pick up the red cube from the table and transfer it to the blue container"

Mid-Level:
1. "Approach the red cube on the table surface"
2. "Position the gripper above the cube and align for optimal grasp"
3. "Close the gripper to secure the cube"
4. "Lift and transport the cube to the blue container"
5. "Lower the cube into the container and release"

Low-Level:
- "Move arm forward 15cm while maintaining 10cm height above table"
- "Rotate wrist 45° clockwise for perpendicular approach angle"
- "Lower gripper 8cm until 2cm above cube surface"
- [... detailed action sequences ...]
```

## 🚀 Features

- **Multi-Modal LLM Integration**: Leverages vision-language models to understand robot actions from video
- **Hierarchical Instruction Generation**: Creates high-level goals, mid-level subtasks, and low-level action sequences
- **Negative Example Generation**: Produces contrastive examples for more robust training
- **Trajectory Summarization**: Generates concise descriptions of entire robot episodes
- **Web Interface**: User-friendly Flask server and Gradio UI for dataset exploration
- **Batch Processing**: Efficiently process entire datasets with progress tracking

## 🛠️ Installation

### Prerequisites

- Python 3.11 or higher
- Google Cloud API key (for Gemini LLM access)
- (Optional) CUDA-capable GPU for faster inference with vLLM

### Quick Start

1. Clone the repository:
```bash
git clone https://github.com/AntreasAntoniou/lerobot-instruction-synthesis.git
cd lerobot-instruction-synthesis
```

2. Install the package:
```bash
pip install -e .
```

3. Set up your Google API key:
```bash
export GOOGLE_API_KEY="your-api-key-here"
# Or create a .env file with: GOOGLE_API_KEY=your-api-key-here
```

## 🎮 Usage

### Command Line Interface

Generate rich captions for your robot dataset:

```bash
# Basic usage - enhance a single episode
lesynthesis generate-instructions --dataset lerobot/pusht --episode 0

# Process entire dataset with rich captions
lesynthesis enrich-dataset --dataset lerobot/pusht --output enhanced_pusht

# Generate with specific detail level
lesynthesis generate-instructions --dataset lerobot/pusht --episode 0 --detail-level high
```

### Python API

```python
from lesynthesis import LLMEnrichmentTool

# Initialize the tool
enrichment_tool = LLMEnrichmentTool()

# Generate rich captions for a robot trajectory
instructions = enrichment_tool.generate_instructions(
    dataset_repo_id="lerobot/pusht",
    episode_index=0
)

# Access multi-level captions
print(f"Goal: {instructions['high_level']}")
print(f"Steps: {instructions['mid_level']}")
print(f"Actions: {instructions['low_level']}")
```

### Web Interface

Launch the interactive web server to explore and annotate datasets:

```bash
# Start the Flask server
lesynthesis-server --port 5001

# Or use the Gradio interface
lesynthesis gradio --dataset lerobot/pusht
```

## 🔧 How It Works

1. **Video Analysis**: Extracts frames from robot trajectory videos
2. **Multi-Modal Understanding**: Sends video data to LLMs with vision capabilities
3. **Structured Generation**: Uses carefully crafted prompts to generate hierarchical instructions
4. **Quality Assurance**: Validates generated captions for consistency and completeness
5. **Dataset Enhancement**: Saves enriched annotations alongside original data

## 🎯 Use Cases

- **Training Language-Conditioned Policies**: Use rich captions to train robots that understand complex commands
- **Dataset Augmentation**: Enhance existing datasets without collecting new demonstrations
- **Human-Robot Interaction**: Generate natural language explanations of robot behaviors
- **Curriculum Learning**: Use hierarchical instructions for progressive skill learning
- **Sim-to-Real Transfer**: Bridge the gap with detailed action descriptions

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of [🤗 LeRobot](https://github.com/huggingface/lerobot) for robot learning
- Powered by state-of-the-art multi-modal LLMs (Gemini, Claude, GPT-4)
- Inspired by the need for richer robot training data in the community

## 📚 Citation

If you use this tool in your research, please cite:

```bibtex
@software{lerobot_instruction_synthesis,
  title = {LeRobot Instruction Synthesis: Rich Caption Generation for Robot Learning},
  author = {Antoniou, Antreas},
  year = {2024},
  url = {https://github.com/AntreasAntoniou/lerobot-instruction-synthesis}
}
```

## 📞 Contact

- **Author**: Antreas Antoniou
- **Email**: [your-email@example.com]
- **GitHub**: [@AntreasAntoniou](https://github.com/AntreasAntoniou)

---

<p align="center">Made with ❤️ for the robotics community</p> 