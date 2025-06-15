# 🤖 LeRobot Instruction Synthesis

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## 🚀 Overview

LeRobot Instruction Synthesis is an advanced tool that leverages Large Language Models (LLMs) to automatically generate multi-level natural language instructions from robot trajectory data. This project enhances robotic datasets by transforming raw sensor data into rich, hierarchical task descriptions that can improve robot learning and human-robot interaction.

### 🎯 Key Features

- **Multi-Level Instruction Generation**: Creates instructions at three levels of granularity:
  - **High-level**: Overall task goals (e.g., "Pick up the cube and transfer it to the other arm")
  - **Mid-level**: Task phases and subtasks (e.g., "Approach the object", "Grasp", "Transfer")
  - **Low-level**: Detailed action sequences with timing information

- **Trajectory Analysis & Summarization**: Automatically analyzes robot movements and generates comprehensive natural language summaries

- **Negative Example Generation**: Uses LLM knowledge to generate "what not to do" examples for safer robot training

- **Multiple Interfaces**:
  - Command-line interface for batch processing
  - REST API server for integration with other systems
  - Web interface for interactive exploration

## 🏆 Hackathon Impact

This project addresses a critical challenge in robotics: the lack of rich, descriptive data for training more interpretable and safer robotic systems. By automatically generating natural language descriptions from trajectory data, we enable:

- Better human understanding of robot behaviors
- Improved safety through negative example generation
- Enhanced robot learning through multi-modal (vision + language) training
- Easier debugging and analysis of robot policies

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
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

## 📖 Usage

### Command Line Interface

#### Generate trajectory summary:
```bash
python -m lesynthesis.enrich_with_llm summarize \
    --dataset_repo_id="lerobot/aloha_sim_transfer_cube_human" \
    --episode_index=0
```

#### Generate negative examples:
```bash
python -m lesynthesis.enrich_with_llm generate_negatives \
    --dataset_repo_id="lerobot/aloha_sim_transfer_cube_human"
```

#### Generate multi-level instructions:
```bash
python -m lesynthesis.enrich_with_llm generate_instructions \
    --dataset_repo_id="lerobot/aloha_sim_transfer_cube_human" \
    --episode_index=0
```

### Web Server

Start the REST API server:
```bash
python -m lesynthesis.enrich_with_llm_server
```

The server will be available at `http://localhost:7777` with endpoints for:
- `/api/load_dataset` - Load a dataset
- `/api/summarize_trajectory/<episode>` - Get trajectory summary
- `/api/generate_negatives` - Generate negative examples
- `/api/generate_instructions/<episode>` - Get multi-level instructions

### Python API

```python
from lesynthesis.enrich_with_llm import LLMEnrichmentTool

# Initialize the tool
tool = LLMEnrichmentTool(model_name="gemini-2.0-flash-exp")

# Generate multi-level instructions
instructions = tool.generate_instructions(
    dataset_repo_id="lerobot/aloha_sim_transfer_cube_human",
    episode_index=0
)

print(f"High-level: {instructions['high_level']}")
print(f"Mid-level phases: {instructions['mid_level']}")
print(f"Low-level actions: {instructions['low_level']}")
```

## 🏗️ Architecture

The system consists of three main components:

1. **Trajectory Analyzer**: Processes robot sensor data to identify key phases and action patterns
2. **LLM Interface**: Communicates with Google's Gemini models to generate natural language
3. **Multi-Interface Layer**: Provides CLI, REST API, and web access

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Robot Dataset  │────▶│ Trajectory       │────▶│ LLM Interface   │
│  (HuggingFace)  │     │ Analyzer         │     │ (Gemini)        │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                           │
                                                           ▼
                        ┌─────────────────────────────────────────┐
                        │        Generated Instructions          │
                        │  • High-level task description        │
                        │  • Mid-level phase breakdown          │
                        │  • Low-level action sequences         │
                        └─────────────────────────────────────────┘
```

## 📊 Example Output

### High-Level Instruction:
> "Pick up the cube with the right arm and transfer it to the left arm"

### Mid-Level Phases:
> 1. Right arm approaches and positions for grasping the cube
> 2. Right arm grasps the cube
> 3. Right arm lifts the cube and maneuvers it to the transfer zone
> 4. Transfer the cube from the right arm to the left arm

### Low-Level Actions:
> - **0.00s**: Right Arm Initial Motion - Move from home position towards cube vicinity
> - **0.80s**: Right Arm Pre-Grasp Alignment - Refine end-effector position for precise alignment
> - **1.60s**: Right Arm Approach - Lower arm and position gripper around cube
> - **3.20s**: Right Arm Grasp - Close gripper to secure the cube
> - **4.00s**: Right Arm Lift and Transfer - Lift cube and move to transfer location
> - **4.80s**: Dual Arm Transfer - Coordinate both arms for cube handoff

## 🧪 Testing

Run the test suite:
```bash
pytest tests/ -v
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built on top of the [LeRobot](https://github.com/huggingface/lerobot) ecosystem
- Powered by Google's Gemini models
- Inspired by the need for richer robotic datasets in the research community

## 📞 Contact

- **Author**: Antreas Antoniou
- **Email**: [your-email@example.com]
- **GitHub**: [@AntreasAntoniou](https://github.com/AntreasAntoniou)

---

<p align="center">Made with ❤️ for the robotics community</p> 