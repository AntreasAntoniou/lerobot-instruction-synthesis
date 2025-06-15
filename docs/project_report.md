# LeRobot LLM Enrichment Tool - Project Report

**Date:** June 15, 2025  
**Author:** Antreas Antoniou & AI Assistant  
**Project:** LeRobot Dataset Enrichment with Large Language Models

## Executive Summary

We developed an LLM enrichment tool for LeRobot datasets that leverages Google's Gemini models to automatically generate natural language descriptions of robot trajectories and create "negative examples" of task execution. This tool transforms data-poor robotic datasets into semantically rich resources that can improve robot learning and safety.

## Problem Statement

### Current Challenges
1. **Limited Semantic Information**: Most robotic datasets contain only raw sensor data (joint angles, positions, images) without natural language descriptions
2. **Lack of Failure Cases**: Datasets typically only contain successful demonstrations, missing valuable information about what NOT to do
3. **Difficulty in Understanding**: Raw trajectory data is hard for humans to interpret and analyze
4. **Safety Concerns**: Without explicit negative examples, robots may not learn to avoid dangerous or inefficient behaviors

### Why This Matters
- Natural language descriptions can enable better multi-modal learning
- Understanding failure modes is crucial for safe robot deployment
- Human-readable summaries facilitate dataset curation and debugging
- Rich semantic information can improve generalization and transfer learning

## Solution Overview

We built a Python tool that integrates with the LeRobot ecosystem to:

### 1. Trajectory Summarization
- Analyzes robot trajectory data from episodes
- Generates comprehensive natural language summaries
- Describes high-level actions, movement phases, and patterns
- Provides insights about task execution quality

### 2. Negative Example Generation
- Uses LLM's world knowledge to generate plausible failure scenarios
- Creates descriptions of unsafe, incorrect, or inefficient behaviors
- Helps identify potential failure modes before they occur
- Useful for training more robust and safety-aware policies

## Technical Implementation

### Architecture
```
┌─────────────────────┐
│   LeRobot Dataset   │
│  (HuggingFace Hub)  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  LLMEnrichmentTool  │
│  - Data Processing  │
│  - Prompt Creation  │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Google Gemini     │
│      LLM API        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│   Rich Console      │
│  Formatted Output   │
└─────────────────────┘
```

### Key Components

1. **Data Extraction**
   - Uses `episode_data_index` to locate episode boundaries
   - Extracts relevant features (states, actions, metadata)
   - Handles multi-modal data (joint states, images, etc.)

2. **Prompt Engineering**
   - Structured prompts for consistent, high-quality outputs
   - Task-specific context for better understanding
   - Clear formatting instructions for readability

3. **API Integration**
   - Modern Google Generative AI SDK implementation
   - Configurable safety settings
   - Support for multiple Gemini model variants

4. **Output Formatting**
   - Rich console output with color coding
   - Structured panels for better readability
   - Clear separation of different output types

### Code Structure
```
lerobot/
├── cli/
│   ├── enrich_with_llm.py      # Main implementation
│   └── README_llm_enrichment.md # Documentation
├── tests/
│   └── test_enrich_with_llm.py # Integration tests
└── examples/
    └── llm_enrichment_demo.py   # Demo script
```

## Key Features

### Trajectory Summarization
- **Input**: Dataset repository ID and episode index
- **Processing**: Extracts episode data, creates structured description
- **Output**: Natural language summary with:
  - High-level task description
  - Key phases and stages
  - Movement patterns
  - Execution insights

### Negative Example Generation
- **Input**: Dataset repository ID
- **Processing**: Extracts unique tasks, generates failure scenarios
- **Output**: List of plausible negative examples:
  - Grasping errors
  - Collision scenarios
  - Timing failures
  - Inefficient behaviors

## Usage Examples

### Command Line Interface
```bash
# Summarize a trajectory
python -m lerobot.cli.enrich_with_llm summarize \
    --dataset_repo_id="lerobot/aloha_sim_transfer_cube_human" \
    --episode_index=0

# Generate negative examples
python -m lerobot.cli.enrich_with_llm generate_negatives \
    --dataset_repo_id="lerobot/aloha_sim_transfer_cube_human"
```

### Python API
```python
from lerobot.cli.enrich_with_llm import LLMEnrichmentTool

tool = LLMEnrichmentTool()
tool.summarize(dataset_repo_id="...", episode_index=0)
negatives = tool.generate_negatives(dataset_repo_id="...")
```

## Results and Impact

### Example Outputs

#### Trajectory Summary (Excerpt)
> "The robot successfully executes a bimanual manipulation task over 8 seconds, involving the precise acquisition of a cube with its right arm, followed by a coordinated transfer of the cube to its left arm..."

#### Negative Examples (Excerpt)
> "1. Grasping Error: The right arm's end-effector approaches the cube but misjudges its exact 3D position, causing the gripper to close on empty space..."

### Benefits Achieved
1. **Enhanced Understanding**: Complex trajectories now have human-readable descriptions
2. **Safety Awareness**: Explicit documentation of potential failure modes
3. **Research Enablement**: Rich semantic data for multi-modal learning research
4. **Debugging Support**: Easier identification of trajectory anomalies

## Technical Challenges and Solutions

### Challenge 1: API Evolution
- **Problem**: Google's Generative AI SDK had breaking changes
- **Solution**: Updated to new API patterns with proper client initialization

### Challenge 2: Dataset API Complexity
- **Problem**: LeRobot datasets have complex episode access patterns
- **Solution**: Used `episode_data_index` for proper episode boundary detection

### Challenge 3: Output Quality
- **Problem**: Ensuring consistent, high-quality LLM outputs
- **Solution**: Structured prompts with clear instructions and examples

## Future Enhancements

### Short Term
1. **Batch Processing**: Process multiple episodes in parallel
2. **Export Formats**: Save summaries to JSON/CSV for analysis
3. **Custom Prompts**: Allow users to specify custom analysis prompts

### Medium Term
1. **Multi-Provider Support**: Add OpenAI, Anthropic, and local LLMs
2. **Visual Analysis**: Incorporate image data into summaries
3. **Anomaly Detection**: Flag unusual patterns in trajectories

### Long Term
1. **Dataset Augmentation**: Generate synthetic trajectories from descriptions
2. **Policy Improvement**: Use insights to refine robot policies
3. **Interactive Analysis**: Web interface for dataset exploration

## Conclusion

The LLM enrichment tool successfully bridges the gap between raw robotic data and human understanding. By leveraging state-of-the-art language models, we've created a system that not only makes datasets more interpretable but also enhances their value for training safer, more robust robotic systems.

This tool represents a step toward more transparent and explainable robotics, where the behavior of autonomous systems can be understood, analyzed, and improved through natural language interfaces.

## Appendix

### Dependencies
- `google-generativeai`: For LLM integration
- `rich`: For formatted console output
- `fire`: For CLI interface
- `python-dotenv`: For environment variable management

### Performance Metrics
- Average summarization time: ~5-10 seconds per episode
- API costs: Approximately $0.001-0.005 per summary
- Supported dataset size: Any LeRobot-compatible dataset

### Testing
- Integration tests with real datasets
- Mocked console output verification
- API key validation and error handling 