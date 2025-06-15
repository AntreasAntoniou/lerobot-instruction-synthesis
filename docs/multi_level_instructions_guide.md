# Multi-Level Instruction Generation for Robot Trajectories

## Overview

The LeRobot LLM Enrichment Tool now includes a powerful feature for generating multi-level instructional descriptions of robot trajectories. This feature analyzes recorded robot demonstrations and creates instructions at various levels of detail, helping to understand both local and global tasks being performed.

## Purpose

When training robots or analyzing their behavior, it's crucial to understand tasks at different levels of abstraction:

1. **Global Understanding**: What is the overall goal?
2. **Task Decomposition**: What are the major phases or subtasks?
3. **Detailed Execution**: What specific actions need to be taken?

This multi-level approach helps in:
- Creating better training data for hierarchical learning
- Debugging robot behaviors
- Generating natural language descriptions for human-robot interaction
- Understanding the structure of complex tasks

## Levels of Instructions

### 1. High-Level (Global Task)
- **Purpose**: Describes the overall goal in a single, clear sentence
- **Example**: "Transfer the cube from the right gripper to the left gripper"
- **Use Case**: Task classification, high-level planning

### 2. Mid-Level (Subtasks/Phases)
- **Purpose**: Breaks down the task into 3-5 major phases
- **Example**:
  - Phase 1: Position the right arm above the cube
  - Phase 2: Lower the gripper and grasp the cube
  - Phase 3: Lift and move toward the left gripper
  - Phase 4: Transfer the cube to the left gripper
  - Phase 5: Release and retract the right gripper
- **Use Case**: Hierarchical planning, skill segmentation

### 3. Low-Level (Detailed Actions)
- **Purpose**: Step-by-step instructions as if programming the robot
- **Example**:
  - Step 1: Move right arm to position (x, y, z) over 2.0 seconds
  - Step 2: Open right gripper to 80% aperture
  - Step 3: Lower right arm by 10cm over 1.5 seconds
- **Use Case**: Motion planning, trajectory optimization

## How It Works

1. **Trajectory Analysis**: The system analyzes the recorded robot trajectory data including:
   - Action sequences (motor commands)
   - Timing information (duration, FPS)
   - Statistical patterns (mean, variance, ranges)

2. **Phase Detection**: Automatically identifies significant changes in robot behavior to detect phase transitions

3. **LLM Generation**: Uses a large language model to interpret the data and generate natural language instructions at each level

## Usage

### Command Line

```bash
# Generate instructions for a specific episode
python -m lerobot.cli.enrich_with_llm generate_instructions \
    --dataset_repo_id="lerobot/aloha_sim_transfer_cube_human" \
    --episode_index=0
```

### Web Interface

1. Load a dataset in the web interface
2. Select an episode
3. Click "Generate Instructions" button
4. View the hierarchical instructions displayed with color coding:
   - 🎯 Blue: High-level global task
   - 📋 Orange: Mid-level phases
   - 🔧 Green: Low-level detailed actions

### Python API

```python
from lerobot.cli.enrich_with_llm import LLMEnrichmentTool

tool = LLMEnrichmentTool()
instructions = tool.generate_instructions(
    dataset_repo_id="lerobot/aloha_sim_transfer_cube_human",
    episode_index=0
)

print(f"High-level: {instructions['high_level']}")
print(f"Mid-level phases: {instructions['mid_level']}")
print(f"Low-level steps: {instructions['low_level']}")
```

## Example Output

For a cube transfer task, the system might generate:

**High-Level:**
> Transfer the red cube from the right robotic arm to the left robotic arm

**Mid-Level:**
1. Initialize and position both arms in ready stance
2. Move right arm to approach and grasp the cube
3. Lift cube and coordinate both arms for transfer position
4. Execute handoff between right and left grippers
5. Retract right arm while left arm secures the cube

**Low-Level:**
1. Move right arm to home position (0.3, 0.0, 0.4) over 1.5 seconds
2. Move left arm to receiving position (-0.3, 0.0, 0.4) over 1.5 seconds
3. Open right gripper to 100% aperture
4. Move right arm to pre-grasp position above cube (0.2, 0.1, 0.3) over 2.0 seconds
5. Lower right arm to cube position (0.2, 0.1, 0.15) over 1.0 seconds
6. Close right gripper to 40% aperture to grasp cube
7. Wait 0.5 seconds for stable grasp
8. Lift right arm with cube to transfer height (0.2, 0.1, 0.35) over 1.5 seconds
... (and more detailed steps)

## Benefits

1. **Training Data Enhancement**: Provides rich textual descriptions for multi-modal learning
2. **Interpretability**: Makes robot behaviors more understandable to humans
3. **Debugging**: Helps identify where tasks fail by comparing expected vs actual behavior
4. **Transfer Learning**: Instructions can guide learning on new but similar tasks
5. **Human-Robot Interaction**: Enables natural language communication about tasks

## Future Enhancements

- **Conditional Instructions**: Include decision points and conditional behaviors
- **Error Recovery**: Generate instructions for handling common failure modes
- **Optimization Hints**: Suggest more efficient ways to perform tasks
- **Multi-Robot Coordination**: Instructions for collaborative tasks
- **Real-time Adaptation**: Generate instructions that adapt to environmental changes

## Conclusion

Multi-level instruction generation bridges the gap between low-level robot control and high-level task understanding. This feature transforms raw trajectory data into structured, hierarchical knowledge that can improve both robot learning and human understanding of robotic behaviors. 