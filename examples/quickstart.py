#!/usr/bin/env python
"""
Quick start example for LeRobot Instruction Synthesis.

This script demonstrates how to use the tool to generate multi-level
instructions from robot trajectory data.
"""

import os
from lesynthesis.synthesizer import CaptionSynthesizer


def main():
    # Make sure you have set your Google API key
    if not os.environ.get("GOOGLE_API_KEY"):
        print("Please set your GOOGLE_API_KEY environment variable")
        print("export GOOGLE_API_KEY='your-key-here'")
        return

    # Initialize the tool
    print("Initializing LeRobot Instruction Synthesis...")
    tool = CaptionSynthesizer(model_name="gemini-2.0-flash-exp")

    # Example dataset from HuggingFace Hub
    dataset_id = "lerobot/aloha_sim_transfer_cube_human"
    episode_index = 0

    print(f"\nAnalyzing dataset: {dataset_id}")
    print(f"Episode: {episode_index}")

    # Generate multi-level instructions
    print("\nGenerating instructions...")
    instructions = tool.generate_instructions(
        dataset_repo_id=dataset_id, episode_index=episode_index
    )

    # Display results
    print("\n" + "=" * 60)
    print("GENERATED INSTRUCTIONS")
    print("=" * 60)

    print(f"\n🎯 HIGH-LEVEL GOAL:")
    print(f"   {instructions['high_level']}")

    print(f"\n📋 MID-LEVEL PHASES:")
    for i, phase in enumerate(instructions["mid_level"], 1):
        print(f"   {i}. {phase}")

    print(f"\n🔧 LOW-LEVEL ACTIONS (first 5):")
    for action in instructions["low_level"][:5]:
        if isinstance(action, dict):
            print(
                f"   • [{action.get('timing', 'N/A')}] {action.get('action', 'N/A')}"
            )
            if action.get("detail"):
                print(f"     {action['detail']}")
        else:
            print(f"   • {action}")

    # Generate trajectory summary
    print("\n" + "=" * 60)
    print("TRAJECTORY SUMMARY")
    print("=" * 60)
    print("\nGenerating trajectory summary...")
    # Note: This will print to console directly
    tool.summarize(dataset_repo_id=dataset_id, episode_index=episode_index)

    # Generate negative examples
    print("\n" + "=" * 60)
    print("NEGATIVE EXAMPLES (What NOT to do)")
    print("=" * 60)
    print("\nGenerating negative examples...")
    negatives = tool.generate_negatives(dataset_repo_id=dataset_id)

    for task, examples in negatives.items():
        print(f"\nTask: {task}")
        print(f"Pitfalls:\n{examples}")


if __name__ == "__main__":
    main()
