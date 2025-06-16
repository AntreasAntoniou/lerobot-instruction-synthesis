#!/usr/bin/env python

"""
Web server for LeSynthesis - provides REST API and web interface for caption synthesis.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from lesynthesis.synthesizer import CaptionSynthesizer

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global state
current_dataset = None
current_dataset_id = None
synthesizer = CaptionSynthesizer()


@app.route("/")
def index():
    """Serve the HTML interface."""
    # Look for the HTML file in the project root
    html_path = Path(__file__).parent.parent / "enrich_with_llm_demo.html"
    return send_file(str(html_path))


@app.route("/api/load_dataset", methods=["POST"])
def load_dataset():
    """Load a dataset and return metadata."""
    global current_dataset, current_dataset_id

    try:
        data = request.json
        dataset_id = data.get("dataset_id")

        print(f"Loading dataset: {dataset_id}")

        current_dataset = LeRobotDataset(dataset_id)
        current_dataset_id = dataset_id

        # Get dataset info
        num_episodes = current_dataset.num_episodes
        tasks = list(set(current_dataset.meta.tasks.values()))

        print(f"Dataset loaded successfully: {num_episodes} episodes")

        return jsonify(
            {
                "success": True,
                "num_episodes": num_episodes,
                "tasks": tasks,
                "fps": current_dataset.fps,
                "robot_type": current_dataset.meta.robot_type or "Unknown",
                "episodes": list(range(num_episodes)),
            }
        )

    except Exception as e:
        print(f"Error loading dataset: {e}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/get_episode_video/<int:episode_index>")
def get_episode_video(episode_index):
    """Get the video path for an episode."""
    global current_dataset

    if current_dataset is None:
        return jsonify({"success": False, "error": "No dataset loaded"})

    try:
        if len(current_dataset.meta.video_keys) > 0:
            video_key = current_dataset.meta.video_keys[0]
            video_path = (
                current_dataset.root
                / current_dataset.meta.get_video_file_path(
                    ep_index=episode_index, vid_key=video_key
                )
            )

            if video_path.exists():
                return send_file(str(video_path), mimetype="video/mp4")

        return jsonify({"success": False, "error": "No video available"})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/get_motor_plot/<int:episode_index>")
def get_motor_plot(episode_index):
    """Generate and return motor activation plot."""
    global current_dataset

    if current_dataset is None:
        return jsonify({"success": False, "error": "No dataset loaded"})

    try:
        # Get episode data
        from_idx = current_dataset.episode_data_index["from"][
            episode_index
        ].item()
        to_idx = current_dataset.episode_data_index["to"][episode_index].item()
        episode_data = current_dataset.hf_dataset.select(
            range(from_idx, to_idx)
        )

        # Extract action data
        actions = np.stack([a.numpy() for a in episode_data["action"]])
        timestamps = np.array(episode_data["timestamp"])

        # Create figure
        n_dims = actions.shape[1]
        fig, axes = plt.subplots(
            min(n_dims, 7), 1, figsize=(12, min(n_dims * 1.5, 10)), sharex=True
        )

        if n_dims == 1:
            axes = [axes]
        elif n_dims > 7:
            axes = axes[:7]
            actions = actions[:, :7]

        # Plot each dimension
        for i, ax in enumerate(axes):
            ax.plot(timestamps, actions[:, i], linewidth=2)
            ax.set_ylabel(f"Motor {i+1}", fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(actions[:, i].min() - 0.1, actions[:, i].max() + 0.1)

        axes[-1].set_xlabel("Time (seconds)", fontsize=12)
        fig.suptitle(
            f"Motor Activations - Episode {episode_index}", fontsize=14
        )
        plt.tight_layout()

        # Save to temporary file
        temp_path = Path(f"/tmp/motor_plot_{episode_index}.png")
        fig.savefig(temp_path, dpi=100, bbox_inches="tight")
        plt.close(fig)

        return send_file(str(temp_path), mimetype="image/png")

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/summarize_trajectory/<int:episode_index>")
def summarize_trajectory(episode_index):
    """Generate trajectory summary."""
    global current_dataset_id, synthesizer

    if current_dataset_id is None:
        return jsonify({"success": False, "error": "No dataset loaded"})

    try:
        # Capture the output
        captured_text = []

        def capture_print(*args, **kwargs):
            if args and hasattr(args[0], "renderable"):
                captured_text.append(str(args[0].renderable))
            else:
                captured_text.append(str(args[0]) if args else "")

        original_print = synthesizer._console.print
        synthesizer._console.print = capture_print

        # Generate summary
        synthesizer.summarize(
            dataset_repo_id=current_dataset_id, episode_index=episode_index
        )

        # Restore original print
        synthesizer._console.print = original_print

        # Extract summary
        summary = ""
        for text in captured_text:
            if "[green]" in text:
                summary = text.replace("[green]", "").replace("[/green]", "")
                break

        return jsonify({"success": True, "summary": summary})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/generate_negatives")
def generate_negatives():
    """Generate negative examples."""
    global current_dataset_id, synthesizer

    if current_dataset_id is None:
        return jsonify({"success": False, "error": "No dataset loaded"})

    try:
        negatives = synthesizer.generate_negatives(
            dataset_repo_id=current_dataset_id
        )

        return jsonify({"success": True, "negatives": negatives})

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


@app.route("/api/generate_instructions/<int:episode_index>")
def generate_instructions(episode_index):
    """Generate multi-level instructions for an episode."""
    global current_dataset_id, synthesizer

    if current_dataset_id is None:
        return jsonify({"success": False, "error": "No dataset loaded"})

    try:
        # Generate instructions
        instructions = synthesizer.generate_instructions(
            dataset_repo_id=current_dataset_id, episode_index=episode_index
        )

        return jsonify(
            {
                "success": True,
                "instructions": {
                    "high_level": instructions["high_level"],
                    "mid_level": instructions["mid_level"],
                    "low_level": instructions["low_level"],
                },
            }
        )

    except Exception as e:
        return jsonify({"success": False, "error": str(e)})


def main(port: int = 7777, host: str = "0.0.0.0", debug: bool = False):
    """
    Start the LeRobot Instruction Synthesis Server.

    Args:
        port: Port to run the server on (default: 7777)
        host: Host to bind to (default: 0.0.0.0)
        debug: Run in debug mode (default: False)
    """
    print(f"Starting LeRobot Instruction Synthesis Server...")
    print(f"Open http://localhost:{port} in your browser")
    app.run(debug=debug, host=host, port=port)


def fire_main():
    """Entry point for Fire CLI."""
    import fire

    fire.Fire(main)


if __name__ == "__main__":
    fire_main()
