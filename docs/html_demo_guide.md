# LeRobot LLM Enrichment Tool - HTML Demo Guide

## Overview

We've created a clean, standalone HTML interface for the LeRobot LLM Enrichment Tool that doesn't rely on Gradio. This provides a more stable and customizable web experience.

## Architecture

The demo consists of two parts:

1. **Flask Server** (`lerobot/cli/enrich_with_llm_server.py`)
   - Lightweight Python web server
   - RESTful API endpoints
   - Handles dataset loading, video serving, and LLM interactions

2. **HTML Interface** (`enrich_with_llm_demo.html`)
   - Pure HTML/CSS/JavaScript
   - No external dependencies
   - Clean, modern UI design

## Features

### 1. Dataset Loading
- Enter any LeRobot dataset repository ID
- Displays dataset metadata (episodes, tasks, FPS, robot type)
- Populates episode dropdown for selection

### 2. Episode Analysis
- **Video Playback**: Shows the robot trajectory video
- **Motor Activations**: Visualizes motor commands over time
- **AI Summary**: Generates natural language description of the trajectory

### 3. Negative Examples
- Generates examples of what NOT to do
- Helps identify potential failure modes
- Useful for safety-aware training

## How to Use

### 1. Start the Server

```bash
python lerobot/cli/enrich_with_llm_server.py
```

The server will start on `http://localhost:5000`

### 2. Open the Interface

Navigate to `http://localhost:5000` in your browser, or run:

```bash
python open_llm_enrichment_demo.py
```

### 3. Load a Dataset

1. Enter a dataset repository ID (e.g., `lerobot/aloha_sim_transfer_cube_human`)
2. Click "Load Dataset"
3. Wait for the dataset information to appear

### 4. Analyze Episodes

1. Select an episode from the dropdown
2. Click "Analyze Episode"
3. View:
   - The robot trajectory video
   - Motor activation plots showing control signals
   - AI-generated summary of what the robot is doing

### 5. Generate Negative Examples

Click "Generate Negative Examples" to create descriptions of incorrect behaviors for all tasks in the dataset.

## API Endpoints

The Flask server provides these endpoints:

- `GET /` - Serves the HTML interface
- `POST /api/load_dataset` - Loads a dataset
- `GET /api/get_episode_video/<episode_index>` - Returns episode video
- `GET /api/get_motor_plot/<episode_index>` - Returns motor activation plot
- `GET /api/summarize_trajectory/<episode_index>` - Generates trajectory summary
- `GET /api/generate_negatives` - Generates negative examples

## Technical Details

### Frontend
- Vanilla JavaScript (no frameworks)
- Responsive CSS Grid layout
- Async/await for API calls
- Tab-based interface

### Backend
- Flask web framework
- Matplotlib for plotting (Agg backend)
- Direct integration with LeRobot datasets
- LLM tool wrapper for clean API

## Advantages Over Gradio

1. **Stability**: No complex dependencies or version conflicts
2. **Customization**: Full control over UI/UX
3. **Performance**: Lightweight and fast
4. **Debugging**: Standard web development tools work
5. **Deployment**: Easy to deploy anywhere

## Troubleshooting

### Server Won't Start
- Check if port 5000 is already in use
- Ensure Flask is installed: `pip install flask flask-cors`

### No Video Displayed
- Some datasets may not have video data
- Check browser console for errors

### LLM Errors
- Ensure GOOGLE_API_KEY is set in environment
- Check API quota limits

### CORS Issues
- The server has CORS enabled by default
- If issues persist, check browser security settings

## Future Enhancements

1. **Multi-camera Support**: Show multiple camera views
2. **Trajectory Comparison**: Compare multiple episodes
3. **Export Features**: Save summaries and plots
4. **Real-time Analysis**: Stream analysis as video plays
5. **Custom Prompts**: Allow users to ask specific questions

## Conclusion

This HTML demo provides a clean, stable interface for the LeRobot LLM Enrichment Tool. It's designed to be easy to use, modify, and extend for your specific needs. 