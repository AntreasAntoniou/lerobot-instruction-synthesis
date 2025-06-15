# Technical Specification: Freestyle Data Collection Pipeline

## System Architecture

### Overview

The Freestyle Data Collection Pipeline consists of four main components:

1. **Data Capture Module**: Records freestyle robot interactions
2. **Video Processing Engine**: Prepares video for AI analysis
3. **AI Annotation System**: Segments and annotates robot actions
4. **Dataset Generator**: Structures data into LeRobot format

### Component Specifications

#### 1. Data Capture Module

**Requirements:**
- Multi-camera support (minimum 2 views: side + wrist)
- Synchronized recording at 30+ FPS
- Real-time motor state logging
- Metadata capture (timestamp, operator ID, session info)

**Technical Stack:**
- Camera SDK: OpenCV / RealSense SDK
- Motor logging: ROS2 / Native robot API
- Storage: HDF5 for efficient video/sensor fusion

**Data Format:**
```python
{
    "session_id": "uuid",
    "timestamp": "ISO-8601",
    "operator": "string",
    "duration": "seconds",
    "videos": {
        "side_camera": "path/to/video",
        "wrist_camera": "path/to/video"
    },
    "motor_states": "path/to/motor_log.h5",
    "metadata": {
        "robot_type": "string",
        "environment": "string",
        "notes": "optional_string"
    }
}
```

#### 2. Video Processing Engine

**Core Functions:**
- Frame extraction and preprocessing
- Temporal smoothing for stable segmentation
- Multi-view synchronization
- Compression for efficient AI processing

**Processing Pipeline:**
```python
def process_video(raw_video_path):
    # 1. Load and validate video
    video = load_video(raw_video_path)
    validate_quality(video)
    
    # 2. Extract key frames for segmentation
    key_frames = extract_key_frames(video, interval=0.5)  # Every 0.5s
    
    # 3. Prepare for AI analysis
    processed_frames = preprocess_frames(key_frames)
    
    # 4. Create temporal windows
    windows = create_temporal_windows(video, window_size=3.0)  # 3s windows
    
    return processed_frames, windows
```

#### 3. AI Annotation System

**Gemini Integration:**
```python
class GeminiAnnotator:
    def __init__(self, model="gemini-2.5-pro-preview-03-25"):
        self.client = genai.Client(vertexai=True)
        self.model = model
        
    def segment_video(self, video_path, motor_data):
        """Identify distinct action segments in freestyle video"""
        prompt = self._build_segmentation_prompt(video_path, motor_data)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._get_config()
        )
        return self._parse_segments(response)
    
    def annotate_segment(self, segment):
        """Generate detailed annotations for a video segment"""
        prompt = self._build_annotation_prompt(segment)
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=self._get_config()
        )
        return self._parse_annotations(response)
```

**Segmentation Strategy:**
1. **Motion-based boundaries**: Detect significant changes in motor states
2. **Visual cues**: Identify object interactions and scene changes
3. **Temporal coherence**: Ensure segments are meaningful units (2-10 seconds)

**Annotation Schema:**
```json
{
    "segment_id": "uuid",
    "start_time": 0.0,
    "end_time": 5.3,
    "task_description": "Pick up red cube and place in container",
    "subtasks": [
        {
            "description": "Approach red cube",
            "start": 0.0,
            "end": 1.8
        },
        {
            "description": "Grasp cube",
            "start": 1.8,
            "end": 2.5
        },
        {
            "description": "Transport to container",
            "start": 2.5,
            "end": 4.2
        },
        {
            "description": "Release cube",
            "start": 4.2,
            "end": 5.3
        }
    ],
    "objects_involved": ["red_cube", "blue_container"],
    "skills_demonstrated": ["grasping", "transport", "precise_placement"],
    "quality_score": 0.92,
    "notes": "Smooth execution with minor hesitation during grasp"
}
```

#### 4. Dataset Generator

**LeRobot Integration:**
```python
class LeRobotDatasetBuilder:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        
    def build_dataset(self, annotated_segments):
        """Convert annotated segments to LeRobot format"""
        dataset = LeRobotDataset.create_empty(self.output_dir)
        
        for segment in annotated_segments:
            episode = self._segment_to_episode(segment)
            dataset.add_episode(episode)
            
        dataset.save()
        return dataset
    
    def _segment_to_episode(self, segment):
        """Convert a single segment to LeRobot episode format"""
        return {
            "task": segment["task_description"],
            "episode_index": segment["segment_id"],
            "frame_start": segment["start_frame"],
            "frame_end": segment["end_frame"],
            "fps": 30,
            "video_paths": segment["video_paths"],
            "actions": segment["motor_states"],
            "metadata": {
                "quality_score": segment["quality_score"],
                "subtasks": segment["subtasks"],
                "auto_generated": True
            }
        }
```

### Quality Assurance

**Validation Pipeline:**
1. **Segment Coherence**: Ensure each segment represents a complete action
2. **Annotation Accuracy**: Cross-validate with motor data
3. **Temporal Consistency**: Check for gaps or overlaps
4. **Human Verification**: Sample 5% for manual review

**Quality Metrics:**
```python
def calculate_quality_metrics(dataset):
    metrics = {
        "segment_completeness": check_action_completion(dataset),
        "annotation_confidence": average_confidence_scores(dataset),
        "temporal_coverage": calculate_coverage_ratio(dataset),
        "diversity_index": measure_task_diversity(dataset),
        "smoothness_score": analyze_trajectory_smoothness(dataset)
    }
    return metrics
```

### Performance Optimization

**Parallel Processing:**
- Concurrent video processing across multiple segments
- Batch API calls to Gemini for efficiency
- Distributed processing for large-scale datasets

**Caching Strategy:**
- Cache Gemini responses for similar actions
- Store preprocessed video frames
- Reuse segmentation boundaries for similar sessions

### Error Handling

**Robust Recovery:**
```python
class RobustProcessor:
    def process_with_retry(self, video_path, max_retries=3):
        for attempt in range(max_retries):
            try:
                return self.process_video(video_path)
            except GeminiAPIError as e:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    return self.fallback_processing(video_path)
            except VideoCorruptionError:
                return self.attempt_recovery(video_path)
```

### Scalability Considerations

**Horizontal Scaling:**
- Queue-based architecture for video processing
- Stateless workers for easy scaling
- Cloud storage for unlimited capacity

**Resource Management:**
- Adaptive quality based on available compute
- Progressive processing (low-res first, then high-res)
- Smart batching for API efficiency

### Integration Points

**APIs and Interfaces:**
```python
# REST API for web interface
@app.route('/api/process_session', methods=['POST'])
def process_freestyle_session():
    session_data = request.json
    job_id = queue_processing_job(session_data)
    return {"job_id": job_id, "status": "queued"}

# CLI for batch processing
python -m freestyle_pipeline process --input-dir ./recordings --output-dir ./datasets

# Python SDK for programmatic access
from freestyle_pipeline import FreestyleProcessor
processor = FreestyleProcessor()
dataset = processor.process_session("path/to/recording")
```

### Security and Privacy

**Data Protection:**
- Encrypted storage for video data
- Anonymized operator information
- Secure API keys management
- GDPR-compliant data handling

### Monitoring and Logging

**System Observability:**
```python
# Metrics to track
METRICS = {
    "processing_time_per_minute": histogram,
    "segmentation_accuracy": gauge,
    "api_calls_count": counter,
    "error_rate": rate,
    "queue_depth": gauge
}

# Structured logging
logger.info("Segment processed", extra={
    "segment_id": segment_id,
    "duration": duration,
    "quality_score": score,
    "processing_time": elapsed
})
```

### Future Enhancements

1. **Multi-modal fusion**: Incorporate force/torque sensors
2. **Real-time processing**: Stream processing during recording
3. **Active learning**: Identify segments needing human review
4. **Cross-robot transfer**: Adapt annotations across robot types
5. **Skill taxonomy**: Automatic skill categorization and indexing 