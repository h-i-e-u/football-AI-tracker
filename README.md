# Football Tracking Project

This project is a football video analytics application that uses YOLO for player and ball detection, tracking, team classification based on jersey color, possession estimation, and fall detection.

![gif_result](./videos/result.gif)

## Overview

The project includes:
- `app.py`: Streamlit app for uploading videos, running live stream mode, or processing videos and downloading annotated output.
- `possession.py`: Tracks ball possession between two teams based on the distance from the ball to players.
- `team_classifier.py`: Groups players into two teams using jersey color clustering with k-means.
- `fall_detector.py`: Detects falls based on the bounding box width-to-height ratio over multiple frames.
- `detect.py`: Simple YOLO script for detection on images, video, or webcam.
- `custom_model/`: Stores custom YOLO models and trained weights.
- `screenshot_extractor/`: Utility for extracting frames from videos.

## Requirements

This project uses Python and the following main libraries:
- `streamlit`
- `opencv-python`
- `ultralytics`
- `supervision`
- `numpy`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Streamlit App

1. Prepare a YOLO model and set the model path in the sidebar. The default path is `custom_model/train/weights/best.pt`.
2. Run the Streamlit app:
```bash
streamlit run app.py
```
3. Open the app in your browser.
4. Upload a video file (`.mp4`, `.avi`, `.mov`) and choose a mode:
   - `Mode 1: Stream` to view a real-time processed video stream.
   - `Mode 2: Process and Download` to process the full video and download the annotated result.

## Main Features

### Mode 1: Stream
- Detects and tracks players and the ball in the video.
- Draws bounding boxes and labels on detected objects.
- Estimates ball possession percentage for both teams.
- Detects potential fall events.

### Mode 2: Process and Download
- Processes the entire uploaded video.
- Saves the annotated video output.
- Provides a download button for the result video.

## Utility Scripts

### `detect.py`
This script uses YOLO to detect objects in an image, video, or webcam stream.

Example usage:
```bash
python detect.py --model custom_model/train/weights/best.pt --source vid.mp4
```

### `screenshot_extractor/video_to_frames.py`
A helper utility to extract frames from videos using settings stored in `screenshot_extractor/config.json`.

## Notes

- If the app cannot find the model, check the `Model path` setting in the sidebar.
- For better results, use a YOLO model trained specifically on football data.
- Processing large videos in `Mode 2` may take time because every frame is analyzed.

## Project Structure

```
football-pj/
├── app.py
├── detect.py
├── fall_detector.py
├── possession.py
├── requirements.txt
├── team_classifier.py
├── custom_model/
│   ├── custom_model.pt
│   ├── hfbest.pt
│   ├── ytcustom_model.pt
│   └── train/weights/best.pt
├── screenshot_extractor/
│   ├── config.json
│   ├── README.md
│   ├── requirements.txt
│   └── video_to_frames.py
└── output/
```
