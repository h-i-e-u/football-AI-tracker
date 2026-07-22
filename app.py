import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import supervision as sv
from possession import PossessionTracker
from team_classifier import TeamTracker, jersey_color
from fall_detector import FallDetector

# Configure the Streamlit page
st.set_page_config(page_title="Football Tracking App", layout="wide", page_icon="⚽")
st.title("⚽ Football Tracking App")
st.sidebar.header("model configuration")

# 1.  Model path
# model_path = st.sidebar.text_input("Path file Best.pt:", "custom_model/train/weights/best.pt")
model_path = "custom_model/train/weights/best.pt"
# initialize the model
@st.cache_resource
def load_model(path):
    if os.path.exists(path):
        return YOLO(path)
    return None

model = load_model(model_path)

if model is None:
    st.error(f"❌ Model not found: {model_path}. check the path!")
else:
    st.sidebar.success("⚡ Load model sucessfully!")
    
    # choose run mode
    run_mode = st.sidebar.radio(
        "🎯 Choose mode:",
        ("Mode 1: Stream", "Mode 2: Process and Download")
    )
    
    # Config bar
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Config")
    img_size = st.sidebar.slider("Image size (imgsz):", 512, 1024, 800, step=32)
    conf_threshold = st.sidebar.slider("Confidence:", 0.0, 1.0, 0.15, step=0.05)

    # --- Load VIDEO ---
    uploaded_file = st.file_uploader("Upload your match (.mp4):", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Create temp file
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # fast track 
        tracker_config = "fasttrack.yaml"

        # --- Mode 1: WATCH LIVE STREAM ---
        if "Mode 1" in run_mode:
            if st.button("🚀 Start Stream"):
                st.info("📺 Show processed video in Real-time...")
                cap = cv2.VideoCapture(tfile.name)
                
                # image frame for simulate real-time stream 
                frame_window = st.image([]) 
                
           

                # init annotator 
                box_annotator = sv.EllipseAnnotator(sv.Color.ROBOFLOW)
                referee_box_annotator = sv.EllipseAnnotator(color=sv.Color.from_hex("#F6FF00"), thickness=2)
                label_annotator_ref = sv.LabelAnnotator(text_thickness=5, text_scale=0.6, text_position=sv.Position.TOP_CENTER,  text_color=sv.Color.from_hex("#000000"), color=sv.Color.from_hex("#F6FF00"))
                fall_box_annotator = sv.BoxAnnotator(color=sv.Color.from_hex("#FF8000"), thickness=3)
                label_annotator_fall = sv.LabelAnnotator(text_thickness=5, text_scale=0.6, text_position=sv.Position.TOP_CENTER,  text_color=sv.Color.from_hex("#FFFFFF"), color=sv.Color.from_hex("#FF8000"))
                label_annotator_team_a = sv.LabelAnnotator(text_thickness=5, text_scale=0.6, text_position=sv.Position.TOP_CENTER,  text_color=sv.Color.from_hex("#FFFFFF"), color=sv.Color.from_hex("#2196F3"))
                label_annotator_team_b = sv.LabelAnnotator(text_thickness=5, text_scale=0.6, text_position=sv.Position.TOP_CENTER,  text_color=sv.Color.from_hex("#FFFFFF"), color=sv.Color.from_hex("#F44336"))
               
                # ---------- UI ----------
                possession_placeholder = st.empty()

                col1, col2, col3 = st.columns([2, 6, 2])

                teamA_metric = col1.empty()
                bar_placeholder = col2.empty()
                teamB_metric = col3.empty()

                # ---------- Tracker ----------
                team_tracker = TeamTracker()
                possession_tracker = PossessionTracker()
                fall_detector = FallDetector(ratio_threshold=1.0, min_frames=5)

                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Run Track form Ultralytics YOLO
                    results = model.track(frame, persist=True, imgsz=img_size, conf=conf_threshold, tracker=tracker_config, verbose=False)
                    
                    # convert result to Supervision Detections
                    detections = sv.Detections.from_ultralytics(results[0])

                    player_boxes = []
                    player_colors = []
                    ball_box = None
                    fall_flags = [False] * len(detections.class_id)

                    player_indices = []
                    referee_indices = []
                    referee_labels = []

                    for idx, (xyxy, class_id) in enumerate(zip(detections.xyxy, detections.class_id)):
                        tracker_id = None
                        if detections.tracker_id is not None and idx < len(detections.tracker_id):
                            tracker_id = detections.tracker_id[idx]

                        class_name = model.model.names[class_id]
                        class_name_lower = class_name.lower()
                        x1, y1, x2, y2 = map(int, xyxy)

                        if class_name_lower in ["referee", "assistant referee", "assistant_referee", "ref"]:
                            referee_indices.append(idx)
                            referee_labels.append(f"REF #{tracker_id}" if tracker_id is not None else "REF")

                        elif class_name in ["player", "goalkeeper"]:
                            box = (x1, y1, x2, y2)

                            if tracker_id is not None:
                                if fall_detector.update(int(tracker_id), box):
                                    fall_flags[idx] = True

                            player_boxes.append((x1, y1, x2, y2))
                            player_colors.append(jersey_color(frame, x1, y1, x2, y2))
                            player_indices.append(idx)

                        elif class_name in ["football", "ball"]:
                            ball_box = (x1, y1, x2, y2)
                    
                    team_ids, team_colors = team_tracker.assign(player_colors)

                    team_id_by_detection = {}
                    for det_idx, team_id in zip(player_indices, team_ids):
                        team_id_by_detection[det_idx] = team_id

                    team_a_indices = [i for i, team_id in team_id_by_detection.items() if team_id == 0]
                    team_b_indices = [i for i, team_id in team_id_by_detection.items() if team_id == 1]

                    team_a_labels = []
                    for i in team_a_indices:
                        tracker_id = None
                        if detections.tracker_id is not None and i < len(detections.tracker_id):
                            tracker_id = detections.tracker_id[i]
                        team_a_labels.append(f"#{tracker_id} A" if tracker_id is not None else "A")

                    team_b_labels = []
                    for i in team_b_indices:
                        tracker_id = None
                        if detections.tracker_id is not None and i < len(detections.tracker_id):
                            tracker_id = detections.tracker_id[i]
                        team_b_labels.append(f"#{tracker_id} B" if tracker_id is not None else "B")

                    #
                    fall_indices = [i for i, flag in enumerate(fall_flags) if flag]
                    fall_labels = []
                    
                    for i in fall_indices:
                        tracker_id = None
                        if detections.tracker_id is not None and i < len(detections.tracker_id):
                            tracker_id = detections.tracker_id[i]
                        fall_labels.append(f"FALL #{tracker_id}" if tracker_id is not None else "FALL")
                    owner = possession_tracker.update(
                        ball_box,
                        player_boxes,
                        team_ids
                    )

                    p0, p1 = possession_tracker.percentages()
                    
                    teamA_metric.metric("🔵 Team A", f"{p0:.1f}%")
                    teamB_metric.metric("🔴 Team B", f"{p1:.1f}%")
                    bar_placeholder.markdown(f"""
                        <div style="display:flex;width:100%;height:30px;border-radius:8px;overflow:hidden">

                        <div style="
                        width:{p0}%;
                        background:#2196F3;
                        color:white;
                        text-align:center;
                        font-weight:bold;
                        line-height:30px;">
                        {p0:.1f}%
                        </div>

                        <div style="
                        width:{p1}%;
                        background:#F44336;
                        color:white;
                        text-align:center;
                        font-weight:bold;
                        line-height:30px;">
                        {p1:.1f}%
                        </div>

                        </div>
                        """,
                        unsafe_allow_html=True)
                    if owner == 0:
                        possession_placeholder.success("⚽ Team A control the ball")

                    elif owner == 1:
                        possession_placeholder.success("⚽ Team B control the ball")

                    else:
                        possession_placeholder.info("⚽ Unknow")

                    # Create a copy to draw
                    annotated_frame = frame.copy()
                    
                    # check and process TRACKING ID
                    if detections.tracker_id is not None:
                        # filter: only leave  valid ID detection to draw Trace and  ID label
                        tracked_detections = detections[detections.tracker_id != None]
                                                
                        #  Create label and ID for object currently track
                        labels = []
                        for i, class_id in enumerate(detections.class_id):
                            class_name = model.model.names[class_id]
                            if fall_flags[i]:
                                labels.append("FALL DETECTED")
                            else:
                                labels.append(f"#{tracker_id} {class_name[:3].upper()}")
                    else:
                        # if there no object have ID , label just display class
                        labels = [model.model.names[class_id] for class_id in detections.class_id]

                    # Draw Bounding Box (apply for all detections)
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)

                    # draw labels for each team separately
                    if team_a_indices:
                        annotated_frame = label_annotator_team_a.annotate(
                            scene=annotated_frame,
                            detections=detections[team_a_indices],
                            labels=team_a_labels
                        )

                    if team_b_indices:
                        annotated_frame = label_annotator_team_b.annotate(
                            scene=annotated_frame,
                            detections=detections[team_b_indices],
                            labels=team_b_labels
                        )
                    if fall_indices:
                        fall_detections = detections[fall_indices]
                        annotated_frame = fall_box_annotator.annotate(scene=annotated_frame, detections=fall_detections)
                        annotated_frame = label_annotator_fall.annotate(
                            scene=annotated_frame,
                            detections=fall_detections,
                            labels=fall_labels
                        )

                    if referee_indices:
                        referee_detections = detections[referee_indices]
                        annotated_frame = referee_box_annotator.annotate(scene=annotated_frame, detections=referee_detections)
                        annotated_frame = label_annotator_ref.annotate(
                            scene=annotated_frame,
                            detections=referee_detections,
                            labels=referee_labels
                        )

                    # convert color for  Web UI (Streamlit)
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(annotated_frame_rgb, channels="RGB")

                cap.release()
                st.success("🎉 video end!")

        # --- mode 2: DOWNLOAD FILE ---
        elif "Mode 2" in run_mode:
            if st.button("⚙️ Start process video"):
                st.info("⏳ Processing... Please don't turn of browser.")
                
                cap = cv2.VideoCapture(tfile.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # format output file
                output_path = "football_tracked_download.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # create progress bar
                progress_bar = st.progress(0)
                frame_count = 0
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # undergound processing
                    results = model.track(frame, persist=True, imgsz=img_size, conf=conf_threshold, tracker=tracker_config, verbose=False)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                    
                    frame_count += 1
                    progress_bar.progress(int((frame_count / total_frames) * 100))
                
                cap.release()
                out.release()
                
                st.success("🎉 Successful! You can download file below:")
                
                # Display download button
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="📥 CLICK HERE TO DOWNLOAD RESULT VIDEO  (.MP4)",
                        data=file,
                        file_name="football_tracked_result.mp4",
                        mime="video/mp4"
                    )
        
        else:
            st.warning("⚠️ ERR: Please choose mode before start!")