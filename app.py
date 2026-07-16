import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os
import supervision as sv
from possession import PossessionTracker
from team_classifier import TeamTracker, jersey_color

# Configure the Streamlit page
st.set_page_config(page_title="Football Tracking App", layout="wide", page_icon="⚽")
st.title("⚽ Football YOLO26 and Track")
st.sidebar.header("model configuration")

# 1.  Model path
model_path = st.sidebar.text_input("Path file Best.pt:", "custom_model/train/weights/best.pt")

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
        ("Mode 1: Xem trực tiếp Stream", "Mode 2: Tải video kết quả về máy")
    )
    
    # Thanh điều chỉnh tham số nâng cao
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎛️ Tham số thuật toán")
    img_size = st.sidebar.slider("Kích thước ảnh (imgsz):", 512, 1024, 800, step=32)
    conf_threshold = st.sidebar.slider("Ngưỡng tin cậy (Confidence):", 0.0, 1.0, 0.15, step=0.05)

    # --- VÙNG TẢI VIDEO ---
    uploaded_file = st.file_uploader("Tải video trận đấu của bạn lên (.mp4):", type=["mp4", "avi", "mov"])

    if uploaded_file is not None:
        # Tạo file tạm để lưu trữ video tải lên
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # Kiểm tra file cấu hình ByteTrack tùy chỉnh (đã tạo ở bước trước)
        tracker_config = "fasttrack.yaml"

        # --- XỬ LÝ CHẾ ĐỘ 1: WATCH LIVE STREAM ---
        if "Mode 1" in run_mode:
            if st.button("🚀 Bắt đầu Stream trực tiếp"):
                st.info("📺 Đang hiển thị luồng xử lý Real-time...")
                cap = cv2.VideoCapture(tfile.name)
                
                # Tạo khung trống để liên tục "bắn" ảnh vào
                frame_window = st.image([]) 
                
           

                # 1. Khởi tạo các annotator ngoài vòng lặp (để tối ưu hiệu năng)
                box_annotator = sv.EllipseAnnotator()
                label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.3, text_position=sv.Position.BOTTOM_CENTER)
                trace_annotator = sv.TraceAnnotator()  # Vẽ đường vết di chuyển của vật thể
               
                # ---------- UI ----------
                possession_placeholder = st.empty()

                col1, col2, col3 = st.columns([2, 6, 2])

                teamA_metric = col1.empty()
                bar_placeholder = col2.empty()
                teamB_metric = col3.empty()

                # ---------- Tracker ----------
                team_tracker = TeamTracker()
                possession_tracker = PossessionTracker()
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Chạy Track từ Ultralytics YOLO
                    results = model.track(frame, persist=True, imgsz=img_size, conf=conf_threshold, tracker=tracker_config, verbose=False)
                    
                    # Chuyển đổi kết quả sang Supervision Detections
                    detections = sv.Detections.from_ultralytics(results[0])

                    player_boxes = []
                    player_colors = []
                    ball_box = None

                    for xyxy, class_id in zip(detections.xyxy, detections.class_id):
                        class_name = model.model.names[class_id]
                        x1, y1, x2, y2 = map(int, xyxy)
                        if class_name == "player":
                            player_boxes.append((x1, y1, x2, y2))
                            player_colors.append(
                                jersey_color(frame, x1, y1, x2, y2)
                            )
                        elif class_name == "football":
                            ball_box = (x1, y1, x2, y2)
                    
                    team_ids, team_colors = team_tracker.assign(player_colors)
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
                        possession_placeholder.success("⚽ Team A control balll")

                    elif owner == 1:
                        possession_placeholder.success("⚽ Team B control ball")

                    else:
                        possession_placeholder.info("⚽ unknow")

                    # Tạo bản sao của khung hình để vẽ
                    annotated_frame = frame.copy()
                    
                    # KIỂM TRA VÀ XỬ LÝ TRACKING ID
                    if detections.tracker_id is not None:
                        # Lọc: Chỉ giữ lại các detection đã có ID hợp lệ để vẽ Trace và nhãn kèm ID
                        # Tránh lỗi thiếu tracker_id ở các frame đầu hoặc khi vật thể mất dấu
                        tracked_detections = detections[detections.tracker_id != None]
                        
                        # 1. Vẽ đường vết (Chỉ truyền tracked_detections đã được lọc)
                        if len(tracked_detections) > 0:
                            annotated_frame = trace_annotator.annotate(scene=annotated_frame, detections=tracked_detections)
                        
                        # 2. Tạo danh sách nhãn kèm ID cho các vật thể đang được track
                        labels = []
                        for class_id, tracker_id in zip(detections.class_id, detections.tracker_id):
                            class_name = model.model.names[class_id]
                            labels.append(f"#{tracker_id} {class_name}")
                    else:
                        # Nếu chưa có đối tượng nào được gán ID, nhãn chỉ hiển thị tên lớp
                        labels = [model.model.names[class_id] for class_id in detections.class_id]

                    # 3. Vẽ Bounding Box (áp dụng cho toàn bộ detections)
                    annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)
                    
                    # 4. Vẽ Nhãn (Labels)
                    annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)
                    
                    # Chuyển hệ màu để hiển thị chuẩn trên giao diện Web (Streamlit)
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(annotated_frame_rgb, channels="RGB")

                cap.release()
                st.success("🎉 Đã phát hết video!")

        # --- XỬ LÝ CHẾ ĐỘ 2: DOWNLOAD FILE ---
        elif "Mode 2" in run_mode:
            if st.button("⚙️ Bắt đầu đóng gói video kết quả"):
                st.info("⏳ Đang xử lý ngầm và lưu file... Vui lòng không tắt trình duyệt.")
                
                cap = cv2.VideoCapture(tfile.name)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                
                # Định dạng file xuất ra
                output_path = "football_tracked_download.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                
                # Tạo thanh tiến trình để người dùng dễ theo dõi
                progress_bar = st.progress(0)
                frame_count = 0
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Xử lý ngầm, không đẩy ảnh lên giao diện để tăng tốc độ ghi file
                    results = model.track(frame, persist=True, imgsz=img_size, conf=conf_threshold, tracker=tracker_config, verbose=False)
                    annotated_frame = results[0].plot()
                    out.write(annotated_frame)
                    
                    frame_count += 1
                    progress_bar.progress(int((frame_count / total_frames) * 100))
                
                cap.release()
                out.release()
                
                st.success("🎉 Đóng gói thành công! Bạn có thể tải file bên dưới:")
                
                # Hiển thị nút bấm Tải về khi quá trình ghi file kết thúc
                with open(output_path, "rb") as file:
                    st.download_button(
                        label="📥 BẤM VÀO ĐÂY ĐỂ TẢI VIDEO KẾT QUẢ (.MP4)",
                        data=file,
                        file_name="football_tracked_result.mp4",
                        mime="video/mp4"
                    )
        
        else:
            st.warning("⚠️ Vui lòng chọn chế độ xử lý trước khi bắt đầu!")