import streamlit as st
import cv2
from ultralytics import YOLO
import tempfile
import os

# Configure the Streamlit page
st.set_page_config(page_title="Football Tracking App", layout="wide", page_icon="⚽")
st.title("⚽ Football YOLO26 and ByteTrack")
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
        tracker_config = "custom_bytetrack.yaml" if os.path.exists("custom_bytetrack.yaml") else "bytetrack.yaml"

        # --- XỬ LÝ CHẾ ĐỘ 1: WATCH LIVE STREAM ---
        if "Mode 1" in run_mode:
            if st.button("🚀 Bắt đầu Stream trực tiếp"):
                st.info("📺 Đang hiển thị luồng xử lý Real-time...")
                cap = cv2.VideoCapture(tfile.name)
                
                # Tạo khung trống để liên tục "bắn" ảnh vào
                frame_window = st.image([]) 
                
                while cap.isOpened():
                    success, frame = cap.read()
                    if not success:
                        break
                    
                    # Chạy ByteTrack trực tiếp trên khung hình
                    results = model.track(frame, persist=True, imgsz=img_size, conf=conf_threshold, tracker=tracker_config, verbose=False)
                    annotated_frame = results[0].plot()
                    
                    # Chuyển hệ màu để hiển thị chuẩn trên giao diện Web
                    annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    frame_window.image(annotated_frame_rgb, channels="RGB", use_container_width=True)
                
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