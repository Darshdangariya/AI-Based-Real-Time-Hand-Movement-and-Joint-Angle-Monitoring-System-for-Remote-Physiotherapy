import streamlit as st
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision
import numpy as np
import time
import math
import os
import platform
from PIL import Image
import tempfile
import json
import hashlib

# Hand-rolled connections and drawing fallback since mp.solutions is missing on this platform
mp_pose_connections = frozenset([(15, 21), (16, 20), (18, 20), (3, 7), (14, 16), (23, 25), (28, 30), (11, 23), (27, 31), (6, 8), (15, 17), (24, 26), (16, 22), (4, 5), (5, 6), (29, 31), (12, 24), (23, 24), (0, 1), (9, 10), (1, 2), (0, 4), (11, 13), (30, 32), (28, 32), (15, 19), (16, 18), (25, 27), (26, 28), (12, 14), (17, 19), (2, 3), (11, 12), (27, 29), (13, 15)])
mp_facemesh_connections = frozenset() # Simplified to not draw complex face mesh connections unless needed

def draw_landmarks_manual(image, landmarks, connections, color_node=(0, 255, 0), color_edge=(0, 255, 255), thickness=2, radius=2):
    """Fallback manual drawing utility using cv2"""
    if not landmarks: return
    h, w, _ = image.shape
    # Draw connections
    if connections:
        for connection in connections:
            start_idx, end_idx = connection
            if start_idx < len(landmarks) and end_idx < len(landmarks):
                p1, p2 = landmarks[start_idx], landmarks[end_idx]
                px1, py1 = int(p1.x * w), int(p1.y * h)
                px2, py2 = int(p2.x * w), int(p2.y * h)
                cv2.line(image, (px1, py1), (px2, py2), color_edge, thickness)
    # Draw nodes
    for point in landmarks:
        px, py = int(point.x * w), int(point.y * h)
        cv2.circle(image, (px, py), radius, color_node, -1)


def is_streamlit_cloud():
    """Detect if running on Streamlit Cloud"""
    return (
        os.environ.get('STREAMLIT_SHARING_MODE') is not None or 
        os.environ.get('STREAMLIT_SERVER_HEADLESS') == 'true' or
        'streamlit.app' in os.environ.get('HOSTNAME', '') or
        platform.system() == 'Linux' and 'site-packages' in __file__
    )

class PostureDetector:
    def __init__(self):
        # Don't initialize MediaPipe here - do it lazily when needed
        self.pose = None
        self.face_mesh = None
        self._initialized = False
    
    def _initialize_mediapipe(self):
        """Lazy initialization of MediaPipe with error handling"""
        if self._initialized:
            return True
            
        try:
            base_options_pose = mp_python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
            options_pose = vision.PoseLandmarkerOptions(
                base_options=base_options_pose,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.pose = vision.PoseLandmarker.create_from_options(options_pose)

            base_options_face = mp_python.BaseOptions(model_asset_path='models/face_landmarker.task')
            options_face = vision.FaceLandmarkerOptions(
                base_options=base_options_face,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.face_mesh = vision.FaceLandmarker.create_from_options(options_face)
            self._initialized = True
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize MediaPipe tasks: {str(e)}\nPlease run `mkdir -p models && wget -O models/pose_landmarker_lite.task -q https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task && wget -O models/face_landmarker.task -q https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task`")
            return False
        
    def calculate_angle(self, a, b, c):
        """Calculate angle between three points"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
        angle = np.abs(radians * 180.0 / np.pi)
        
        if angle > 180.0:
            angle = 360 - angle
            
        return angle
    
    def calculate_distance(self, point1, point2):
        """Calculate distance between two points"""
        return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def analyze_posture(self, landmarks, face_landmarks, exercise, sensitivity=1.0):
        """Main posture analysis method"""
        # Initialize MediaPipe only when needed
        if not self._initialize_mediapipe():
            return {
                "status": "MediaPipe initialization failed",
                "correct": False,
                "tips": ["Please try refreshing the page"]
            }
            
        feedback = {
            "status": "Not detected",
            "correct": False,
            "tips": []
        }
        
        try:
            if exercise == "Cervical Flexion":
                result = self.detect_cervical_flexion(landmarks, face_landmarks, sensitivity)
            elif exercise == "Cervical Extension":
                result = self.detect_cervical_extension(landmarks, face_landmarks, sensitivity)
            elif exercise == "Lateral Tilt":
                result = self.detect_lateral_tilt(landmarks, face_landmarks, "left", sensitivity)
            elif exercise == "Neck Rotation":
                result = self.detect_rotation(face_landmarks, "left", sensitivity)
            elif exercise == "Chin Tuck":
                result = self.detect_chin_tuck(landmarks, face_landmarks, sensitivity)
            else:
                return feedback
                
            feedback["correct"] = result["correct"]
            feedback["status"] = result["message"]
            feedback["tips"] = result.get("tips", [])
            
        except Exception as e:
            feedback["status"] = f"Error: {str(e)}"
            feedback["tips"] = ["Position yourself properly in frame"]
            
        return feedback
    
    def detect_cervical_flexion(self, landmarks, face_landmarks, sensitivity=1.0):
        """Detect chin-to-chest movement"""
        try:
            nose_tip = face_landmarks[1]
            left_shoulder = landmarks[11] # LEFT_SHOULDER
            right_shoulder = landmarks[12] # RIGHT_SHOULDER
            
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            nose_to_shoulder_distance = abs(nose_tip.y - shoulder_center_y)
            
            baseline_distance = 0.15
            distance_ratio = nose_to_shoulder_distance / baseline_distance
            
            threshold = 0.85 / sensitivity
            
            if distance_ratio <= threshold:
                return {"correct": True, "message": "Excellent Cervical Flexion! 95%", "tips": []}
            elif distance_ratio <= threshold * 1.2:
                return {"correct": True, "message": "Good Cervical Flexion 80%", "tips": ["Bring chin closer to chest"]}
            else:
                return {"correct": False, "message": "Incomplete Flexion", "tips": ["Bring chin towards chest", "Keep shoulders relaxed"]}
                
        except Exception as e:
            return {"correct": False, "message": "Position Error", "tips": ["Face the camera directly"]}
    
    def detect_cervical_extension(self, landmarks, face_landmarks, sensitivity=1.0):
        """Detect head back/upward movement"""
        try:
            nose_tip = face_landmarks[1]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            nose_to_shoulder_distance = nose_tip.y - shoulder_center_y
            
            baseline_distance = -0.15
            
            threshold = baseline_distance * sensitivity
            
            if nose_to_shoulder_distance <= threshold:
                return {"correct": True, "message": "Excellent Cervical Extension! 95%", "tips": []}
            elif nose_to_shoulder_distance <= threshold * 0.7:
                return {"correct": True, "message": "Good Cervical Extension 80%", "tips": ["Tilt head back slightly more"]}
            else:
                return {"correct": False, "message": "Incomplete Extension", "tips": ["Gently tilt head back", "Look upward slowly"]}
                
        except Exception as e:
            return {"correct": False, "message": "Position Error", "tips": ["Face the camera directly"]}
    
    def detect_lateral_tilt(self, landmarks, face_landmarks, direction, sensitivity=1.0):
        """Detect head tilt to side"""
        try:
            nose_tip = face_landmarks[1]
            left_ear = face_landmarks[234]
            right_ear = face_landmarks[454]
            
            ear_y_diff = abs(left_ear.y - right_ear.y)
            threshold = 0.02 * sensitivity
            
            if ear_y_diff >= threshold:
                return {"correct": True, "message": "Good Lateral Tilt! 85%", "tips": []}
            else:
                return {"correct": False, "message": "Incomplete Tilt", "tips": ["Tilt head more to the side", "Keep shoulders level"]}
                
        except Exception as e:
            return {"correct": False, "message": "Position Error", "tips": ["Face the camera directly"]}
    
    def detect_rotation(self, face_landmarks, direction, sensitivity=1.0):
        """Detect head rotation left/right"""
        try:
            nose_tip = face_landmarks[1]
            left_cheek = face_landmarks[234]
            right_cheek = face_landmarks[454]
            
            nose_center_x = nose_tip.x
            face_center_x = (left_cheek.x + right_cheek.x) / 2
            rotation_offset = abs(nose_center_x - face_center_x)
            
            threshold = 0.03 * sensitivity
            
            if rotation_offset >= threshold:
                return {"correct": True, "message": "Good Neck Rotation! 85%", "tips": []}
            else:
                return {"correct": False, "message": "Incomplete Rotation", "tips": ["Turn head more to the side", "Keep chin level"]}
                
        except Exception as e:
            return {"correct": False, "message": "Position Error", "tips": ["Face the camera directly"]}
    
    def detect_chin_tuck(self, landmarks, face_landmarks, sensitivity=1.0):
        """Detect chin tuck movement"""
        try:
            nose_tip = face_landmarks[1]
            chin_tip = face_landmarks[175]
            left_shoulder = landmarks[11]
            right_shoulder = landmarks[12]
            
            shoulder_center = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
            chin_to_shoulder_distance = math.sqrt((chin_tip.x - shoulder_center[0])**2 + (chin_tip.y - shoulder_center[1])**2)
            
            baseline_distance = 0.3
            distance_ratio = chin_to_shoulder_distance / baseline_distance
            
            threshold = 0.9 / sensitivity
            
            if distance_ratio <= threshold:
                return {"correct": True, "message": "Excellent Chin Tuck! 90%", "tips": []}
            elif distance_ratio <= threshold * 1.1:
                return {"correct": True, "message": "Good Chin Tuck 75%", "tips": ["Pull chin back slightly more"]}
            else:
                return {"correct": False, "message": "Incomplete Chin Tuck", "tips": ["Pull chin back", "Create double chin effect"]}
                
        except Exception as e:
            return {"correct": False, "message": "Position Error", "tips": ["Face the camera directly", "Ensure full head and shoulders visible"]}

class ImageProcessor:
    """Handle image upload and processing for Streamlit Cloud"""
    def __init__(self):
        # Don't initialize MediaPipe here - do it lazily when needed
        self.pose = None
        self.face_mesh = None
        self._initialized = False
    
    def _initialize_mediapipe(self):
        """Lazy initialization of MediaPipe with error handling"""
        if self._initialized:
            return True
            
        try:
            base_options_pose = mp_python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
            options_pose = vision.PoseLandmarkerOptions(
                base_options=base_options_pose,
                running_mode=vision.RunningMode.IMAGE,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.pose = vision.PoseLandmarker.create_from_options(options_pose)

            base_options_face = mp_python.BaseOptions(model_asset_path='models/face_landmarker.task')
            options_face = vision.FaceLandmarkerOptions(
                base_options=base_options_face,
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.face_mesh = vision.FaceLandmarker.create_from_options(options_face)
            self._initialized = True
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize MediaPipe tasks for image: {str(e)}")
            return False
    
    def process_uploaded_image(self, uploaded_file):
        """Process uploaded image for pose detection"""
        # Initialize MediaPipe only when needed
        if not self._initialize_mediapipe():
            return {
                'success': False,
                'error': 'MediaPipe initialization failed',
                'image': None,
                'pose_landmarks': None,
                'face_landmarks': None
            }
            
        try:
            # Convert uploaded file to opencv format
            image = Image.open(uploaded_file)
            image_np = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(image_np.shape) == 3:
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image_np
                
            # Process with MediaPipe
            rgb_image = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            pose_results = self.pose.detect(mp_image)
            face_results = self.face_mesh.detect(mp_image)
            
            # Draw landmarks if detected
            annotated_image = image_bgr.copy()
            
            p_lm = None
            f_lm = None
            
            if pose_results.pose_landmarks:
                p_lm = pose_results.pose_landmarks[0]
                draw_landmarks_manual(annotated_image, p_lm, mp_pose_connections)
            
            if face_results.face_landmarks:
                f_lm = face_results.face_landmarks
                for face_landmarks in face_results.face_landmarks:
                    draw_landmarks_manual(annotated_image, face_landmarks, mp_facemesh_connections, color_node=(0, 255, 0), radius=1)
            
            return {
                'success': True,
                'image': cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                'pose_landmarks': p_lm,
                'face_landmarks': f_lm
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image': None,
                'pose_landmarks': None,
                'face_landmarks': None
            }

class VideoStreamProcessor:
    """Handle video and live camera processing"""
    def __init__(self):
        self.pose = None
        self.face_mesh = None
        self._initialized = False
        self._frame_index = 0
        
    def _initialize_mediapipe(self):
        if self._initialized:
            return True
            
        try:
            base_options_pose = mp_python.BaseOptions(model_asset_path='models/pose_landmarker_lite.task')
            options_pose = vision.PoseLandmarkerOptions(
                base_options=base_options_pose,
                running_mode=vision.RunningMode.VIDEO,
                min_pose_detection_confidence=0.5,
                min_pose_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.pose = vision.PoseLandmarker.create_from_options(options_pose)

            base_options_face = mp_python.BaseOptions(model_asset_path='models/face_landmarker.task')
            options_face = vision.FaceLandmarkerOptions(
                base_options=base_options_face,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5)
            self.face_mesh = vision.FaceLandmarker.create_from_options(options_face)
            self._initialized = True
            return True
        except Exception as e:
            st.error(f"❌ Failed to initialize MediaPipe tasks for video: {str(e)}")
            return False
            
    def process_frame(self, frame_bgr):
        if not self._initialize_mediapipe():
            return {'success': False, 'image': frame_bgr, 'pose_landmarks': None, 'face_landmarks': None}
            
        try:
            rgb_image = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            
            # Use deterministic monotonically increasing timestamp based on frame count
            # Assuming ~30 fps (33ms per frame) for both live camera and video
            self._frame_index += 1
            frame_timestamp_ms = self._frame_index * 33
            
            pose_results = self.pose.detect_for_video(mp_image, frame_timestamp_ms)
            face_results = self.face_mesh.detect_for_video(mp_image, frame_timestamp_ms)
            
            annotated_image = frame_bgr.copy()
            
            p_lm = None
            f_lm = None
            
            if pose_results.pose_landmarks:
                p_lm = pose_results.pose_landmarks[0]
                draw_landmarks_manual(annotated_image, p_lm, mp_pose_connections)
            
            if face_results.face_landmarks:
                f_lm = face_results.face_landmarks
                for face_landmarks in face_results.face_landmarks:
                    draw_landmarks_manual(annotated_image, face_landmarks, mp_facemesh_connections, color_node=(0, 255, 0), radius=1)
                    
            return {
                'success': True,
                'image': cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                'pose_landmarks': p_lm,
                'face_landmarks': f_lm
            }
        except Exception as e:
            return {'success': False, 'image': frame_bgr, 'pose_landmarks': None, 'face_landmarks': None}

def main():
    st.set_page_config(
        page_title="Cervical Posture Detection",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state for authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False

    USER_DATA_FILE = "users.json"

    def load_users():
        if os.path.exists(USER_DATA_FILE):
            try:
                with open(USER_DATA_FILE, "r") as f:
                    return json.load(f)
            except:
                return {"admin": "admin"}
        return {"admin": "admin"}

    def save_users(users):
        with open(USER_DATA_FILE, "w") as f:
            json.dump(users, f, indent=4)

    def login():
        """Login and Registration interface"""
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="text-align: center; padding: 2rem; background: #f8f9fa; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: #1f4e79;">🏥 Welcome to PostureCare</h2>
                <p>Please login or register to continue</p>
            </div>
            """, unsafe_allow_html=True)
            
            tab1, tab2 = st.tabs(["🔐 Login", "📝 Register"])
            
            with tab1:
                with st.form("login_form"):
                    username = st.text_input("Username")
                    password = st.text_input("Password", type="password")
                    submit = st.form_submit_button("Login", use_container_width=True)
                    
                    if submit:
                        users = load_users()
                        if username in users and users[username] == password:
                            st.session_state.authenticated = True
                            st.success("✅ Login successful!")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error("❌ Invalid username or password")
            
            with tab2:
                with st.form("register_form"):
                    new_username = st.text_input("Choose Username")
                    new_password = st.text_input("Choose Password", type="password")
                    confirm_password = st.text_input("Confirm Password", type="password")
                    register_submit = st.form_submit_button("Create Account", use_container_width=True)
                    
                    if register_submit:
                        if not new_username or not new_password:
                            st.error("⚠️ Please fill in all fields")
                        elif new_password != confirm_password:
                            st.error("⚠️ Passwords do not match")
                        else:
                            users = load_users()
                            if new_username in users:
                                st.error("⚠️ Username already exists")
                            else:
                                users[new_username] = new_password
                                save_users(users)
                                st.success("🎉 Account created! You can now login.")

    if not st.session_state.authenticated:
        login()
        return

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #1f4e79, #2d5aa0);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .status-excellent {
        background: linear-gradient(90deg, #28a745, #20c997);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .status-good {
        background: linear-gradient(90deg, #ffc107, #fd7e14);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    .status-poor {
        background: linear-gradient(90deg, #dc3545, #c82333);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Cervical Posture Detection System</h1>
        <p>AI-Powered Real-time Posture Monitoring for Cervical Physiotherapy</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'image_processor' not in st.session_state:
        try:
            st.session_state.image_processor = ImageProcessor()
        except Exception as e:
            st.error(f"❌ Failed to initialize image processor: {str(e)}")
            st.stop()
    if 'exercise_count' not in st.session_state:
        st.session_state.exercise_count = 0
    if 'success_count' not in st.session_state:
        st.session_state.success_count = 0

    # Sidebar
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Input method info
        if is_streamlit_cloud():
            st.warning("🌐 Running on Streamlit Cloud")
            st.info("📷 Best used with Image Upload or Video Upload on Cloud")
        
        # Input method section
        st.markdown("#### 📷 Input Method")
        input_source = st.radio("Select Input Source", ["Image Upload", "Video Upload", "Live Camera"])
        
        uploaded_file = None
        uploaded_video = None
        
        if input_source == "Image Upload":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['png', 'jpg', 'jpeg'],
                help="Upload a clear photo showing your head and shoulders"
            )
            if uploaded_file is not None:
                st.success("✅ Image uploaded successfully!")
            else:
                st.info("📁 Please upload an image to analyze posture")
                
        elif input_source == "Video Upload":
            uploaded_video = st.file_uploader(
                "Choose a video file",
                type=['mp4', 'mov', 'avi'],
                help="Upload a clear video showing your head and shoulders"
            )
            if uploaded_video is not None:
                st.success("✅ Video uploaded successfully!")
            else:
                st.info("📁 Please upload a video to analyze posture")
        else:
            st.info("🎥 Live Camera Selected")
        
        st.divider()
        
        # Exercise selection
        st.markdown("#### 💪 Exercise Selection")
        exercise_options = {
            "Cervical Flexion": "⬇️ Cervical Flexion",
            "Cervical Extension": "⬆️ Cervical Extension", 
            "Lateral Tilt": "↔️ Lateral Tilt",
            "Neck Rotation": "🔄 Neck Rotation",
            "Chin Tuck": "👤 Chin Tuck"
        }
        
        exercise = st.selectbox(
            "Choose Exercise",
            list(exercise_options.keys()),
            format_func=lambda x: exercise_options[x]
        )
        
        st.divider()
        
        # Sensitivity settings
        st.markdown("#### ⚙️ Detection Settings")
        sensitivity = st.slider(
            "🎯 Detection Sensitivity", 
            0.5, 2.0, 1.0, 0.1,
            help="Adjust if exercises are too easy (↑) or too hard (↓) to detect"
        )
        
        if sensitivity < 0.8:
            st.info("🔒 More Strict Detection")
        elif sensitivity > 1.2:
            st.warning("🔓 More Lenient Detection")
        else:
            st.success("⚖️ Balanced Detection")
        
        st.divider()
        
        # Session stats
        st.markdown("#### 📊 Session Stats")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Attempts", st.session_state.exercise_count)
        with col2:
            success_rate = (st.session_state.success_count / max(1, st.session_state.exercise_count)) * 100
            st.metric("Success Rate", f"{success_rate:.0f}%")

        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.exercise_count = 0
            st.session_state.success_count = 0
            st.rerun()

    # Main display area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown(f"### 📷 {input_source} Analysis")
        
        if input_source == "Image Upload":
            if uploaded_file is not None:
                with st.spinner("🔍 Analyzing posture..."):
                    try:
                        result = st.session_state.image_processor.process_uploaded_image(uploaded_file)
                    except Exception as e:
                        st.error(f"❌ Error during image processing: {str(e)}")
                        result = {'success': False, 'error': str(e)}
                
                if result['success']:
                    st.image(result['image'], caption="Processed Image with Pose Detection", use_column_width=True)
                    
                    if result['pose_landmarks'] and result['face_landmarks']:
                        try:
                            detector = PostureDetector()
                            landmarks = result['pose_landmarks'].landmark
                            face_landmarks = result['face_landmarks'][0]
                            feedback = detector.analyze_posture(landmarks, face_landmarks, exercise, sensitivity)
                            
                            if feedback["correct"]:
                                if "Excellent" in feedback["status"]:
                                    st.markdown(f'<div class="status-excellent">🎉 {feedback["status"]}</div>', unsafe_allow_html=True)
                                else:
                                    st.markdown(f'<div class="status-good">✅ {feedback["status"]}</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="status-poor">❌ {feedback["status"]}</div>', unsafe_allow_html=True)
                            
                            if feedback["tips"]:
                                st.markdown("#### 💡 Tips for Improvement:")
                                for tip in feedback["tips"]:
                                    st.markdown(f"• {tip}")
                            
                            if feedback["correct"]:
                                st.session_state.success_count += 1
                            st.session_state.exercise_count += 1
                        except Exception as e:
                            st.error(f"❌ Error during posture analysis: {str(e)}")
                    else:
                        st.warning("⚠️ Could not detect pose landmarks. Please upload a clearer image with your full head and shoulders visible.")
                else:
                    st.error(f"❌ Error processing image: {result.get('error', 'Unknown Error')}")
                    st.info("💡 Try uploading a different image or check if the image format is supported")
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #6c757d;">
                    <h3>📷 Upload Image for Analysis</h3>
                    <p>Upload a clear photo showing your head and shoulders in the sidebar</p>
                    <small>Supported formats: PNG, JPG, JPEG</small>
                </div>
                """, unsafe_allow_html=True)
                
        elif input_source == "Video Upload":
            if uploaded_video is not None:
                tfile = tempfile.NamedTemporaryFile(delete=False) 
                tfile.write(uploaded_video.read())
                vf = cv2.VideoCapture(tfile.name)
                
                stframe = st.empty()
                status_placeholder = st.empty()
                feedback_placeholder = st.empty()
                
                stop_button = st.button("Stop Video")
                
                if 'video_processor' not in st.session_state:
                    st.session_state.video_processor = VideoStreamProcessor()
                detector = PostureDetector()
                
                while vf.isOpened() and not stop_button:
                    ret, frame = vf.read()
                    if not ret:
                        break
                    
                    # Process frame
                    result = st.session_state.video_processor.process_frame(frame)
                    
                    if result['success']:
                        stframe.image(result['image'], channels="RGB", use_column_width=True)
                        
                        if result['pose_landmarks'] and result['face_landmarks']:
                            try:
                                landmarks = result['pose_landmarks']
                                face_landmarks = result['face_landmarks'][0]
                                feedback = detector.analyze_posture(landmarks, face_landmarks, exercise, sensitivity)
                                
                                if feedback["correct"]:
                                    if "Excellent" in feedback["status"]:
                                        status_placeholder.markdown(f'<div class="status-excellent">🎉 {feedback["status"]}</div>', unsafe_allow_html=True)
                                    else:
                                        status_placeholder.markdown(f'<div class="status-good">✅ {feedback["status"]}</div>', unsafe_allow_html=True)
                                else:
                                    status_placeholder.markdown(f'<div class="status-poor">❌ {feedback["status"]}</div>', unsafe_allow_html=True)
                                    
                                tips_html = "#### 💡 Tips for Improvement:<br>" + "<br>".join([f"• {tip}" for tip in feedback.get("tips", [])])
                                feedback_placeholder.markdown(tips_html, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error during video posture analysis: {str(e)}")
                    else:
                        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                        if 'error' in result:
                            st.warning(f"Video frame processing error: {result['error']}")
                        
                vf.release()
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #6c757d;">
                    <h3>🎥 Upload Video for Analysis</h3>
                    <p>Upload a clear video showing your head and shoulders in the sidebar</p>
                    <small>Supported formats: MP4, MOV, AVI</small>
                </div>
                """, unsafe_allow_html=True)
                
        elif input_source == "Live Camera":
            run = st.checkbox("Turn on Live Camera")
            stframe = st.empty()
            status_placeholder = st.empty()
            feedback_placeholder = st.empty()
            
            if run:
                camera = cv2.VideoCapture(0)
                if 'video_processor' not in st.session_state:
                    st.session_state.video_processor = VideoStreamProcessor()
                detector = PostureDetector()
                
                while run:
                    ret, frame = camera.read()
                    if not ret:
                        st.error("Failed to access camera.")
                        break
                        
                    result = st.session_state.video_processor.process_frame(frame)
                    
                    if result['success']:
                        stframe.image(result['image'], channels="RGB", use_column_width=True)
                        
                        if result['pose_landmarks'] and result['face_landmarks']:
                            try:
                                landmarks = result['pose_landmarks']
                                face_landmarks = result['face_landmarks'][0]
                                feedback = detector.analyze_posture(landmarks, face_landmarks, exercise, sensitivity)
                                
                                if feedback["correct"]:
                                    if "Excellent" in feedback["status"]:
                                        status_placeholder.markdown(f'<div class="status-excellent">🎉 {feedback["status"]}</div>', unsafe_allow_html=True)
                                    else:
                                        status_placeholder.markdown(f'<div class="status-good">✅ {feedback["status"]}</div>', unsafe_allow_html=True)
                                else:
                                    status_placeholder.markdown(f'<div class="status-poor">❌ {feedback["status"]}</div>', unsafe_allow_html=True)
                                    
                                tips_html = "#### 💡 Tips for Improvement:<br>" + "<br>".join([f"• {tip}" for tip in feedback.get("tips", [])])
                                feedback_placeholder.markdown(tips_html, unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error during live camera posture analysis: {str(e)}")
                    else:
                        stframe.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
                        if 'error' in result:
                            st.warning(f"Camera frame processing error: {result['error']}")
                        
                camera.release()
            else:
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #6c757d;">
                    <h3>📷 Live Camera Analysis</h3>
                    <p>Check "Turn on Live Camera" to start detecting in real-time</p>
                </div>
                """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Exercise Guidelines")
        
        exercise_info = {
            "Cervical Flexion": {
                "description": "Slowly bring your chin towards your chest",
                "tips": ["Clinical ROM: 45-50°", "Hold for 5-8 seconds", "Feel C7-T1 stretch"],
            },
            "Cervical Extension": {
                "description": "Gently tilt your head back to look upward", 
                "tips": ["Clinical ROM: 45-55°", "Don't force beyond comfort", "Control the movement"],
            },
            "Lateral Tilt": {
                "description": "Tilt your head to bring ear towards shoulder",
                "tips": ["Normal ROM: 40-45°", "Keep shoulders level", "Feel contralateral stretch"],
            },
            "Neck Rotation": {
                "description": "Turn your head left and right",
                "tips": ["Normal ROM: 80-90°", "Keep chin level", "Smooth controlled movement"],
            },
            "Chin Tuck": {
                "description": "Pull your chin back creating a double chin",
                "tips": ["Activate deep cervical flexors", "Feel suboccipital stretch", "Maintain eye level"],
            }
        }
        
        info = exercise_info[exercise]
        st.markdown(f"**{exercise}**")
        st.markdown(f"_{info['description']}_")
        
        st.markdown("#### 💡 Tips:")
        for tip in info["tips"]:
            st.markdown(f"• {tip}")
    
    # Footer
    st.divider()
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        if st.button("🔄 Reset Session", use_container_width=True):
            st.session_state.exercise_count = 0
            st.session_state.success_count = 0
            st.success("Session reset!")
    
    with col2:
        st.markdown("""
        <div style="text-align: center; padding: 1rem;">
            <small>💡 <strong>Tip:</strong> Take clear photos with good lighting for best results!</small>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
