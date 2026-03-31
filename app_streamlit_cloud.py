import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import math
import os
import platform
from PIL import Image

# Initialize MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

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
        self.pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
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
            nose_tip = face_landmarks.landmark[1]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
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
            nose_tip = face_landmarks.landmark[1]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
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
            nose_tip = face_landmarks.landmark[1]
            left_ear = face_landmarks.landmark[234]
            right_ear = face_landmarks.landmark[454]
            
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
            nose_tip = face_landmarks.landmark[1]
            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]
            
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
            nose_tip = face_landmarks.landmark[1]
            chin_tip = face_landmarks.landmark[175]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
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
        self.pose = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5
        )
    
    def process_uploaded_image(self, uploaded_file):
        """Process uploaded image for pose detection"""
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
            pose_results = self.pose.process(rgb_image)
            face_results = self.face_mesh.process(rgb_image)
            
            # Draw landmarks if detected
            annotated_image = image_bgr.copy()
            
            if pose_results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    annotated_image, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                )
            
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(
                        annotated_image, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                    )
            
            return {
                'success': True,
                'image': cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB),
                'pose_landmarks': pose_results.pose_landmarks,
                'face_landmarks': face_results.multi_face_landmarks
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'image': None,
                'pose_landmarks': None,
                'face_landmarks': None
            }

def main():
    st.set_page_config(
        page_title="Cervical Posture Detection",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

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
        st.session_state.image_processor = ImageProcessor()
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
            st.info("📷 Camera not available - Use image upload instead")
        
        # Image upload section
        st.markdown("#### 📷 Image Upload")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear photo showing your head and shoulders"
        )
        
        if uploaded_file is not None:
            st.success("✅ Image uploaded successfully!")
        else:
            st.info("📁 Please upload an image to analyze posture")
        
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

    # Main display area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("### 📷 Image Analysis")
        
        if uploaded_file is not None:
            # Process uploaded image
            with st.spinner("🔍 Analyzing posture..."):
                result = st.session_state.image_processor.process_uploaded_image(uploaded_file)
            
            if result['success']:
                # Display processed image
                st.image(result['image'], caption="Processed Image with Pose Detection", use_column_width=True)
                
                # Analyze posture if landmarks detected
                if result['pose_landmarks'] and result['face_landmarks']:
                    detector = PostureDetector()
                    landmarks = result['pose_landmarks'].landmark
                    face_landmarks = result['face_landmarks'][0]
                    feedback = detector.analyze_posture(landmarks, face_landmarks, exercise, sensitivity)
                    
                    # Display results
                    if feedback["correct"]:
                        if "Excellent" in feedback["status"]:
                            st.markdown(f'<div class="status-excellent">🎉 {feedback["status"]}</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="status-good">✅ {feedback["status"]}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="status-poor">❌ {feedback["status"]}</div>', unsafe_allow_html=True)
                    
                    # Show tips
                    if feedback["tips"]:
                        st.markdown("#### 💡 Tips for Improvement:")
                        for tip in feedback["tips"]:
                            st.markdown(f"• {tip}")
                    
                    # Update session stats
                    if feedback["correct"]:
                        st.session_state.success_count += 1
                    st.session_state.exercise_count += 1
                else:
                    st.warning("⚠️ Could not detect pose landmarks. Please upload a clearer image with your full head and shoulders visible.")
            else:
                st.error(f"❌ Error processing image: {result['error']}")
        else:
            # Upload placeholder
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #6c757d;">
                <h3>📷 Upload Image for Analysis</h3>
                <p>Upload a clear photo showing your head and shoulders in the sidebar</p>
                <small>Supported formats: PNG, JPG, JPEG</small>
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
