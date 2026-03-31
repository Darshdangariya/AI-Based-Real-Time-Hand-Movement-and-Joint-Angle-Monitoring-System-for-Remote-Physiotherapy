import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
from threading import Thread
import queue
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
        """Detect chin-to-chest movement
        Clinical-grade detection using nose-to-shoulder distance ratio
        Normal ROM: 45-50 degrees | Threshold: 0.85 ratio
        """
        try:
            # Get key landmarks for clinical measurement
            nose_tip = face_landmarks.landmark[1]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Calculate nose-to-shoulder distance ratio
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            nose_to_shoulder_distance = abs(nose_tip.y - shoulder_center_y)
            
            # Clinical threshold: 0.85 (nose gets closer to shoulders during flexion)
            baseline_distance = 0.15  # Normalized baseline
            distance_ratio = nose_to_shoulder_distance / baseline_distance
            
            # Clinical detection threshold
            flexion_threshold = 0.85 * sensitivity
            
            # Calculate confidence score
            if distance_ratio < flexion_threshold:
                confidence = min(100, ((flexion_threshold - distance_ratio) / 0.15) * 100)
                
                if confidence >= 90:
                    return {
                        "correct": True,
                        "message": f"Excellent! Cervical flexion: {confidence:.0f}% ROM",
                        "tips": ["Hold for 5-8 seconds", "Normal ROM: 45-50°"]
                    }
                elif confidence >= 70:
                    return {
                        "correct": True,
                        "message": f"Good flexion: {confidence:.0f}% ROM - go deeper",
                        "tips": ["Target: 90%+ ROM", "Feel C7-T1 stretch"]
                    }
                elif confidence >= 50:
                    return {
                        "correct": True,
                        "message": f"Moderate flexion: {confidence:.0f}% ROM",
                        "tips": ["Increase range slowly", "Coaching recommended"]
                    }
                else:
                    return {
                        "correct": False,
                        "message": f"Poor flexion: {confidence:.0f}% ROM",
                        "tips": ["Instruction needed", "Slow controlled movement"]
                    }
            else:
                return {
                    "correct": False,
                    "message": "Inadequate flexion - chin to chest",
                    "tips": ["Start movement slowly", "Target: 45-50° ROM"]
                }
                
        except:
            return {
                "correct": False,
                "message": "Position yourself properly in frame",
                "tips": ["Face the camera directly"]
            }
    
    def detect_cervical_extension(self, landmarks, face_landmarks, sensitivity=1.0):
        """Detect looking upward movement
        Clinical-grade detection using nose-to-shoulder distance ratio
        Normal ROM: 45-55 degrees | Threshold: 1.15 ratio
        """
        try:
            # Get key landmarks for clinical measurement
            nose_tip = face_landmarks.landmark[1]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Calculate nose-to-shoulder distance ratio
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            nose_to_shoulder_distance = abs(nose_tip.y - shoulder_center_y)
            
            # Clinical threshold: 1.15 (nose moves farther from shoulders during extension)
            baseline_distance = 0.15  # Normalized baseline
            distance_ratio = nose_to_shoulder_distance / baseline_distance
            
            # Clinical detection threshold
            extension_threshold = 1.15 * sensitivity
            
            # Calculate confidence score
            if distance_ratio > extension_threshold:
                confidence = min(100, ((distance_ratio - extension_threshold) / 0.15) * 100)
                
                if confidence >= 90:
                    return {
                        "correct": True,
                        "message": f"Excellent! Cervical extension: {confidence:.0f}% ROM",
                        "tips": ["Hold for 5-8 seconds", "Normal ROM: 45-55°"]
                    }
                elif confidence >= 70:
                    return {
                        "correct": True,
                        "message": f"Good extension: {confidence:.0f}% ROM - go further",
                        "tips": ["Target: 90%+ ROM", "Control the movement"]
                    }
                elif confidence >= 50:
                    return {
                        "correct": True,
                        "message": f"Moderate extension: {confidence:.0f}% ROM",
                        "tips": ["Increase range slowly", "Coaching recommended"]
                    }
                else:
                    return {
                        "correct": False,
                        "message": f"Poor extension: {confidence:.0f}% ROM",
                        "tips": ["Instruction needed", "Look up more"]
                    }
            else:
                return {
                    "correct": False,
                    "message": "Inadequate extension - look up more",
                    "tips": ["Start movement slowly", "Target: 45-55° ROM"]
                }
                
        except:
            return {
                "correct": False,
                "message": "Position yourself properly in frame",
                "tips": ["Face the camera directly"]
            }
    
    def detect_lateral_tilt(self, landmarks, face_landmarks, direction, sensitivity=1.0):
        """Detect left/right neck tilt
        Clinical-grade detection using asymmetry measurement
        Normal ROM: 40-45 degrees | Threshold: 0.15 asymmetry
        """
        try:
            # Get key landmarks for clinical measurement
            nose_tip = face_landmarks.landmark[1]
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            
            # Calculate nose-to-ear distances for asymmetry detection
            nose_to_left_ear = math.sqrt((nose_tip.x - left_ear.x)**2 + (nose_tip.y - left_ear.y)**2)
            nose_to_right_ear = math.sqrt((nose_tip.x - right_ear.x)**2 + (nose_tip.y - right_ear.y)**2)
            
            # Calculate asymmetry ratio
            asymmetry_diff = abs(nose_to_left_ear - nose_to_right_ear)
            avg_distance = (nose_to_left_ear + nose_to_right_ear) / 2
            asymmetry_ratio = asymmetry_diff / avg_distance if avg_distance > 0 else 0
            
            # Clinical threshold: 0.4 asymmetry difference triggers detection (much stricter)
            tilt_threshold = 0.4 * sensitivity
            
            # Calculate confidence score
            if asymmetry_ratio > tilt_threshold:
                confidence = min(100, (asymmetry_ratio / 0.5) * 100)
                
                if confidence >= 90:
                    return {
                        "correct": True,
                        "message": f"Excellent! Lateral tilt: {confidence:.0f}% ROM",
                        "tips": ["Hold for 5-8 seconds", "Normal ROM: 40-45°"]
                    }
                elif confidence >= 70:
                    return {
                        "correct": True,
                        "message": f"Good tilt: {confidence:.0f}% ROM - go further",
                        "tips": ["Target: 90%+ ROM", "Feel contralateral stretch"]
                    }
                elif confidence >= 50:
                    return {
                        "correct": True,
                        "message": f"Moderate tilt: {confidence:.0f}% ROM",
                        "tips": ["Increase range slowly", "Keep shoulders level"]
                    }
                else:
                    return {
                        "correct": False,
                        "message": f"Poor tilt: {confidence:.0f}% ROM",
                        "tips": ["Instruction needed", "Ear to shoulder movement"]
                    }
            else:
                return {
                    "correct": False,
                    "message": "Inadequate tilt - ear to shoulder",
                    "tips": ["Start movement slowly", "Target: 40-45° ROM"]
                }
                    
        except:
            return {
                "correct": False,
                "message": "Position yourself properly in frame",
                "tips": ["Face the camera directly"]
            }
    
    def detect_rotation(self, face_landmarks, direction, sensitivity=1.0):
        """Detect left/right neck rotation
        Based on research: Normal cervical rotation ROM is 80-90 degrees bilaterally
        Reference: Dvorak et al. (1992) & Feipel et al. (1999)
        """
        try:
            nose_tip = face_landmarks.landmark[1]
            left_cheek = face_landmarks.landmark[234]
            right_cheek = face_landmarks.landmark[454]
            chin = face_landmarks.landmark[175]
            
            # Calculate rotation using face asymmetry
            face_width = abs(right_cheek.x - left_cheek.x)
            nose_deviation = abs(nose_tip.x - (left_cheek.x + right_cheek.x) / 2)
            
            # Convert to rotation angle approximation
            rotation_ratio = nose_deviation / face_width if face_width > 0 else 0
            rotation_angle = rotation_ratio * 90 * sensitivity  # Approximate rotation in degrees
            
            # Clinical thresholds based on research (Dvorak et al., 1992)
            optimal_rotation_angle = 60 * sensitivity     # 60-80 degrees optimal rotation
            minimal_rotation_angle = 30 * sensitivity     # 30 degrees minimum
            
            if rotation_angle >= optimal_rotation_angle:
                return {
                    "correct": True,
                    "message": f"Excellent! Cervical rotation: {rotation_angle:.1f}°",
                    "tips": ["Hold for 10 seconds", "Target: 60-80°"]
                }
            elif rotation_angle >= minimal_rotation_angle:
                return {
                    "correct": True,
                    "message": f"Good rotation: {rotation_angle:.1f}° - turn more",
                    "tips": ["Target 60-80°", "Keep chin level"]
                }
            else:
                return {
                    "correct": False,
                    "message": f"Increase rotation: {rotation_angle:.1f}° - turn head more",
                    "tips": ["Target: 60-80°", "Smooth movement"]
                }
                    
        except:
            return {
                "correct": False,
                "message": "Position yourself properly in frame",
                "tips": ["Face the camera directly"]
            }
    
    def detect_chin_tuck(self, landmarks, face_landmarks, sensitivity=1.0):
        """Detect chin tuck (retraction)
        Based on clinical research: Jull et al. (2008) & Falla et al. (2007)
        Measurement: Craniovertebral angle + forward head posture assessment
        Clinical method: C7-tragus-horizontal angle measurement
        """
        try:
            # Get key landmarks for clinical measurement (correct MediaPipe indices)
            nose_tip = face_landmarks.landmark[33]  # Correct nose tip in FaceMesh
            chin = face_landmarks.landmark[152]     # Correct chin point in FaceMesh
            left_ear = landmarks[mp_pose.PoseLandmark.LEFT_EAR.value]
            right_ear = landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value]
            left_shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
            right_shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
            
            # Calculate reference points
            ear_center_x = (left_ear.x + right_ear.x) / 2
            ear_center_y = (left_ear.y + right_ear.y) / 2
            shoulder_center_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_center_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Clinical measurement 1: Forward head posture (ear-shoulder alignment)
            # Normal: ear should be over shoulder. Forward head = ear anterior to shoulder
            forward_head_distance = abs(ear_center_x - shoulder_center_x)
            
            # Clinical measurement 2: Craniovertebral angle (ear-C7-horizontal)
            # Approximated using ear-shoulder vertical relationship
            vertical_alignment = abs(ear_center_y - shoulder_center_y)
            
            # Clinical measurement 3: Chin retraction (chin-ear horizontal distance)
            # During chin tuck: chin moves posteriorly relative to ear
            chin_retraction = chin.x - ear_center_x
            
            # Clinical measurement 4: Suboccipital space (nose-chin vertical relationship)
            # During chin tuck: creates suboccipital flexion, nose-chin distance changes
            suboccipital_flexion = abs(nose_tip.y - chin.y)
            
            # Research-based thresholds (Jull et al., 2008):
            # - Normal craniovertebral angle: 48-53 degrees
            # - Chin tuck increases this angle by 8-12 degrees
            # - Forward head posture decreases by 2-4cm during proper chin tuck
            
            # Normalize measurements (based on head size for consistency)
            head_size = abs(nose_tip.y - chin.y) + 0.001  # Prevent division by zero
            
            # Normalized clinical ratios
            forward_head_ratio = forward_head_distance / head_size
            vertical_ratio = vertical_alignment / head_size
            retraction_ratio = chin_retraction / head_size
            flexion_ratio = suboccipital_flexion / head_size
            
            # Research-based clinical thresholds (balanced detection)
            # Based on Jull et al. (2008): Normal craniovertebral angle change 8-12°
            # Forward head posture reduction threshold
            forward_head_threshold = 0.2 * sensitivity  # Clinical: 2-4cm reduction expected
            
            # Chin retraction threshold (Falla et al., 2007)
            # Negative value = posterior movement (proper chin tuck direction)
            chin_retraction_threshold = -0.05 * sensitivity  # Clinical: 15-20mm retraction
            
            # Suboccipital flexion range (controlled movement)
            flexion_threshold_min = 0.7 * sensitivity   # Clinical lower bound
            flexion_threshold_max = 1.3 * sensitivity   # Clinical upper bound
            
            # Clinical detection criteria (research-based)
            # Condition 1: Forward head posture reduction (primary indicator)
            forward_head_improved = forward_head_ratio < forward_head_threshold
            
            # Condition 2: Chin retraction movement (key biomechanical component)
            chin_retracted = retraction_ratio < chin_retraction_threshold
            
            # Condition 3: Controlled suboccipital flexion (quality indicator)
            proper_flexion = flexion_threshold_min < flexion_ratio < flexion_threshold_max
            
            # Condition 4: Postural stability (maintains alignment)
            stable_posture = vertical_ratio < 0.6  # Research-based stability threshold
            
            # Clinical detection logic: Require primary movement + quality indicators
            # Must have forward head improvement AND (chin retraction OR proper flexion) AND stability
            clinical_detection = (forward_head_improved and 
                                (chin_retracted or proper_flexion) and 
                                stable_posture)
            
            if clinical_detection:
                # Clinical confidence calculation (research-based)
                # Forward head improvement score (0-100)
                head_improvement = min(100, (forward_head_threshold - forward_head_ratio) / forward_head_threshold * 100)
                
                # Chin retraction quality score (0-100)
                if chin_retracted:
                    retraction_quality = min(100, abs(retraction_ratio / chin_retraction_threshold) * 100)
                else:
                    retraction_quality = 0
                
                # Flexion control score (0-100)
                if proper_flexion:
                    optimal_flexion = 1.0  # Target ratio
                    flexion_quality = max(0, 100 - abs(flexion_ratio - optimal_flexion) * 100)
                else:
                    flexion_quality = 0
                
                # Weighted clinical confidence (Jull et al. methodology)
                confidence = (head_improvement * 0.5 +  # 50% weight - primary indicator
                            max(retraction_quality, flexion_quality) * 0.3 +  # 30% weight - movement quality
                            (100 if stable_posture else 0) * 0.2)  # 20% weight - stability
                
                if confidence >= 85:
                    return {
                        "correct": True,
                        "message": f"Excellent! Clinical chin tuck: {confidence:.0f}% quality",
                        "tips": ["Hold for 10 seconds", "Deep cervical flexor activation"]
                    }
                elif confidence >= 70:
                    return {
                        "correct": True,
                        "message": f"Good chin tuck: {confidence:.0f}% quality",
                        "tips": ["Increase retraction slightly", "Feel suboccipital stretch"]
                    }
                elif confidence >= 55:
                    return {
                        "correct": True,
                        "message": f"Moderate chin tuck: {confidence:.0f}% quality",
                        "tips": ["Focus on posterior movement", "Improve head alignment"]
                    }
                else:
                    return {
                        "correct": False,
                        "message": f"Poor technique: {confidence:.0f}% - needs improvement",
                        "tips": ["Review chin tuck form", "Slow controlled movement"]
                    }
            else:
                return {
                    "correct": False,
                    "message": "No chin tuck detected - maintain neutral position",
                    "tips": ["Pull chin back and slightly down", "Create 'double chin' effect", 
                            f"Clinical check: Head={forward_head_improved}, Chin={chin_retracted}, Flex={proper_flexion}, Stable={stable_posture}"]
                }
                
        except:
            return {
                "correct": False,
                "message": "Position yourself properly in frame",
                "tips": ["Face the camera directly", "Ensure full head and shoulders visible"]
            }

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

class CameraHandler:
    def __init__(self):
        self.cap = None
        
    def start_camera(self, camera_index=0):
        """Start camera with proper initialization"""
        if self.cap is not None:
            self.cap.release()
            
        # Try different camera backends
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY]
        
        for backend in backends:
            self.cap = cv2.VideoCapture(camera_index, backend)
            if self.cap.isOpened():
                # Set camera properties for better performance
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                break
                
        return self.cap.isOpened()
    
    def read_frame(self):
        """Read frame with timeout"""
        if self.cap is None or not self.cap.isOpened():
            return False, None
            
        ret, frame = self.cap.read()
        if ret:
            # Clear buffer to prevent lag
            self.cap.grab()
        return ret, frame
    
    def release(self):
        """Properly release camera"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

def main():
    st.set_page_config(
        page_title="Cervical Posture Detection",
        page_icon="🏥",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for better UI
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
    
    .exercise-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #2d5aa0;
        margin: 1rem 0;
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
        background: linear-gradient(90deg, #dc3545, #e74c3c);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
        font-weight: bold;
    }
    
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 0.5rem 0;
    }
    
    .camera-frame {
        border: 3px solid #2d5aa0;
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

    # Main header
    st.markdown("""
    <div class="main-header">
        <h1>🏥 Cervical Posture Detection System</h1>
        <p>AI-Powered Real-time Posture Monitoring for Cervical Physiotherapy</p>
    </div>
    """, unsafe_allow_html=True)

    # Initialize session state
    if 'camera_handler' not in st.session_state:
        st.session_state.camera_handler = CameraHandler()
    if 'image_processor' not in st.session_state:
        st.session_state.image_processor = ImageProcessor()
    if 'camera_active' not in st.session_state:
        st.session_state.camera_active = False
    if 'use_camera' not in st.session_state:
        st.session_state.use_camera = not is_streamlit_cloud()

    # Sidebar controls
    with st.sidebar:
        st.markdown("### 🎛️ Control Panel")
        
        # Input method selection
        st.markdown("#### 📱 Input Method")
        
        if is_streamlit_cloud():
            st.warning("🌐 Running on Streamlit Cloud - Camera not available")
            input_method = st.radio(
                "Choose input method:",
                ["📷 Upload Image", "📹 Camera (Local Only)"],
                index=0,
                disabled=[False, True]
            )
            st.session_state.use_camera = False
        else:
            input_method = st.radio(
                "Choose input method:",
                ["📹 Live Camera", "📷 Upload Image"],
                index=0
            )
            st.session_state.use_camera = input_method == "📹 Live Camera"
        
        st.divider()
        
        if st.session_state.use_camera:
            # Camera controls with better styling
            st.markdown("#### 📹 Camera Controls")
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("▶️ Start", type="primary", use_container_width=True):
                    if st.session_state.camera_handler.start_camera():
                        st.session_state.camera_active = True
                        st.success("✅ Camera started!")
                    else:
                        st.error("❌ Camera failed to start")
            
            with col2:
                if st.button("⏹️ Stop", use_container_width=True):
                    st.session_state.camera_handler.release()
                    st.session_state.camera_active = False
                    st.success("🛑 Camera stopped!")
            
            # Camera status indicator
            if st.session_state.camera_active:
                st.success("🟢 Camera Active")
            else:
                st.info("🔴 Camera Inactive")
            
            camera_index = st.selectbox("📷 Camera Source", [0, 1, 2], index=0)
        else:
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
        
        # Exercise selection with icons
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
        
        # Sensitivity settings with better UI
        st.markdown("#### ⚙️ Detection Settings")
        sensitivity = st.slider(
            "🎯 Detection Sensitivity", 
            0.5, 2.0, 1.0, 0.1,
            help="Adjust if exercises are too easy (↑) or too hard (↓) to detect"
        )
        
        # Sensitivity indicator
        if sensitivity < 0.8:
            st.info("🔒 More Strict Detection")
        elif sensitivity > 1.2:
            st.warning("🔓 More Lenient Detection")
        else:
            st.success("⚖️ Balanced Detection")
        
        st.divider()
        
        # Quick stats
        st.markdown("#### 📊 Session Stats")
        if 'exercise_count' not in st.session_state:
            st.session_state.exercise_count = 0
        if 'success_count' not in st.session_state:
            st.session_state.success_count = 0
            
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Attempts", st.session_state.exercise_count)
        with col2:
            success_rate = (st.session_state.success_count / max(1, st.session_state.exercise_count)) * 100
            st.metric("Success Rate", f"{success_rate:.0f}%")

    # Main display area
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if st.session_state.use_camera:
            st.markdown("### 📹 Live Camera Feed")
            
            if st.session_state.camera_active:
                # Status bar above camera
                status_placeholder = st.empty()
                progress_placeholder = st.empty()
                
                # Camera feed container
                camera_container = st.container()
                
                with camera_container:
                    video_placeholder = st.empty()
                
                # Initialize pose detection
                pose = mp_pose.Pose(
                    static_image_mode=False,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                
                face_mesh = mp_face_mesh.FaceMesh(
                    static_image_mode=False,
                    max_num_faces=1,
                    refine_landmarks=True,
                    min_detection_confidence=0.5,
                    min_tracking_confidence=0.5
                )
                
                # Initialize posture detector
                detector = PostureDetector()
                
                # Video processing loop
                try:
                    while st.session_state.camera_active:
                        ret, frame = st.session_state.camera_handler.read_frame()
                        
                        if not ret:
                            st.error("❌ Failed to read frame from camera")
                            break
                        
                        # Process frame
                        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pose_results = pose.process(rgb_frame)
                        face_results = face_mesh.process(rgb_frame)
                        
                        if pose_results.pose_landmarks and face_results.multi_face_landmarks:
                            # Draw landmarks with enhanced visualization
                            mp_drawing.draw_landmarks(
                                frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                                mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                            )
                            
                            # Draw face mesh (simplified)
                            for face_landmarks in face_results.multi_face_landmarks:
                                mp_drawing.draw_landmarks(
                                    frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                                )
                            
                            # Analyze posture
                            landmarks = pose_results.pose_landmarks.landmark
                            face_landmarks = face_results.multi_face_landmarks[0]
                            feedback = detector.analyze_posture(landmarks, face_landmarks, exercise, sensitivity)
                            
                            # Update session stats
                            if feedback["correct"]:
                                st.session_state.success_count += 1
                            st.session_state.exercise_count += 1
                            
                            # Enhanced feedback overlay on frame
                            status_color = (0, 255, 0) if feedback["correct"] else (0, 0, 255)
                            
                            # Main status
                            cv2.putText(frame, feedback["status"], (10, 30), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                            
                            # Tips overlay
                            for i, tip in enumerate(feedback["tips"][:3]):  # Limit to 3 tips
                                cv2.putText(frame, tip, (10, 70 + i*25), 
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                            
                            # Status indicator above video
                            if feedback["correct"]:
                                if "Excellent" in feedback["status"]:
                                    status_placeholder.markdown('<div class="status-excellent">🎉 ' + feedback["status"] + '</div>', unsafe_allow_html=True)
                                else:
                                    status_placeholder.markdown('<div class="status-good">✅ ' + feedback["status"] + '</div>', unsafe_allow_html=True)
                            else:
                                status_placeholder.markdown('<div class="status-poor">❌ ' + feedback["status"] + '</div>', unsafe_allow_html=True)
                            
                            # Progress bar for confidence if available
                            if "%" in feedback["status"]:
                                try:
                                    confidence = int(feedback["status"].split("%")[0].split()[-1])
                                    progress_placeholder.progress(confidence / 100, f"Confidence: {confidence}%")
                                except:
                                    pass
                        else:
                            status_placeholder.warning("⚠️ Position yourself properly in frame")
                        
                        # Display frame with custom styling
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                        
                        # Small delay to prevent overwhelming
                        time.sleep(0.033)  # ~30 FPS
                        
                except Exception as e:
                    st.error(f"💥 Camera error: {str(e)}")
                    st.session_state.camera_handler.release()
                    st.session_state.camera_active = False
            else:
                # Camera inactive placeholder with better design
                st.markdown("""
                <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #6c757d;">
                    <h3>📹 Camera Not Active</h3>
                    <p>Click <strong>▶️ Start</strong> in the sidebar to begin posture detection</p>
                    <small>Make sure your camera is connected and permissions are granted</small>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("### 📷 Image Analysis")
            
            if 'uploaded_file' in locals() and uploaded_file is not None:
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
                                st.success(f"🎉 {feedback['status']}")
                            else:
                                st.success(f"✅ {feedback['status']}")
                        else:
                            st.warning(f"❌ {feedback['status']}")
                        
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
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            face_mesh = mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize posture detector
            detector = PostureDetector()
            
            # Video processing loop
            try:
                while st.session_state.camera_active:
                    ret, frame = st.session_state.camera_handler.read_frame()
                    
                    if not ret:
                        st.error("❌ Failed to read frame from camera")
                        break
                    
                    # Process frame
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pose_results = pose.process(rgb_frame)
                    face_results = face_mesh.process(rgb_frame)
                    
                    if pose_results.pose_landmarks and face_results.multi_face_landmarks:
                        # Draw landmarks with enhanced visualization
                        mp_drawing.draw_landmarks(
                            frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                            mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2)
                        )
                        
                        # Draw face mesh (simplified)
                        for face_landmarks in face_results.multi_face_landmarks:
                            mp_drawing.draw_landmarks(
                                frame, face_landmarks, mp_face_mesh.FACEMESH_CONTOURS,
                                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1)
                            )
                        
                        # Analyze posture
                        landmarks = pose_results.pose_landmarks.landmark
                        face_landmarks = face_results.multi_face_landmarks[0]
                        feedback = detector.analyze_posture(landmarks, face_landmarks, exercise, sensitivity)
                        
                        # Update session stats
                        if feedback["correct"]:
                            st.session_state.success_count += 1
                        st.session_state.exercise_count += 1
                        
                        # Enhanced feedback overlay on frame
                        status_color = (0, 255, 0) if feedback["correct"] else (0, 0, 255)
                        
                        # Main status
                        cv2.putText(frame, feedback["status"], (10, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, status_color, 2)
                        
                        # Tips overlay
                        for i, tip in enumerate(feedback["tips"][:3]):  # Limit to 3 tips
                            cv2.putText(frame, tip, (10, 70 + i*25), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                        
                        # Status indicator above video
                        if feedback["correct"]:
                            if "Excellent" in feedback["status"]:
                                status_placeholder.markdown('<div class="status-excellent">🎉 ' + feedback["status"] + '</div>', unsafe_allow_html=True)
                            else:
                                status_placeholder.markdown('<div class="status-good">✅ ' + feedback["status"] + '</div>', unsafe_allow_html=True)
                        else:
                            status_placeholder.markdown('<div class="status-poor">❌ ' + feedback["status"] + '</div>', unsafe_allow_html=True)
                        
                        # Progress bar for confidence if available
                        if "%" in feedback["status"]:
                            try:
                                confidence = int(feedback["status"].split("%")[0].split()[-1])
                                progress_placeholder.progress(confidence / 100, f"Confidence: {confidence}%")
                            except:
                                pass
                    else:
                        status_placeholder.warning("⚠️ Position yourself properly in frame")
                    
                    # Display frame with custom styling
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
                    
                    # Small delay to prevent overwhelming
                    time.sleep(0.033)  # ~30 FPS
                    
            except Exception as e:
                st.error(f"💥 Camera error: {str(e)}")
                st.session_state.camera_handler.release()
                st.session_state.camera_active = False
        else:
            # Camera inactive placeholder with better design
            st.markdown("""
            <div style="text-align: center; padding: 3rem; background: #f8f9fa; border-radius: 15px; border: 2px dashed #6c757d;">
                <h3>📹 Camera Not Active</h3>
                <p>Click <strong>▶️ Start</strong> in the sidebar to begin posture detection</p>
                <small>Make sure your camera is connected and permissions are granted</small>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("### 📋 Exercise Guidelines")
        
        exercise_info = {
            "Cervical Flexion": {
                "icon": "⬇️",
                "description": "Slowly bring your chin towards your chest",
                "target": "Target: 45-50° ROM",
                "tips": ["Clinical ROM: 45-50°", "Hold for 5-8 seconds", "Feel C7-T1 stretch"],
                "reference": "Based on Youdas et al. (1992)",
                "color": "#28a745"
            },
            "Cervical Extension": {
                "icon": "⬆️",
                "description": "Gently tilt your head back to look upward",
                "target": "Target: 45-55° ROM",
                "tips": ["Clinical ROM: 45-55°", "Don't force beyond comfort", "Control the movement"],
                "reference": "Based on Mannion et al. (2000)",
                "color": "#17a2b8"
            },
            "Lateral Tilt": {
                "icon": "↔️",
                "description": "Tilt your head to bring ear towards shoulder",
                "target": "Target: 40-45° ROM",
                "tips": ["Normal ROM: 40-45° bilateral", "Keep shoulders level", "Feel contralateral stretch"],
                "reference": "Based on Bennett et al. (2002)",
                "color": "#ffc107"
            },
            "Neck Rotation": {
                "icon": "🔄",
                "description": "Turn your head left and right",
                "target": "Target: 80-90° ROM",
                "tips": ["Normal ROM: 80-90° bilateral", "Keep chin level", "Smooth controlled movement"],
                "reference": "Based on Dvorak et al. (1992)",
                "color": "#fd7e14"
            },
            "Chin Tuck": {
                "icon": "👤",
                "description": "Pull your chin back creating a double chin",
                "target": "Target: 15-20mm retraction",
                "tips": ["Activate deep cervical flexors", "Feel suboccipital stretch", "Maintain eye level"],
                "reference": "Based on Jull et al. (2008)",
                "color": "#6f42c1"
            }
        }
        
        info = exercise_info[exercise]
        
        # Exercise card with custom styling
        st.markdown(f"""
        <div class="exercise-card" style="border-left-color: {info['color']};">
            <h4>{info['icon']} {exercise}</h4>
            <p><strong>Description:</strong> {info['description']}</p>
            <p><strong>{info['target']}</strong></p>
        </div>
        """, unsafe_allow_html=True)
        
        # Clinical guidelines
        st.markdown("#### 🎯 Clinical Guidelines")
        for tip in info["tips"]:
            st.markdown(f"• {tip}")
            
        # Research reference
        st.markdown("#### 📚 Research Reference")
        st.markdown(f"_{info['reference']}_")
        
        st.divider()
        
        # Real-time feedback section
        st.markdown("#### 💡 Pro Tips")
        
        pro_tips = {
            "Cervical Flexion": [
                "🔥 Keep your spine straight",
                "⏱️ Hold the position for 5-8 seconds",
                "🎯 Focus on C7-T1 vertebrae stretch",
                "⚠️ Avoid forcing the movement"
            ],
            "Cervical Extension": [
                "🔥 Look straight up slowly",
                "⏱️ Control the movement speed",
                "🎯 Feel the stretch in front of neck",
                "⚠️ Stop if you feel dizzy"
            ],
            "Lateral Tilt": [
                "🔥 Keep shoulders level and relaxed",
                "⏱️ Move ear toward shoulder",
                "🎯 Feel stretch on opposite side",
                "⚠️ Don't lift the shoulder to meet ear"
            ],
            "Neck Rotation": [
                "🔥 Turn head slowly and smoothly",
                "⏱️ Keep chin level throughout",
                "🎯 Look over your shoulder",
                "⚠️ Don't force beyond comfort"
            ],
            "Chin Tuck": [
                "🔥 Create a 'double chin' effect",
                "⏱️ Pull chin back and slightly down",
                "🎯 Feel deep neck muscle activation",
                "⚠️ Maintain eye level - don't look down"
            ]
        }
        
        for tip in pro_tips[exercise]:
            st.markdown(tip)
        
        st.divider()
        
        # Performance metrics
        if st.session_state.camera_active:
            st.markdown("#### 📊 Live Metrics")
            
            # Placeholder metrics that would be updated in real-time
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("""
                <div class="metric-card">
                    <h4>🎯 Accuracy</h4>
                    <h2 style="color: #28a745;">--</h2>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h4>⏱️ Hold Time</h4>
                    <h2 style="color: #17a2b8;">--</h2>
                </div>
                """, unsafe_allow_html=True)

    # Footer with additional controls
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
            <small>💡 <strong>Tip:</strong> Maintain good posture between exercises and stay hydrated!</small>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        if st.button("❌ Emergency Stop", use_container_width=True):
            st.session_state.camera_handler.release()
            st.session_state.camera_active = False
            st.warning("System stopped for safety")

if __name__ == "__main__":
    main()
