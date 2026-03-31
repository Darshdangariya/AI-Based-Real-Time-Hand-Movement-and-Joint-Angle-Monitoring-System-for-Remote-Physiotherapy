# 🏥 Cervical Posture Detection System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://your-app-name.streamlit.app)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **AI-Powered Real-time Posture Monitoring for Cervical Physiotherapy**

A cutting-edge computer vision application that uses MediaPipe and clinical research to provide real-time feedback on cervical spine exercises. Perfect for physiotherapy clinics, home rehabilitation, and posture improvement programs.

## 🎯 Features

### 📹 **Real-Time Detection**
- Live camera feed with instant posture analysis
- Clinical-grade pose estimation using Google's MediaPipe
- Sub-second response time for immediate feedback

### 🏥 **Clinical Accuracy**
- **Research-Based Algorithms**: Built on peer-reviewed studies (Jull et al. 2008, Falla et al. 2007)
- **5 Exercise Types**: Cervical flexion, extension, lateral tilt, rotation, and chin tuck
- **Precise Measurements**: ROM calculations based on clinical standards
- **Confidence Scoring**: Weighted assessment system for movement quality

### 💻 **Professional Interface**
- Modern, responsive UI with real-time feedback
- Color-coded status indicators (Excellent/Good/Poor)
- Progress tracking and session statistics
- Customizable sensitivity settings

### 🎯 **Exercise Library**
| Exercise | Target ROM | Clinical Reference |
|----------|------------|-------------------|
| **Cervical Flexion** | 45-50° | Youdas et al. (1992) |
| **Cervical Extension** | 45-55° | Mannion et al. (2000) |
| **Lateral Tilt** | 40-45° | Bennett et al. (2002) |
| **Neck Rotation** | 80-90° | Dvorak et al. (1992) |
| **Chin Tuck** | 15-20mm | Jull et al. (2008) | 
  - Lateral Neck Tilt (Left and Right)
  - Neck Rotation (Turn head left/right)
  - Chin Tuck (Retract chin)
- **Visual feedback** with color-coded instructions
- **Web-based interface** accessible through any browser
- **No installation required** - deploy on cloud platforms

## 🚀 Quick Start

### Local Development

1. Clone the repository:
```bash
git clone <repository-url>
cd cervical-posture-detection
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Run the application:
```bash
streamlit run app.py
```

4. Open your browser and navigate to `http://localhost:8501`

### Cloud Deployment

#### Streamlit Cloud
1. Fork this repository
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Deploy using your GitHub repository

#### Hugging Face Spaces
1. Create a new Space on [Hugging Face](https://huggingface.co/spaces)
2. Upload the files or connect your GitHub repository
3. Select Streamlit as the SDK

## 📋 How to Use

1. **Camera Setup**: Allow camera access when prompted
2. **Exercise Selection**: Choose an exercise from the sidebar
3. **Position Yourself**: Sit straight, face the camera with good lighting
4. **Follow Feedback**: 
   - Green text = Correct posture
   - Red text = Needs adjustment
5. **Practice**: Hold correct positions and follow the real-time guidance

## 🔧 Technical Approach

### Computer Vision Pipeline
- **Pose Detection**: MediaPipe Pose for body landmark detection
- **Face Analysis**: MediaPipe Face Mesh for detailed facial landmarks
- **Real-time Processing**: Streamlit WebRTC for live video streaming

### Exercise Detection Algorithms

#### Cervical Flexion
- Calculates chin-to-chest distance using face landmarks
- Compares chin position relative to shoulder level
- Provides feedback on head tilt angle

#### Cervical Extension
- Monitors upward head tilt using nose position
- Measures vertical displacement from neutral position
- Ensures proper backward neck extension

#### Lateral Tilts
- Analyzes ear-to-ear angle for side tilts
- Uses geometric calculations for tilt detection
- Separate detection for left and right movements

#### Neck Rotations
- Tracks face orientation using nose offset
- Calculates horizontal rotation angles
- Detects left and right turning movements

#### Chin Tuck
- Measures horizontal chin retraction
- Compares chin alignment with shoulder position
- Detects proper chin-back movement

### Key Technologies
- **MediaPipe**: Google's machine learning framework for pose estimation
- **OpenCV**: Computer vision operations and image processing
- **Streamlit**: Web application framework
- **WebRTC**: Real-time communication for camera access

## 📊 Performance Considerations

- **Frame Rate**: Optimized for 30 FPS real-time processing
- **Accuracy**: Calibrated thresholds for reliable exercise detection
- **Latency**: Minimal delay between movement and feedback
- **Browser Compatibility**: Works on Chrome, Firefox, Safari, Edge

## 🎨 User Interface

- **Clean Design**: Intuitive interface with clear instructions
- **Real-time Feedback**: Instant visual cues for posture correction
- **Exercise Guidance**: Detailed descriptions and benefits for each exercise
- **Responsive Layout**: Works on desktop and mobile devices

## 🔒 Privacy & Security

- **Local Processing**: All video processing happens in the browser
- **No Data Storage**: Video frames are not saved or transmitted
- **Secure Connections**: HTTPS/WSS for encrypted communication

## 🏥 Clinical Applications

This app is designed to assist with:
- **Physical Therapy**: Guided exercise sessions
- **Posture Correction**: Office workers and students
- **Rehabilitation**: Post-injury neck exercises
- **Prevention**: Maintaining healthy neck posture

## 📈 Future Enhancements

- Exercise repetition counting
- Progress tracking and analytics
- Multiple user profiles
- Exercise difficulty levels
- Integration with fitness trackers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- MediaPipe team for pose estimation technology
- Streamlit for the web framework
- OpenCV community for computer vision tools

---

**Note**: This application is for educational and fitness purposes. Consult healthcare professionals for medical advice and proper exercise guidance.
