# Contributing to Cervical Posture Detection System

We welcome contributions from developers, researchers, and healthcare professionals! This guide will help you get started.

## 🤝 Ways to Contribute

### 🐛 **Bug Reports**
- Found a bug? Please [open an issue](https://github.com/yourusername/cervical-posture-detection/issues)
- Include screenshots, error messages, and steps to reproduce
- Specify your browser, OS, and camera type

### 💡 **Feature Requests**
- Suggest new exercises or detection algorithms
- Propose UI/UX improvements
- Request clinical features or integrations

### 🔬 **Research Contributions**
- Submit new clinical algorithms based on research
- Improve detection accuracy with better thresholds
- Add validation studies or clinical data

### 📝 **Documentation**
- Improve README, comments, or guides
- Add tutorials or video demonstrations
- Translate to other languages

## 🚀 Development Setup

### Prerequisites
- Python 3.8 or higher
- Git
- Camera/webcam
- Code editor (VS Code recommended)

### Setup Steps
```bash
# 1. Fork the repository on GitHub
# 2. Clone your fork
git clone https://github.com/yourusername/cervical-posture-detection.git
cd cervical-posture-detection

# 3. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. Run the app
streamlit run app.py
```

### Development Guidelines

#### Code Style
- Follow PEP 8 style guidelines
- Use meaningful variable names
- Add docstrings to functions and classes
- Comment complex algorithms

#### Testing
```bash
# Test the app locally
streamlit run app.py

# Test different exercises
# Test with different cameras
# Test sensitivity settings
```

#### Commit Messages
Use clear, descriptive commit messages:
```bash
# Good examples:
git commit -m "Add: New lateral flexion detection algorithm"
git commit -m "Fix: Camera initialization on Windows"
git commit -m "Improve: UI responsiveness on mobile devices"
git commit -m "Update: Clinical thresholds based on Smith et al. 2023"
```

## 📋 Pull Request Process

### Before Submitting
1. **Test thoroughly** with different cameras and lighting
2. **Check code style** and add appropriate comments
3. **Update documentation** if needed
4. **Ensure no breaking changes** to existing features

### Submitting a PR
1. **Create feature branch**:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes** and commit
3. **Push to your fork**:
   ```bash
   git push origin feature/your-feature-name
   ```

4. **Open a Pull Request** on GitHub
5. **Fill out the PR template** with:
   - Description of changes
   - Testing performed
   - Screenshots/videos if UI changes
   - Clinical references if algorithm changes

### Review Process
- Core maintainers will review your PR
- May request changes or additional testing
- Once approved, your changes will be merged
- You'll be credited as a contributor!

## 🏥 Clinical Contributions

### Algorithm Development
If contributing new detection algorithms:
- **Base on peer-reviewed research**
- **Include clinical references** in code comments
- **Validate with test subjects** if possible
- **Document accuracy and limitations**

### Research Integration
When adding new clinical findings:
- **Cite original research papers**
- **Include DOI links when available**
- **Explain clinical significance**
- **Update documentation with new references**

## 🎯 Priority Areas

We especially welcome contributions in these areas:

### 🔬 **Algorithm Improvements**
- Better chin tuck detection accuracy
- Improved lighting adaptation
- Reduced false positives
- Multi-person detection

### 📱 **Mobile Optimization**
- Touch-friendly interface
- Portrait mode support
- Reduced bandwidth usage
- Offline capability

### 🌍 **Accessibility**
- Screen reader compatibility
- Keyboard navigation
- High contrast themes
- Multiple language support

### 📊 **Analytics & Reporting**
- Progress tracking over time
- Export functionality
- Clinical reporting formats
- Integration with health systems

## 📖 Documentation Standards

### Code Documentation
```python
def detect_chin_tuck(self, landmarks, face_landmarks, sensitivity=1.0):
    """Detect chin tuck (retraction) exercise.
    
    Based on clinical research: Jull et al. (2008) & Falla et al. (2007)
    Measurement: Craniovertebral angle + forward head posture assessment
    Clinical method: C7-tragus-horizontal angle measurement
    
    Args:
        landmarks: MediaPipe pose landmarks
        face_landmarks: MediaPipe face mesh landmarks  
        sensitivity (float): Detection sensitivity multiplier (0.5-2.0)
        
    Returns:
        dict: {
            "correct": bool,
            "message": str,
            "tips": list[str]
        }
        
    Clinical References:
        - Jull, G. et al. (2008). "Retraining cervical joint position sense"
        - Falla, D. et al. (2007). "Deep cervical flexor muscle dysfunction"
    """
```

### README Updates
When adding features, update:
- Feature list
- Usage instructions
- Technical details
- Clinical references

## 🧪 Testing Guidelines

### Manual Testing Checklist
- [ ] Camera starts/stops correctly
- [ ] All 5 exercises detect properly
- [ ] Sensitivity slider works
- [ ] UI is responsive
- [ ] No console errors
- [ ] Works in different browsers

### Clinical Testing
If possible, test with:
- Healthcare professionals
- Physical therapy patients
- Different age groups
- Various camera setups
- Different lighting conditions

## 🏆 Recognition

### Contributors
All contributors will be:
- Listed in README acknowledgments
- Credited in GitHub contributors
- Mentioned in release notes
- Invited to join core team (for significant contributions)

### Clinical Advisors
Healthcare professionals providing clinical guidance:
- Special recognition in documentation
- Clinical advisor status
- Input on algorithm development
- Research collaboration opportunities

## 📞 Getting Help

### Development Questions
- **GitHub Discussions**: For general questions
- **Issues**: For bugs or specific problems
- **Email**: maintainer@yourapp.com for sensitive topics

### Clinical Questions
- **Research Collaboration**: research@yourapp.com
- **Clinical Validation**: clinical@yourapp.com
- **Healthcare Partnerships**: partnerships@yourapp.com

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

**Thank you for contributing to better cervical health! 🙏**

Every contribution, no matter how small, helps make this tool more effective for physiotherapy and rehabilitation.
