# Deployment Guide

## Streamlit Cloud Deployment

1. **Prepare Repository**
   - Ensure all files are committed to GitHub
   - Verify `requirements.txt` is complete
   - Test locally before deployment

2. **Deploy on Streamlit Cloud**
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Connect your GitHub account
   - Select repository: `cervical-posture-detection`
   - Main file path: `app.py`
   - Click "Deploy"

3. **Configuration**
   - Streamlit will automatically detect `requirements.txt`
   - Deployment typically takes 3-5 minutes
   - Check logs for any errors

## Hugging Face Spaces Deployment

1. **Create New Space**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose Streamlit SDK
   - Set space name: `cervical-posture-detection`

2. **Upload Files**
   - Upload `app.py`, `requirements.txt`, `README.md`
   - Ensure proper file structure
   - Add `.streamlit/config.toml` if needed

3. **Settings**
   - Hardware: CPU Basic (sufficient for this app)
   - Visibility: Public
   - License: MIT

## Local Testing Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run app locally
streamlit run app.py

# Run tests
python test_app.py

# Check code quality
pip install flake8
flake8 app.py
```

## Troubleshooting

### Camera Access Issues
- Ensure HTTPS connection for camera access
- Check browser permissions
- Try different browsers (Chrome recommended)

### Performance Issues
- Reduce video resolution if needed
- Check internet connection
- Monitor system resources

### Deployment Errors
- Verify all dependencies in requirements.txt
- Check Python version compatibility
- Review deployment logs

## Production Considerations

1. **Security**
   - All processing happens client-side
   - No video data is stored
   - HTTPS encryption for all communications

2. **Performance**
   - Optimized for real-time processing
   - Efficient landmark detection
   - Minimal latency design

3. **Scalability**
   - Stateless application design
   - Client-side processing reduces server load
   - WebRTC for direct peer connections
