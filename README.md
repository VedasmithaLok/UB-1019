# 🏥 AI X-Ray Detection with Multilingual Report

## Overview
An AI-powered chest X-ray analysis system that detects pneumonia and generates multilingual medical reports with voice output and visual heatmaps.

## Features
✅ **AI-Powered Detection** - Deep learning model for pneumonia classification
✅ **Grad-CAM Visualization** - Heatmap showing areas of concern
✅ **Multilingual Support** - Reports in Hindi, Tamil, Telugu, Kannada, English
✅ **Text-to-Speech** - Audio reports for accessibility
✅ **QR Code Generation** - Easy report sharing
✅ **Patient-Friendly Explanations** - Simplified medical terminology

## Tech Stack
- **Frontend**: Streamlit
- **Backend**: Python, PyTorch
- **Model**: ResNet18 / MobileNetV2
- **Translation**: Google Translate API
- **Voice**: gTTS (Google Text-to-Speech)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (optional - if not already trained)
python train_fast.py

# Run the application
streamlit run app.py
```

## Usage
1. Upload a chest X-ray image (JPG/PNG)
2. Select preferred language
3. View AI analysis with heatmap
4. Read medical report and patient-friendly explanation
5. Listen to audio report
6. Scan QR code for report sharing

## Model Performance
- **Training Dataset**: 5,216 images (balanced)
- **Validation Dataset**: 624 images
- **Classes**: Normal, Pneumonia
- **Architecture**: MobileNetV2 (fast) or ResNet18 (accurate)

## Project Structure
```
ai_xray_multilingual_project/
├── app.py                      # Main Streamlit app
├── backend/
│   ├── model.py               # AI model & prediction
│   ├── report_generator.py   # Medical report generation
│   ├── translator.py          # Translation service
│   ├── voice.py               # Text-to-speech
│   ├── gradcam.py            # Heatmap visualization
│   └── utils/
│       └── qr_generator.py   # QR code generation
├── data/                      # Training data
├── train_fast.py             # Model training script
└── requirements.txt          # Dependencies
```

## Future Enhancements
- Multi-disease detection (TB, COVID-19, etc.)
- PDF report generation
- Cloud deployment
- Mobile app integration
- Doctor consultation booking

## Disclaimer
⚠️ This is an AI-assisted diagnostic tool. Always consult a qualified medical professional for final diagnosis and treatment.

## License
MIT License

## Team
[Your Team Name]
