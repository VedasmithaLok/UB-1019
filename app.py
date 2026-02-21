import streamlit as st
import os
from backend.model import XRayModel
from backend.report_generator import generate_medical_report, simplify_report
from backend.translator import translate_text
from backend.voice import generate_voice
from backend.utils.qr_generator import generate_qr
from backend.pdf_generator import generate_pdf_report

st.set_page_config(page_title="AI X-Ray Analysis", page_icon="🏥", layout="wide")

# Initialize session state for history
if 'history' not in st.session_state:
    st.session_state.history = []

st.title("🏥 AI X-Ray Detection with Multilingual Report")
st.markdown("---")

# Initialize model with error handling
@st.cache_resource
def load_model():
    try:
        return XRayModel()
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.info("Please ensure xray_model.pth exists in the project folder.")
        return None

model = load_model()

if model is None:
    st.stop()

col1, col2 = st.columns([1, 1])

with col1:
    uploaded_file = st.file_uploader("Upload X-ray Image", type=["jpg", "png", "jpeg"])

with col2:
    language_map = {
        "English": "en",
        "Hindi": "hi",
        "Tamil": "ta",
        "Telugu": "te",
        "Kannada": "kn"
    }
    language_name = st.selectbox("Select Language", list(language_map.keys()))
    language = language_map[language_name]

if uploaded_file:
    try:
        file_path = "temp_image.png"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.read())

        with st.spinner("Analyzing X-ray..."):
            disease, confidence, heatmap_path = model.predict(file_path)
            report, severity = generate_medical_report(disease, confidence)
            simple_report = simplify_report(disease)
            
            # Add to history
            from datetime import datetime
            st.session_state.history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %I:%M %p'),
                'disease': disease,
                'confidence': confidence,
                'severity': severity
            })

        # Display results
        st.markdown("---")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📊 Analysis Results")
            st.markdown(f"**Prediction:** {disease}")
            
            color = "green"
            if severity == "High":
                color = "red"
            elif severity == "Moderate":
                color = "orange"
            
            st.markdown(f"<h4 style='color:{color}'>Severity: {severity}</h4>", unsafe_allow_html=True)
            st.markdown(f"**Confidence:** {round(confidence*100, 2)}%")
            
            st.subheader("🔬 Analysis Heatmap")
            if os.path.exists(heatmap_path):
                st.image(heatmap_path, caption="Red areas = High attention regions", use_container_width=True)
        
        with col2:
            st.subheader("📋 Medical Report")
            st.text(report)
            
            if language != "en":
                with st.spinner("Translating..."):
                    translated_report = translate_text(report, language)
                    translated_simple = translate_text(simple_report, language)
                
                st.subheader(f"🌐 Translated Report ({language_name})")
                st.text(translated_report)
                
                st.subheader("👤 Patient Friendly Explanation")
                st.text(translated_simple)
                
                with st.spinner("Generating voice..."):
                    voice_file = generate_voice(translated_simple, language)
                    if os.path.exists(voice_file):
                        st.audio(voice_file)
            else:
                st.subheader("👤 Patient Friendly Explanation")
                st.text(simple_report)
        
        st.markdown("---")
        st.subheader("📱 QR Code for Report Sharing")
        
        # Create comprehensive QR data
        from datetime import datetime
        qr_data = f"""AI X-RAY ANALYSIS REPORT
        
Date: {datetime.now().strftime('%B %d, %Y %I:%M %p')}
Diagnosis: {disease}
Severity: {severity}
Confidence: {round(confidence*100, 2)}%

Recommendation: {'Immediate medical consultation advised' if disease.lower() == 'pneumonia' else 'No immediate intervention required'}

Note: AI-assisted preliminary report. Confirm with radiologist."""
        
        qr_file = generate_qr(qr_data)
        if os.path.exists(qr_file):
            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(qr_file, width=200)
            with col2:
                st.info("📱 Scan this QR code to:")
                st.markdown("""
                - Save diagnosis on your phone
                - Share with your doctor
                - Store in health apps
                - Send to family members
                """)
        
        # PDF Download Button
        st.markdown("---")
        st.subheader("📄 Download Report")
        
        pdf_file = generate_pdf_report(disease, confidence, severity, report, heatmap_path)
        
        if os.path.exists(pdf_file):
            with open(pdf_file, "rb") as f:
                pdf_data = f.read()
            
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_data,
                file_name=f"XRay_Report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mime="application/pdf"
            )
        
        # Cleanup
        if os.path.exists(file_path):
            os.remove(file_path)
            
    except Exception as e:
        st.error(f"Error processing image: {e}")
        st.info("Please try uploading a different image.")

# History Sidebar
with st.sidebar:
    st.header("📜 Scan History")
    if st.session_state.history:
        for i, scan in enumerate(reversed(st.session_state.history), 1):
            with st.expander(f"Scan #{len(st.session_state.history) - i + 1} - {scan['timestamp']}"):
                st.write(f"**Result:** {scan['disease']}")
                st.write(f"**Confidence:** {round(scan['confidence']*100, 2)}%")
                st.write(f"**Severity:** {scan['severity']}")
        if st.button("🗑️ Clear History"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No scans yet")

st.markdown("---")
st.caption("⚠️ This is an AI-assisted tool. Always consult a medical professional for diagnosis.")
