import qrcode
from datetime import datetime

def generate_qr(data, filename="report_qr.png"):
    """
    Generate QR code with enhanced report information
    
    The QR code contains:
    - Diagnosis result
    - Severity level
    - Confidence score
    - Date and time of analysis
    - Recommendation summary
    
    When scanned, this provides instant access to key medical information
    that can be shared with doctors, stored in health apps, or sent to family.
    """
    
    # Create QR code with higher error correction for medical data
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_H,
        box_size=10,
        border=4,
    )
    
    qr.add_data(data)
    qr.make(fit=True)
    
    # Create image with better visibility
    img = qr.make_image(fill_color="black", back_color="white")
    img.save(filename)
    
    return filename
