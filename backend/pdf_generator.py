from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.units import inch
from datetime import datetime

def generate_pdf_report(disease, confidence, severity, report_text, heatmap_path, filename="xray_report.pdf"):
    """
    Generate professional PDF report for X-ray analysis
    """
    doc = SimpleDocTemplate(filename, pagesize=letter)
    story = []
    styles = getSampleStyleSheet()
    
    # Title
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        textColor=colors.HexColor('#1f77b4'),
        spaceAfter=30,
        alignment=1
    )
    story.append(Paragraph("AI X-RAY ANALYSIS REPORT", title_style))
    story.append(Spacer(1, 0.3*inch))
    
    # Patient Info Table
    date = datetime.now().strftime("%B %d, %Y")
    time = datetime.now().strftime("%I:%M %p")
    
    info_data = [
        ['Report Date:', date],
        ['Report Time:', time],
        ['Analysis Type:', 'Chest X-Ray (AI-Assisted)'],
        ['Diagnosis:', disease],
        ['Confidence:', f'{round(confidence*100, 2)}%'],
        ['Severity:', severity]
    ]
    
    info_table = Table(info_data, colWidths=[2*inch, 4*inch])
    info_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
        ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 10),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('GRID', (0, 0), (-1, -1), 1, colors.black)
    ]))
    story.append(info_table)
    story.append(Spacer(1, 0.3*inch))
    
    # Heatmap Image
    if heatmap_path and os.path.exists(heatmap_path):
        story.append(Paragraph("<b>Analysis Heatmap:</b>", styles['Heading2']))
        story.append(Spacer(1, 0.1*inch))
        img = Image(heatmap_path, width=3*inch, height=3*inch)
        story.append(img)
        story.append(Spacer(1, 0.2*inch))
    
    # Medical Report
    story.append(Paragraph("<b>Detailed Medical Report:</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1*inch))
    
    for line in report_text.split('\n'):
        if line.strip():
            story.append(Paragraph(line, styles['Normal']))
    
    story.append(Spacer(1, 0.3*inch))
    
    # Disclaimer
    disclaimer_style = ParagraphStyle(
        'Disclaimer',
        parent=styles['Normal'],
        fontSize=8,
        textColor=colors.red,
        alignment=1
    )
    story.append(Paragraph(
        "<b>DISCLAIMER:</b> This is an AI-assisted preliminary report. "
        "Final diagnosis must be confirmed by a board-certified radiologist.",
        disclaimer_style
    ))
    
    # Build PDF
    doc.build(story)
    return filename

import os
