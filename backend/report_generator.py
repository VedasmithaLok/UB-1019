from datetime import datetime

def generate_medical_report(disease, confidence):
    # For Normal cases, severity is inverse of confidence
    if disease.lower() == "normal":
        if confidence > 0.7:
            severity = "Low"
        elif confidence > 0.4:
            severity = "Moderate"
        else:
            severity = "High"
    # For Pneumonia cases, severity is based on confidence
    else:
        if confidence > 0.7:
            severity = "High"
        elif confidence > 0.4:
            severity = "Moderate"
        else:
            severity = "Low"

    date = datetime.now().strftime("%B %d, %Y")
    time = datetime.now().strftime("%I:%M %p")
    
    if disease.lower() == "normal":
        report = f'''RADIOLOGY REPORT
{'='*50}

EXAMINATION: Chest X-Ray (Posteroanterior View)
DATE: {date}
TIME: {time}
MODALITY: Digital Radiography

CLINICAL INDICATION:
Routine chest screening / AI-assisted analysis

TECHNIQUE:
Single frontal view of the chest was obtained.

FINDINGS:
The lungs are clear bilaterally with no evidence of focal 
consolidation, pleural effusion, or pneumothorax.

The cardiac silhouette is within normal limits.
The mediastinal contours are unremarkable.
Bony structures appear intact.

IMPRESSION:
No acute cardiopulmonary abnormality detected.
Lungs appear clear and well-aerated.

AI CONFIDENCE: {round(confidence*100, 2)}%
SEVERITY ASSESSMENT: {severity}

RECOMMENDATION:
- No immediate intervention required
- Continue routine health monitoring
- Follow-up as clinically indicated

NOTE: This is an AI-assisted preliminary report.
Final interpretation should be confirmed by a board-certified
radiologist for clinical decision-making.
{'='*50}'''
    else:
        report = f'''RADIOLOGY REPORT
{'='*50}

EXAMINATION: Chest X-Ray (Posteroanterior View)
DATE: {date}
TIME: {time}
MODALITY: Digital Radiography

CLINICAL INDICATION:
Suspected pneumonia / AI-assisted analysis

TECHNIQUE:
Single frontal view of the chest was obtained.

FINDINGS:
There is evidence of airspace opacity in the lung fields
consistent with pneumonic consolidation.

The affected areas show increased density suggesting
active inflammatory process.

Cardiac silhouette size is within normal limits.
No pleural effusion or pneumothorax identified.

IMPRESSION:
Findings consistent with PNEUMONIA.
Airspace disease present in lung parenchyma.

AI CONFIDENCE: {round(confidence*100, 2)}%
SEVERITY ASSESSMENT: {severity}

RECOMMENDATION:
- Immediate clinical correlation advised
- Consider antibiotic therapy as per clinical protocol
- Follow-up chest X-ray in 7-10 days recommended
- Consult pulmonologist if symptoms persist

NOTE: This is an AI-assisted preliminary report.
Urgent clinical correlation and confirmation by a
board-certified radiologist is strongly recommended.
{'='*50}'''
    
    return report, severity


def simplify_report(disease):
    if disease.lower() == "normal":
        return """🎉 GOOD NEWS - Your Lungs Are Healthy!

What This Means:
✓ Your chest X-ray looks completely normal
✓ No signs of infection, fluid, or abnormalities detected
✓ Your lungs are clear and functioning well
✓ The air passages are open and healthy

What You Should Do:
• Continue your regular healthy lifestyle
• Maintain good hygiene practices
• Stay physically active
• Eat a balanced diet
• Get adequate sleep
• Avoid smoking and pollution

When to See a Doctor:
• If you develop breathing difficulties
• Persistent cough lasting more than 2 weeks
• Chest pain or discomfort
• Fever with respiratory symptoms

Remember: This AI analysis is a screening tool. For complete peace of mind, always consult your doctor for final confirmation."""
    
    return """⚠️ ATTENTION NEEDED - Possible Lung Infection Detected

What This Means:
• The X-ray shows white/cloudy areas in your lungs
• This suggests pneumonia (lung infection/inflammation)
• Your lungs have fluid or pus buildup in air sacs
• This is making it harder for oxygen to reach your blood

Common Symptoms You May Experience:
• Cough (with or without mucus/phlegm)
• Fever and chills
• Difficulty breathing or shortness of breath
• Chest pain when breathing or coughing
• Fatigue and weakness
• Loss of appetite

What You Should Do IMMEDIATELY:
1. Consult a doctor TODAY - Don't delay!
2. Get proper medical examination
3. Doctor will likely prescribe antibiotics
4. Take complete course of medicine as prescribed
5. Get plenty of rest and stay hydrated
6. Follow-up X-ray may be needed in 1-2 weeks

Treatment Usually Includes:
• Antibiotics (for bacterial pneumonia)
• Fever-reducing medications
• Cough medicine if needed
• Rest and increased fluid intake
• In severe cases: hospitalization may be required

Recovery Timeline:
• Mild cases: 1-3 weeks with treatment
• Moderate cases: 3-6 weeks
• You should start feeling better within 2-3 days of starting antibiotics

Important Reminders:
⚠️ This is a serious condition but TREATABLE
⚠️ Early treatment prevents complications
⚠️ Complete your full antibiotic course even if you feel better
⚠️ This AI report must be confirmed by a qualified doctor

Don't Panic - With proper medical care, most people recover completely from pneumonia!"""
