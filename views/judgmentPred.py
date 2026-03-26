import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from docx import Document as DocxDocument
from views.nvidia_llm import invoke, build_messages

# === ENV SETUP ===
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(_PROJECT_ROOT, ".env"), override=True)

PREDICTION_SYSTEM_PROMPT = """\
You are an expert AI Legal Analyst specializing in Indian law with deep knowledge of the Indian Penal Code, 
Code of Criminal Procedure, Indian Evidence Act, and Indian constitutional law.

Analyze case details and provide a structured legal analysis.

**Instructions:**
1. Carefully read and understand all case details.
2. Identify the nature of the case (criminal, civil, constitutional, etc.).
3. Determine the most likely verdict based on legal principles, precedents, and the strength of evidence described.
4. Provide a confidence level (High, Medium, Low) with justification.
5. List all relevant laws, sections, and acts that apply.
6. Reference any landmark cases or precedents that are relevant.

**Response Format (use exactly this structure):**

VERDICT: [Guilty/Not Guilty/Liable/Not Liable/Decree in favor of Plaintiff/Decree in favor of Defendant]

CONFIDENCE: [High/Medium/Low]

CASE TYPE: [Criminal/Civil/Constitutional/Family/Property/Corporate]

RATIONALE:
[Provide a clear, concise analysis of why this verdict is most likely. Reference specific facts from the case and how they relate to applicable legal principles. 3-5 paragraphs.]

RELEVANT LAWS:
- [Act/Section 1]: [Brief description of applicability]
- [Act/Section 2]: [Brief description of applicability]
- [Add more as needed]

PRECEDENT CASES:
- [Case Name 1] ([Year]): [Brief relevance]
- [Case Name 2] ([Year]): [Brief relevance]
- [Add more as needed]

KEY FACTORS:
- [Factor 1 that influenced the prediction]
- [Factor 2 that influenced the prediction]
- [Add more as needed]
"""

# === FUNCTIONS ===

def extract_text_from_file(file_obj, file_type):
    """Extract text from uploaded files (PDF, DOCX, or Image)."""
    text = ""
    if file_type == "pdf":
        pdf_reader = PdfReader(file_obj)
        text = "\n\n".join([
            page.extract_text().strip() 
            for page in pdf_reader.pages 
            if page.extract_text()
        ])
    elif file_type == "docx":
        doc = DocxDocument(file_obj)
        text = "\n\n".join([p.text.strip() for p in doc.paragraphs if p.text])
    elif file_type == "image":
        # For images, use OCR via NVIDIA model (text description)
        # Since NVIDIA text model can't process images directly,
        # we fall back to basic Pillow text extraction or inform user
        try:
            from PIL import Image
            img = Image.open(file_obj)
            # Use NVIDIA to describe what we can extract
            text = f"[Image file uploaded: {os.path.basename(str(file_obj))}. Please provide the case details as text for accurate analysis.]"
        except Exception:
            text = "[Could not process image. Please provide case details as text.]"
    else:
        raise ValueError("Unsupported file type.")
    return text


def predict_verdict(case_details):
    """
    Predict the verdict using NVIDIA AI for comprehensive case analysis.
    
    Args:
        case_details (str): The full text of the case.
    
    Returns:
        dict with verdict, confidence, rationale, laws, and precedents
    """
    user_prompt = f"Analyze this case and predict the verdict:\n\n{case_details}"
    messages = build_messages(PREDICTION_SYSTEM_PROMPT, user_message=user_prompt)
    
    analysis_text = invoke(messages, temperature=0.5, max_tokens=4096)
    
    # Extract verdict and confidence from the response
    verdict = "Analysis Complete"
    confidence = "Medium"
    
    for line in analysis_text.split('\n'):
        line_stripped = line.strip()
        if line_stripped.startswith('VERDICT:'):
            verdict = line_stripped.replace('VERDICT:', '').strip()
        elif line_stripped.startswith('CONFIDENCE:'):
            confidence = line_stripped.replace('CONFIDENCE:', '').strip()
    
    return {
        "verdict": verdict,
        "confidence": confidence,
        "analysis": analysis_text
    }


def analyze_case_text(case_text):
    """
    Analyze case text directly (without file upload).
    
    Args:
        case_text (str): Direct text input of case details.
    
    Returns:
        dict with verdict, confidence, and analysis
    """
    if not case_text or len(case_text.strip()) < 20:
        raise ValueError("Please provide more detailed case information (minimum 20 characters).")
    
    return predict_verdict(case_text)
