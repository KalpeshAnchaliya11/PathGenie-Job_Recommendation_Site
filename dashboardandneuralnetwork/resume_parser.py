import fitz  # PyMuPDF
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

def extract_text_from_pdf(file_path):
    text = ""
    with fitz.open(file_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def extract_skills(text):
    skill_keywords = ['python', 'java', 'c++', 'flask', 'sql', 'excel', 'communication',
                      'machine learning', 'data analysis', 'django', 'tensorflow', 'keras', 'pandas',
                       'numpy', 'dsa', 'oops','javascript', 'react', 'html', 'bootstrap', 'git', 'github', 'dbms', 'os']
    extracted = []
    text = text.lower()
    for skill in skill_keywords:
        if skill in text:
            extracted.append(skill)
    return extracted
