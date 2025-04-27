import streamlit as st
import joblib
import numpy as np
import re
import fitz  # PyMuPDF
import PyPDF2

# Load model and vectorizer
model = joblib.load('resume_categorization_model.pkl')
vectorizer = joblib.load('resume_vectorizer.pkl')

# Function to clean text
def clean_text(text):
    text = re.sub('http\S+\s*', ' ', text)
    text = re.sub('RT|cc', ' ', text)
    text = re.sub('#\S+', '', text)
    text = re.sub('@\S+', ' ', text)
    text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\]^_`{|}~"""), ' ', text)
    text = re.sub(r'[^\x00-\x7f]', r' ', text)
    text = re.sub('\s+', ' ', text)
    return text.strip()

# Function to extract text from PDF (with fallback)
def extract_text_from_pdf(uploaded_file):
    # Try using PyMuPDF first
    try:
        pdf_doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
        text = ""
        for page in pdf_doc:
            text += page.get_text()
        if len(text.strip()) > 30:
            return text
    except:
        pass

    # If PyMuPDF fails or very little text, fallback to PyPDF2
    uploaded_file.seek(0)  # Reset file pointer
    reader = PyPDF2.PdfReader(uploaded_file)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# Custom CSS for animations and bright colors
st.markdown("""
    <style>
    .main {
        background: linear-gradient(to right, #00c6ff, #0072ff);
        color: white;
    }
    button {
        background: linear-gradient(45deg, #ff6ec4, #7873f5);
        border: none;
        color: white;
        padding: 12px 24px;
        font-size: 16px;
        border-radius: 12px;
        transition: all 0.4s ease;
        margin-top: 10px;
    }
    button:hover {
        transform: scale(1.1);
        background: linear-gradient(45deg, #42e695, #3bb2b8);
    }
    .stButton>button {
        width: 100%;
    }
    .title {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        animation: slidein 2s ease-out;
    }
    @keyframes slidein {
        from {
            margin-top: 100px;
            opacity: 0;
        }
        to {
            margin-top: 0px;
            opacity: 1;
        }
    }
    </style>
""", unsafe_allow_html=True)

# App title
st.markdown('<div class="title">Resume Categorization 🚀</div>', unsafe_allow_html=True)
st.markdown("### Upload your PDF resume below and find out your category!")

# File uploader
uploaded_file = st.file_uploader("Choose your Resume (PDF only)", type=["pdf"])

if uploaded_file is not None:
    if st.button("Submit Resume"):
        with st.spinner('Analyzing your resume...✨'):
            # Extract and clean text
            extracted_text = extract_text_from_pdf(uploaded_file)

            if not extracted_text or len(extracted_text.strip()) < 30:
                st.error("❌ Unable to extract enough text from this PDF. Please upload a clearer resume.")
            else:
                st.markdown("##### Extracted Resume Text Preview 👀:")
                st.code(extracted_text[:1000] + ("..." if len(extracted_text) > 1000 else ""))

                cleaned_text = clean_text(extracted_text)

                # Vectorize
                features = vectorizer.transform([cleaned_text])

                # Predict
                prediction = model.predict(features)

                # Display result
                st.success(f"🎯 **Predicted Category:** {prediction[0]}")

