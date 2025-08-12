from flask import Flask, request, render_template
from PyPDF2 import PdfReader
import re
import pickle
import os

app = Flask(__name__)

# ================= Load models =================
rf_classifier_categorization = pickle.load(open('models/rf_classifier_categorization.pkl', 'rb'))
tfidf_vectorizer_categorization = pickle.load(open('models/tfidf_vectorizer_categorization.pkl', 'rb'))
rf_classifier_job_recommendation = pickle.load(open('models/rf_classifier_job_recommendation.pkl', 'rb'))
tfidf_vectorizer_job_recommendation = pickle.load(open('models/tfidf_vectorizer_job_recommendation.pkl', 'rb'))

# ================= Resume cleaning =================
def cleanResume(txt):
    cleanText = re.sub('http\\S+\\s', ' ', txt)
    cleanText = re.sub('RT|cc', ' ', cleanText)
    cleanText = re.sub('#\\S+\\s', ' ', cleanText)
    cleanText = re.sub('@\\S+', ' ', cleanText)
    cleanText = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"""), ' ', cleanText)
    cleanText = re.sub(r'[^\x00-\x7f]', ' ', cleanText)
    cleanText = re.sub('\\s+', ' ', cleanText)
    return cleanText.strip()

# ================= Predictions =================
def predict_category(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_categorization.transform([resume_text])
    return rf_classifier_categorization.predict(resume_tfidf)[0]

def job_recommendation(resume_text):
    resume_text = cleanResume(resume_text)
    resume_tfidf = tfidf_vectorizer_job_recommendation.transform([resume_text])
    return rf_classifier_job_recommendation.predict(resume_tfidf)[0]

# ================= PDF Reader =================
def pdf_to_text(file):
    reader = PdfReader(file)
    return " ".join(page.extract_text() or "" for page in reader.pages)

# ================= Resume parsing =================
def extract_contact_number_from_resume(text):
    match = re.search(r"\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b", text)
    return match.group() if match else None

def extract_email_from_resume(text):
    match = re.search(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", text)
    return match.group() if match else None

def extract_skills_from_resume(text):
    skills_list = ["Python", "Data Analysis", "Machine Learning", "SQL", "Java", "C++", "JavaScript", "HTML", "CSS", "React", "Angular"]
    skills = [skill for skill in skills_list if re.search(rf"\b{re.escape(skill)}\b", text, re.IGNORECASE)]
    return skills

def extract_education_from_resume(text):
    education_keywords = ["Computer Science", "Information Technology", "Software Engineering", "Data Science", "Mechanical Engineering", "Electrical Engineering", "Civil Engineering", "Business Administration", "Economics", "Finance"]
    education = [edu for edu in education_keywords if re.search(rf"(?i)\b{re.escape(edu)}\b", text)]
    return education

def extract_name_from_resume(text):
    match = re.search(r"(\b[A-Z][a-z]+\b)\s(\b[A-Z][a-z]+\b)", text)
    return match.group() if match else None

# ================= Routes =================
@app.route('/')
def resume():
    return render_template("resume.html",
                           linkedin_url="https://www.linkedin.com/feed/",
                           github_url="https://github.com/theashu-matrix",
                           youtube_url="https://www.youtube.com/")

@app.route('/pred', methods=['POST'])
def pred():
    if 'resume' in request.files:
        file = request.files['resume']
        filename = file.filename

        if filename.endswith('.pdf'):
            text = pdf_to_text(file)
        elif filename.endswith('.txt'):
            text = file.read().decode('utf-8')
        else:
            return render_template('resume.html', message="❌ Invalid file format. Please upload a PDF or TXT file.")

        predicted_category = predict_category(text)
        recommended_job = job_recommendation(text)
        phone = extract_contact_number_from_resume(text) or "Not Found"
        email = extract_email_from_resume(text) or "Not Found"
        extracted_skills = extract_skills_from_resume(text) or ["No skills detected"]
        extracted_education = extract_education_from_resume(text) or ["Not Found"]
        name = extract_name_from_resume(text) or "Not Found"

        return render_template('resume.html',
                               predicted_category=predicted_category,
                               recommended_job=recommended_job,
                               phone=phone,
                               name=name,
                               email=email,
                               extracted_skills=extracted_skills,
                               extracted_education=extracted_education,
                               linkedin_url="https://www.linkedin.com/feed/",
                               github_url="https://github.com/theashu-matrix",
                               youtube_url="https://www.youtube.com/")
    else:
        return render_template("resume.html", message="⚠ No resume file uploaded.")

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)
