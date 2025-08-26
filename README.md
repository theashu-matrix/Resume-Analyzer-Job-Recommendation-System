#Resume Analyzer & Job Recommendation System

This project is a Flask-based web application that analyzes resumes (PDF or TXT) to:

Extract essential candidate details (name, email, phone, skills, education)

Predict resume category (domain)

Recommend suitable job roles based on resume content

It uses machine learning models (Random Forest + TF-IDF) for text classification.

Features

Upload resumes in PDF or TXT format.

Automatic resume parsing: Extracts name, phone, email, education, and skills.

Resume categorization using ML.

Job role recommendation using ML.

Simple and interactive web interface built with Flask.

Supports LinkedIn, GitHub, and YouTube links for quick access.

Tech Stack

Backend: Python, Flask

Machine Learning: Scikit-learn (Random Forest, TF-IDF Vectorizer)

PDF Processing: PyPDF2

Frontend: HTML (Jinja Templates), CSS

Other: Regex for parsing contact details


