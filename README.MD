# Smart Interview Generator

A Streamlit app that uses OpenAI's GPT to generate technical and behavioral interview questions from a candidate's **Resume** or a **Job Description**.

---

## Features

- Upload `.txt`, `.pdf`, or `.docx` resumes or job descriptions
- Extracts and analyzes text
- Generates AI-driven interview questions using OpenAI API
- Handles input validation, file parsing, and error reporting

---

## Tech Stack

- [Streamlit](https://streamlit.io)
- [OpenAI GPT API](https://platform.openai.com/)
- Python (with `docx`, `PyPDF2`, etc.)

---

## How to Run Locally

```bash
git clone https://github.com/kajalkb/smart-interview-generator.git
cd smart-interview-generator
pip install -r requirements.txt
streamlit run app.py
