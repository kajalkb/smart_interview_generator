import os
import streamlit as st
import pdfplumber
from docx import Document
from PyPDF2 import PdfReader
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

# --- Hugging Face Compatibility ---
os.environ["STREAMLIT_TELEMETRY_ENABLED"] = "0"  # Disable usage tracking (optional)

# --- Constants ---
SUPPORTED_EXTENSIONS = [".txt", ".pdf", ".docx"]
MAX_FILE_SIZE_MB = 5
MIN_TEXT_LENGTH = 100

# --- API Key Check ---
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    st.error("OPENAI_API_KEY not set. Add it in 'Settings > Secrets' on Hugging Face.")
    st.stop()

# --- LangChain Model Init ---
model = init_chat_model("gpt-4o-mini", model_provider="openai")

# --- Text Extraction Function ---
def extract_text_from_file(uploaded_file):
    try:
        ext = os.path.splitext(uploaded_file.name)[1].lower()

        if ext not in SUPPORTED_EXTENSIONS:
            raise ValueError("Unsupported file type. Please upload .txt, .pdf, or .docx")

        if ext == ".txt":
            return uploaded_file.read().decode("utf-8")

        elif ext == ".pdf":
            try:
                with pdfplumber.open(uploaded_file) as pdf:
                    return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
            except:
                reader = PdfReader(uploaded_file)
                return "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])

        elif ext == ".docx":
            doc = Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])

    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# --- Streamlit App ---
def main():
    st.title("📄 Smart Interview Generator")

    uploaded_file = st.file_uploader("Upload Resume or Job Description", type=["pdf", "docx", "txt"])

    if uploaded_file:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        if file_size_mb > MAX_FILE_SIZE_MB:
            st.warning("❗ File size exceeds 5 MB.")
            return

        extracted_text = extract_text_from_file(uploaded_file)

        if extracted_text:
            if len(extracted_text.strip()) < MIN_TEXT_LENGTH:
                st.warning("⚠️ File content too short or empty.")
            else:
                st.success("✅ Text extracted successfully.")
                st.text_area("Extracted Text", extracted_text, height=200)

                messages = [
                    SystemMessage(content=extracted_text),
                    HumanMessage(content="Generate customized interview questions."),
                ]

                try:
                    response = model.invoke(messages)
                    st.subheader("🧠 AI-Generated Questions")
                    st.text_area("Output", response.content, height=300)
                except Exception as e:
                    st.error(f"❌ API Error: {e}")

if __name__ == "__main__":
    main()
