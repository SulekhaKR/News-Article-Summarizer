import streamlit as st
from transformers import pipeline
from gtts import gTTS
import requests
from bs4 import BeautifulSoup
import PyPDF2
import tempfile

# ----------------- Helpers -----------------
def extract_text_from_url(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    return ' '.join([p.get_text() for p in soup.find_all('p')])

def extract_text_from_pdf(file):
    reader = PyPDF2.PdfReader(file)
    return ' '.join([page.extract_text() for page in reader.pages if page.extract_text()])

def get_summarizer(length):
    if length == 60:
        return pipeline("summarization", model="google/flan-t5-base", use_fast=False)
    elif length == 150:
        return pipeline("summarization", model="google/pegasus-cnn_dailymail", use_fast=False)
    elif length == 500:
        return pipeline("summarization", model="pszemraj/led-large-book-summary", use_fast=False)
    else:
        raise ValueError("Invalid summary length.")

def safe_generate_summary(text, length):
    summarizer = get_summarizer(length)

    # Set input chunk size depending on model
    if "long-t5" in summarizer.model.name_or_path.lower():
        max_input_words = 8000   # LongT5 can handle very long text
    else:
        max_input_words = 700    # T5 & BART ~1024 tokens (~700 words)

    words = text.split()
    chunks = []
    for i in range(0, len(words), max_input_words):
        chunk = " ".join(words[i:i + max_input_words])
        chunks.append(chunk)

    summaries = []
    for chunk in chunks:
        summary = summarizer(
            chunk,
            max_length=length,
            min_length=max(20, int(length * 0.8)),
            do_sample=False
        )[0]["summary_text"]
        summaries.append(summary.strip())

    return " ".join(summaries)

def generate_audio(summary_text):
    tts = gTTS(summary_text)
    temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(temp_audio.name)
    return temp_audio.name

# ----------------- Page Config -----------------
st.set_page_config(page_title="AI Summarizer", page_icon="📄", layout="wide")

# ----------------- Custom CSS -----------------
st.markdown("""
    <style>
        .stApp {
            background-image: linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)), 
                              url("https://images.unsplash.com/photo-1507842217343-583bb7270b66?auto=format&fit=crop&w=1500&q=80");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
            color: white !important;
        }
        .glass-card {
            background: rgba(0, 0, 0, 0.6);
            border-radius: 15px;
            padding: 20px;
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            box-shadow: 0px 4px 20px rgba(0,0,0,0.5);
            margin-bottom: 20px;
            color: #f5f5f5;
        }
        .stButton>button {
            background: linear-gradient(90deg, #ff6a00, #ee0979);
            color: white;
            border-radius: 12px;
            font-size: 16px;
            padding: 10px 25px;
            transition: 0.3s;
            border: none;
        }
        .stButton>button:hover {
            transform: scale(1.05);
            background: linear-gradient(90deg, #ee0979, #ff6a00);
        }
        .block-container { padding-top: 2rem; }
        .summary-box {
            max-height: 300px;
            overflow-y: auto;
            padding: 10px;
            border-radius: 10px;
            background: rgba(255,255,255,0.05);
        }
    </style>
""", unsafe_allow_html=True)

# ----------------- UI Layout -----------------
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
st.title("📄 AI-Powered Summarizer with Audio")
st.markdown("### ✨ Summarize text, articles, or PDFs into concise summaries & listen to them in audio form.")
st.markdown("</div>", unsafe_allow_html=True)

# Input section
st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
col1, col2 = st.columns(2)

with col1:
    input_type = st.radio("🔹 Choose Input Type", ["Text", "URL", "PDF Upload", "LongT5 Article Digest"])

with col2:
    summary_length = st.selectbox("📏 Select Summary Length", [60, 150, 500])

if input_type == "Text":
    user_text = st.text_area("✍️ Enter your text", height=150, placeholder="Paste or type text here...")
elif input_type == "URL":
    url = st.text_input("🌐 Enter URL", placeholder="https://example.com/article")
elif input_type == "PDF Upload":
    pdf_file = st.file_uploader("📂 Upload PDF", type=["pdf"])
elif input_type == "LongT5 Article Digest":
    digest_input = st.text_area("📰 Paste long article references", height=150, placeholder="Paste messy references here...")

st.markdown("</div>", unsafe_allow_html=True)

# Generate button
generate = st.button("🚀 Generate Summary")

# Output section
if generate:
    if input_type == "Text" and user_text:
        text = user_text
    elif input_type == "URL" and url:
        with st.spinner("🔎 Fetching text from URL..."):
            text = extract_text_from_url(url)
    elif input_type == "PDF Upload" and pdf_file:
        with st.spinner("📖 Reading PDF file..."):
            text = extract_text_from_pdf(pdf_file)
    elif input_type == "LongT5 Article Digest" and digest_input:
        text = "Summarize the newspaper article reference list by grouping them by source and ordering by date:\n" + digest_input
    else:
        st.error("⚠️ Please provide a valid input.")
        st.stop()

    with st.spinner("✨ Generating summary..."):
        summary = safe_generate_summary(text, summary_length)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("📝 Summary")
    st.markdown(f"<div class='summary-box'>{summary}</div>", unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

    with st.spinner("🎧 Generating audio..."):
        audio_path = generate_audio(summary)

    st.markdown("<div class='glass-card'>", unsafe_allow_html=True)
    st.subheader("🔊 Listen to Summary")
    st.audio(audio_path)
    st.download_button("⬇️ Download Audio", open(audio_path, "rb"), file_name="summary.mp3")
    st.markdown("</div>", unsafe_allow_html=True)
