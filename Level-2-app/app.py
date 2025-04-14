import streamlit as st
from langchain_core.prompts import PromptTemplate
from langchain_openai import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from io import StringIO

# Set page config
st.set_page_config(page_title="AI Long Text Summarizer", page_icon="📝", layout="centered")

# Custom CSS (optional)
st.markdown("""
    <style>
        .main { background-color: #f5f5f5; }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            border: none;
            border-radius: 8px;
            padding: 10px 24px;
        }
        .stTextInput>div>div>input {
            border-radius: 8px;
        }
    </style>
    """, unsafe_allow_html=True)

# App Title
st.title("📝 AI Long Text Summarizer")
st.markdown("Summarize large text files using OpenAI and LangChain")

# Intro Section
with st.expander("🔎 What does this app do?"):
    st.markdown("""
    - Upload a `.txt` file with long content.
    - It will split the text and summarize it using **OpenAI GPT**.
    - Great for research papers, blogs, long docs.
    """)

# YouTube Subscription Section
st.markdown("---")
st.markdown("📺 Follow [Amarnatha Gowda](https://www.youtube.com/@AMARRNATHHGOWDA) for more AI content!")
st.markdown("---")

# API Key Input
def get_openai_api_key():
    return st.text_input("🔐 OpenAI API Key", placeholder="Ex: sk-...", type="password")

openai_api_key = get_openai_api_key()

# File Uploader
uploaded_file = st.file_uploader("📄 Upload your text file", type="txt")

# Summary Output Area
st.markdown("### 📌 Summary Output:")

if uploaded_file:
    if not openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key to continue.")
        st.stop()

    try:
        content = uploaded_file.read().decode("utf-8")
        word_count = len(content.split())
        if word_count > 20000:
            st.error("❌ File too long. Max allowed: 20,000 words.")
            st.stop()

        # Split text
        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", "\n"], 
            chunk_size=5000, 
            chunk_overlap=350
        )
        docs = text_splitter.create_documents([content])

        # Load LLM
        llm = OpenAI(temperature=0, openai_api_key=openai_api_key)
        summarize_chain = load_summarize_chain(llm=llm, chain_type="map_reduce")

        with st.spinner("🧠 Summarizing... please wait"):
            summary = summarize_chain.run(docs)
        
        st.success("✅ Summary generated successfully!")
        st.text_area("Summary:", summary, height=300)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
