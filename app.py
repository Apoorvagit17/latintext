import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import google.generativeai as genai
from difflib import SequenceMatcher

# ------------------------
# Load Your Fine-Tuned Model (safetensors)
# ------------------------
model_path = r"C:\Users\APOORVA\Downloads\reconstruction_model"  # where model and tokenizer are saved

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# ------------------------
# Gemini Setup
# ------------------------
import os
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


# ------------------------
# Helper: Reconstruct Text
# ------------------------
def reconstruct(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    output = model.generate(**inputs, max_new_tokens=50)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# ------------------------
# Helper: Translate via Gemini
# ------------------------
def translate_with_gemini(text):
    model_g = genai.GenerativeModel("gemini-2.0-flash")
    prompt = f"""You are a Latin language expert. Translate the following Latin sentence into English:

    Latin: {text}

    English:"""
    response = model_g.generate_content(prompt)
    return response.text.strip()

# ------------------------
# Helper: Highlight Insertions
# ------------------------
def highlight_insertions(original, reconstructed):
    matcher = SequenceMatcher(None, original, reconstructed)
    result = ""
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode == "equal":
            result += reconstructed[j1:j2]
        elif opcode == "insert":
            result += f"<span style='color:blue'><b>{reconstructed[j1:j2]}</b></span>"
        elif opcode == "replace":
            result += f"<span style='color:blue'><b>{reconstructed[j1:j2]}</b></span>"
        else:
            result += reconstructed[j1:j2]
    return result

# ------------------------
# Streamlit UI
# ------------------------
st.set_page_config(page_title="Latin Text Reconstructor", layout="centered")
st.title("📜 Latin + Cyrillic Damaged Text Reconstructor")

input_text = st.text_area("Enter Damaged Latin Text (with Cyrillic corruption):", height=150)

if st.button("🔄 Reconstruct and Translate"):
    if not input_text.strip():
        st.warning("Please enter some text.")
    else:
        with st.spinner("Reconstructing and translating..."):
            reconstructed = reconstruct(input_text)
            translation = translate_with_gemini(reconstructed)
            highlighted = highlight_insertions(input_text, reconstructed)

        st.markdown("---")
        st.subheader("🟩 Reconstructed Latin Text:")
        st.markdown(f"<div style='font-size:18px'>{highlighted}</div>", unsafe_allow_html=True)

        st.subheader("🟦 English Translation:")
        st.markdown(f"<div style='font-size:18px; color:green'><b>{translation}</b></div>", unsafe_allow_html=True)
