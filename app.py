# app.py
import streamlit as st
from pathlib import Path
from rag_core import ingest_folder, answer_query

st.set_page_config(page_title="Notes Q&A (RAG)", page_icon="🗂️", layout="centered")

DATA_DIR = Path("data")
STORAGE_DIR = Path("storage")
DATA_DIR.mkdir(exist_ok=True)
STORAGE_DIR.mkdir(exist_ok=True)

st.title("🗂️ Personal Notes Q&A (RAG)")
st.caption("Upload notes → ingest → ask questions with citations.")

# --- File uploader ---
uploaded_files = st.file_uploader(
    "Upload your notes (.txt, .md, .pdf)",
    type=["txt", "md", "pdf"],
    accept_multiple_files=True
)

if uploaded_files:
    for uploaded in uploaded_files:
        file_path = DATA_DIR / uploaded.name
        with open(file_path, "wb") as f:
            f.write(uploaded.getbuffer())
    st.success(f"✅ Uploaded {len(uploaded_files)} file(s).")
    

    if st.button("Ingest Files"):
        with st.spinner("Building index..."):
            ingest_folder(DATA_DIR, STORAGE_DIR)
        st.success("Index built! You can now ask questions below.")

# --- Q&A interface ---
query = st.text_input("Ask a question about your notes:")

# Either Enter (query filled) or pressing button will trigger
go = st.button("Answer")

if (go or query.strip()) and query.strip():
    with st.spinner("Generating answer..."):
        answer, ctx = answer_query(query.strip(), STORAGE_DIR)

    st.markdown("### Answer")
    st.write(answer)

    st.markdown("### Context & Citations")
    for c in ctx:
        with st.container(border=True):
            st.markdown(
                f"**[{c['rank']}] {c['source']} — {c['locator']}**  \n"
                f"Similarity: `{c['score']:.3f}`"
            )
            st.write(c["text"][:1200] + ("..." if len(c["text"]) > 1200 else ""))

