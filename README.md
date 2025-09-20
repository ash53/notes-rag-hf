# 🗂️ Notes Q&A with RAG (Hugging Face)

A lightweight **Retrieval-Augmented Generation (RAG)** app that lets you upload your personal notes (`.txt`, `.md`, `.pdf`) and ask natural language questions about them.

Powered by:

- **Sentence Transformers** for embeddings
- **FAISS** for vector search
- **Hugging Face Inference API** (`facebook/bart-large-cnn`) for text generation
- **Streamlit** for the user interface

---

## 🚀 Features

- Upload `.txt`, `.md`, `.pdf` files directly in the app
- Automatic chunking & embedding storage with FAISS
- Ask questions in natural language
- Answers include **citations** from your notes
- Easy deployment on **Streamlit Cloud**

---

## 🛠️ Installation (Local)

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/notes-rag-hf.git
   cd notes-rag-hf
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # macOS/Linux
   .venv\Scripts\activate      # Windows
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Create a `.env` file with your Hugging Face token:

   ```ini
   HF_API_TOKEN=your_huggingface_token_here
   ```

---

## ▶️ Usage

1. Place your `.txt`, `.md`, or `.pdf` notes in the `data/` folder.
2. Build the FAISS index:

   ```bash
   python ingest.py
   ```

3. Launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

4. Open the app in your browser:
   - Upload notes
   - Click **Ingest Files**
   - Ask questions in natural language

---

## 🌐 Deploy on Streamlit Cloud

1. Push this repo to GitHub.
2. Go to [Streamlit Cloud](https://share.streamlit.io) → New app.
3. Select this repo, choose `app.py` as entry point.
4. In **App → Settings → Secrets**, add:

   ```toml
   HF_API_TOKEN="your_real_token_here"
   ```

5. Deploy 🎉

---

## 📂 Project Structure

```
notes-rag-hf/
├─ app.py              # Streamlit UI
├─ rag_core.py         # Core RAG logic (embeddings, FAISS, HF API calls)
├─ ingest.py           # Manual ingestion script
├─ requirements.txt    # Dependencies
├─ README.md           # Documentation
├─ .gitignore          # Git ignore rules
├─ .env.example        # Template for Hugging Face API key
├─ data/               # Uploaded notes (empty with .gitkeep)
├─ storage/            # FAISS index + metadata (empty with .gitkeep)
```

---

## 🧩 Example

Upload a note:

```
Lasso regression adds an L1 penalty. Ridge regression adds an L2 penalty.
```

Ask:

```
What is the difference between Lasso and Ridge regression?
```

Answer (summarized by BART):

```
Lasso regression uses an L1 penalty, while Ridge regression uses an L2 penalty.
```

---

## 📌 Notes

- Uses **Hugging Face Inference API** (`bart-large-cnn`) — stable and free.
- For larger models, switch to a Hugging Face Inference Endpoint or Groq/OpenAI.

---

## 📜 License

MIT License © 2025 Sadia Zaman
