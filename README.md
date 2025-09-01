# 🗂️ Notes Q&A with RAG (Ollama)

A lightweight **Retrieval-Augmented Generation (RAG)** app that lets you upload your personal notes (`.txt`, `.md`, `.pdf`) and ask natural language questions about them.

Powered by:

- **Sentence Transformers** for embeddings
- **FAISS** for vector search
- **Ollama** (`llama3` or any other local model) for generation
- **Streamlit** for the user interface

---

## 🚀 Features

- Upload `.txt`, `.md`, `.pdf` files directly in the app
- Automatic chunking & embedding storage with FAISS
- Ask questions in natural language
- Answers include **citations** from your notes
- 100% **local and private** – runs entirely on your machine with Ollama

---

## 🛠️ Installation

1. Clone this repository:

   ```bash
   git clone https://github.com/yourusername/notes-rag.git
   cd notes-rag
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

4. Install [Ollama](https://ollama.ai/download) and pull a model:
   ```bash
   ollama pull llama3
   ```

---

## ▶️ Usage

1. Start the Ollama server (leave it running):

   ```bash
   ollama serve
   ```

2. In another terminal, launch the Streamlit app:

   ```bash
   streamlit run app.py
   ```

3. Open the app in your browser:
   - Upload notes (`.txt`, `.md`, `.pdf`)
   - Click **Ingest Files**
   - Ask questions in natural language

---

## 📂 Project Structure

```
notes-rag/
├─ app.py              # Streamlit UI
├─ rag_core.py         # Core RAG logic (embeddings, FAISS, Ollama calls)
├─ ingest.py           # (Optional) manual ingestion script
├─ requirements.txt    # Dependencies
├─ README.md           # Documentation
├─ .gitignore          # Git ignore rules
├─ data/               # Uploaded notes
├─ storage/            # FAISS index + metadata
```

---

## 🧩 Example

Upload a note with:

```
Lasso regression adds an L1 penalty. Ridge regression adds an L2 penalty. Both are regularization techniques.
```

Ask:

```
What is the difference between Lasso and Ridge regression?
```

Answer:

```
Lasso regression adds an L1 penalty, while Ridge regression adds an L2 penalty. [1]
```

---

## 📌 Notes

- Currently set up for **Ollama only** (no OpenAI key required).
- Works offline with any local model available in Ollama (`llama3`, `mistral`, etc.).

---

## 🏗️ Roadmap

- [ ] Add hybrid search (semantic + keyword)
- [ ] Add session memory for follow-up questions
- [ ] Highlight matched text in context

---

## 📜 License

MIT License © 2025 Sadia Zaman
