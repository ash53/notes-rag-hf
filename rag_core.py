# rag_core.py
import os, json, faiss, uuid
from pathlib import Path
from typing import List, Dict
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv
import requests

# Load environment variables
load_dotenv()

# Embedding model
EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Hugging Face API config
HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = "facebook/bart-large-cnn"  # summarization model

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    locator: str

# --------------------------
# Hugging Face API call
# --------------------------
def call_huggingface(text: str) -> str:
    if not HF_API_TOKEN:
        return "(HF API call failed: No API token found)"
    
    headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}
    payload = {"inputs": text}

    url = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
    print(f"📡 Calling HF: {url}")

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code != 200:
        return f"(HF API call failed: {response.status_code} {response.text})"

    try:
        output = response.json()
        if isinstance(output, list) and "summary_text" in output[0]:
            return output[0]["summary_text"].strip()
        elif isinstance(output, dict) and "error" in output:
            return f"(HF error: {output['error']})"
        else:
            return str(output)
    except Exception as e:
        return f"(HF API parse error: {e})"

# --------------------------
# Main RAG pipeline
# --------------------------
def answer_query(query: str, storage_dir: Path):
    ctx = retrieve(query, storage_dir, top_k=2)  # limit context to 2 chunks for BART
    prompt = format_rag_prompt(query, ctx)
    ans = call_huggingface(prompt)
    return ans, ctx

# --------------------------
# Document loading & chunking
# --------------------------
def load_text_from_file(path: Path) -> str:
    if path.suffix.lower() in [".txt", ".md"]:
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join([page.extract_text() or "" for page in reader.pages])
    return ""

def simple_chunk(text: str, max_chars: int = 1200, overlap: int = 150):
    text = " ".join(text.split())
    chunks, i = [], 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        k = text.rfind(".", i, j)
        if k == -1 or k <= i + 200:
            k = j
        chunks.append(text[i:k].strip())
        i = max(k - overlap, k)
    return [c for c in chunks if c]

# --------------------------
# Embedding backend
# --------------------------
class EmbeddingBackend:
    def __init__(self, model_name: str = EMB_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        return self.model.encode(
            texts, show_progress_bar=False, normalize_embeddings=True
        )

# --------------------------
# Vector Store (FAISS)
# --------------------------
class VectorStore:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "faiss.index"
        self.meta_path = self.storage_dir / "meta.json"
        self.index = None
        self.meta = {}

    def load(self):
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        if self.meta_path.exists():
            self.meta = json.loads(self.meta_path.read_text(encoding="utf-8"))

    def save(self):
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        self.meta_path.write_text(
            json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    def build_new(self, embeddings, ids: List[str]):
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

# --------------------------
# Ingestion
# --------------------------
def ingest_folder(data_dir: Path, storage_dir: Path):
    emb = EmbeddingBackend()
    vs = VectorStore(storage_dir)
    vs.load()
    chunks = []
    files = sorted(
        [p for p in data_dir.glob("*") if p.suffix.lower() in [".txt", ".md", ".pdf"]]
    )
    for f in files:
        text = load_text_from_file(f)
        pieces = simple_chunk(text)
        for idx, piece in enumerate(pieces):
            chunks.append(Chunk(str(uuid.uuid4()), piece, f.name, f"chunk {idx+1}"))
    if not chunks:
        print("No chunks found.")
        return
    texts = [c.text for c in chunks]
    embeddings = emb.encode(texts)
    ids = [c.id for c in chunks]
    vs.build_new(embeddings, ids)
    vs.meta = {
        c.id: {"text": c.text, "source": c.source, "locator": c.locator} for c in chunks
    }
    vs.save()
    print(f"Ingested {len(chunks)} chunks from {len(files)} files.")

# --------------------------
# Retrieval
# --------------------------
def retrieve(query: str, storage_dir: Path, top_k=2):
    emb = EmbeddingBackend()
    vs = VectorStore(storage_dir)
    vs.load()
    if vs.index is None:
        raise RuntimeError("Index not found. Run ingest.py first.")
    qv = emb.encode([query])
    D, I = vs.index.search(qv, top_k)
    metas = list(vs.meta.values())
    results = []
    for rank, (idx, score) in enumerate(zip(I[0], D[0])):
        if idx < len(metas):
            m = metas[idx]
            results.append({"rank": rank + 1, "score": float(score), **m})
    return results

# --------------------------
# Prompt formatting
# --------------------------
def format_rag_prompt(query: str, contexts: List[Dict]) -> str:
    combined_context = " ".join([c['text'] for c in contexts])
    return f"Question: {query}\n\nContext: {combined_context}"
