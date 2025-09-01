# rag_core.py
import os, json, faiss, uuid, subprocess
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from dotenv import load_dotenv

EMB_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@dataclass
class Chunk:
    id: str
    text: str
    source: str
    locator: str

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

class EmbeddingBackend:
    def __init__(self, model_name: str = EMB_MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: List[str]):
        return self.model.encode(texts, show_progress_bar=False, normalize_embeddings=True)

class VectorStore:
    def __init__(self, storage_dir: Path):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.storage_dir / "faiss.index"
        self.meta_path  = self.storage_dir / "meta.json"
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
        self.meta_path.write_text(json.dumps(self.meta, ensure_ascii=False, indent=2), encoding="utf-8")

    def build_new(self, embeddings, ids: List[str]):
        d = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(d)
        self.index.add(embeddings)

    def search(self, query_vec, k=5):
        D, I = self.index.search(query_vec, k)
        return list(zip(I[0].tolist(), D[0].tolist()))

def ingest_folder(data_dir: Path, storage_dir: Path):
    emb = EmbeddingBackend()
    vs = VectorStore(storage_dir)
    vs.load()
    chunks = []
    files = sorted([p for p in data_dir.glob("*") if p.suffix.lower() in [".txt", ".md", ".pdf"]])
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
    vs.meta = {c.id: {"text": c.text, "source": c.source, "locator": c.locator} for c in chunks}
    vs.save()
    print(f"Ingested {len(chunks)} chunks from {len(files)} files.")

def retrieve(query: str, storage_dir: Path, top_k=4):
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
            results.append({"rank": rank+1, "score": float(score), **m})
    return results

def format_rag_prompt(query: str, contexts: List[Dict]) -> str:
    bulleted = "\n\n".join([f"[{i+1}] Source: {c['source']} ({c['locator']})\n{c['text']}" for i, c in enumerate(contexts)])
    return f"""You are a helpful assistant. Answer strictly using the context. Cite like [1], [2].
Question:
{query}
Context:
{bulleted}
"""

def call_openai(prompt: str, model: str = "gpt-4o-mini") -> str:
    from openai import OpenAI
    load_dotenv()
    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    return resp.choices[0].message.content.strip()

def call_ollama(prompt: str, model: str = "llama3") -> str:
    proc = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return proc.stdout.decode("utf-8").strip()

def answer_query(query: str, storage_dir: Path, use_openai=False):
    ctx = retrieve(query, storage_dir, top_k=4)
    prompt = format_rag_prompt(query, ctx)
    if use_openai:
        try:
            ans = call_openai(prompt)
        except Exception as e:
            ans = f"(OpenAI call failed: {e})"
    else:
        try:
            ans = call_ollama(prompt)
        except Exception as e:
            ans = f"(Ollama call failed: {e})"
    return ans, ctx
