"""
Generates embeddings for crawled_data.json and stores them in ChromaDB.

Usage:
    python embed.py
"""

import hashlib
import json
import sys

import chromadb
from fastembed import TextEmbedding

CRAWLED_FILE = "crawled_data.json"
CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "mindfulminerals"
CHUNK_SIZE = 400   # words
CHUNK_OVERLAP = 50  # words


def chunk_text(text: str) -> list[str]:
    words = text.split()
    if len(words) <= CHUNK_SIZE:
        return [text] if text.strip() else []
    chunks = []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + CHUNK_SIZE])
        if chunk.strip():
            chunks.append(chunk)
    return chunks


def stable_id(url: str, chunk_index: int, text: str) -> str:
    raw = f"{url}|{chunk_index}|{text[:60]}"
    return hashlib.md5(raw.encode()).hexdigest()


def embed_and_store() -> None:
    # Load crawled data
    try:
        with open(CRAWLED_FILE, encoding="utf-8") as f:
            documents = json.load(f)
    except FileNotFoundError:
        sys.exit(f"[ERROR] {CRAWLED_FILE} not found. Run crawler.py first.")

    print(f"Loaded {len(documents)} pages from {CRAWLED_FILE}")

    print("Loading embedding model...")
    model = TextEmbedding("BAAI/bge-small-en-v1.5")

    # Set up ChromaDB
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    try:
        client.delete_collection(COLLECTION_NAME)
        print("Cleared existing ChromaDB collection")
    except Exception:
        pass

    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Prepare chunks
    all_chunks: list[str] = []
    all_ids: list[str] = []
    all_metadatas: list[dict] = []

    for doc in documents:
        content = doc.get("content", "").strip()
        if not content:
            continue
        chunks = chunk_text(content)
        for j, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_ids.append(stable_id(doc["url"], j, chunk))
            all_metadatas.append({"url": doc["url"], "type": doc.get("type", "page")})

    if not all_chunks:
        sys.exit("[ERROR] No content to embed. Check crawled_data.json.")

    print(f"Embedding {len(all_chunks)} chunks...")

    # Encode in batches
    batch_size = 64
    all_embeddings: list[list[float]] = []
    for i in range(0, len(all_chunks), batch_size):
        batch = all_chunks[i : i + batch_size]
        vecs = [v.tolist() for v in model.embed(batch)]
        all_embeddings.extend(vecs)
        print(f"  {min(i + batch_size, len(all_chunks))}/{len(all_chunks)} encoded", end="\r")

    print()

    # Store in ChromaDB (max 5 000 per upsert)
    upsert_batch = 500
    for i in range(0, len(all_chunks), upsert_batch):
        collection.add(
            documents=all_chunks[i : i + upsert_batch],
            embeddings=all_embeddings[i : i + upsert_batch],
            ids=all_ids[i : i + upsert_batch],
            metadatas=all_metadatas[i : i + upsert_batch],
        )

    print(f"Done! {len(all_chunks)} chunks stored in {CHROMA_PATH}/{COLLECTION_NAME}")


if __name__ == "__main__":
    embed_and_store()
