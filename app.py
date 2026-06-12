"""
FastAPI RAG chatbot for Mindful Minerals.

Usage:
    uvicorn app:app --reload --port 8000
    # or
    python app.py
"""

import os

import groq
import chromadb
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

load_dotenv()

app = FastAPI(title="Mindful Minerals Chatbot")

# --- lazy singletons -------------------------------------------------------

_model: SentenceTransformer | None = None
_collection: chromadb.Collection | None = None
_groq: groq.Groq | None = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_collection() -> chromadb.Collection:
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path="./chroma_db")
        try:
            _collection = client.get_collection("mindfulminerals")
        except Exception:
            raise HTTPException(
                status_code=503,
                detail="Knowledge base not ready. Run: python crawler.py && python embed.py",
            )
    return _collection


def get_groq() -> groq.Groq:
    global _groq
    if _groq is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(status_code=503, detail="GROQ_API_KEY not set in .env")
        _groq = groq.Groq(api_key=api_key)
    return _groq


# --- prompt ----------------------------------------------------------------

SYSTEM_PROMPT = """You are a friendly product assistant for Mindful Minerals, a premium skincare and wellness brand.

Guidelines:
- Answer questions about products, prices, ingredients, and promotions using the provided website context only.
- When a product shows both a regular price and a compare_at_price (or original price), it is ON SALE — highlight this clearly.
- If asked "what's on sale", list every discounted product found in the context with both prices.
- Format currency as $XX.XX.
- If the context doesn't contain an answer, say: "I don't have that detail right now — please check mindfulminerals.com for the latest information."
- Be concise, warm, and helpful. Bullet points are fine for lists."""


# --- routes ----------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    sources: list[str]


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("templates/index.html", encoding="utf-8") as f:
        return HTMLResponse(f.read())


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    model = get_model()
    collection = get_collection()
    client = get_groq()

    query_vec = model.encode(req.message).tolist()

    results = collection.query(
        query_embeddings=[query_vec],
        n_results=6,
    )

    chunks: list[str] = results["documents"][0]
    metas: list[dict] = results["metadatas"][0]
    sources = list({m["url"] for m in metas})

    context = "\n\n---\n\n".join(chunks)

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=1024,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Website context:\n{context}\n\n"
                    f"Customer question: {req.message}"
                ),
            },
        ],
    )

    return ChatResponse(
        response=response.choices[0].message.content,
        sources=sources,
    )


@app.get("/health")
async def health():
    try:
        col = get_collection()
        return {"status": "ok", "chunks_indexed": col.count()}
    except Exception as e:
        return {"status": "not_ready", "detail": str(e)}


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
