# app.py
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import os
import requests
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
import numpy as np

# ===== Load environment =====
if os.getenv("RAILWAY_ENVIRONMENT") is None:
    load_dotenv()
HF_API_KEY = os.getenv("HF_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "data-pdf")
PINECONE_HOST = os.getenv(
    "PINECONE_HOST",
    "https://data-pdf-mbi0n30.svc.aped-4627-b74a.pinecone.io"
)

# Candidate Hugging Face models
CANDIDATE_MODELS = [
    "thenlper/gte-small",             
    "BAAI/bge-small-en-v1.5",         
    "sentence-transformers/all-mpnet-base-v2"
]

# ===== Initialize clients =====
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX_NAME)
genai.configure(api_key=GEMINI_API_KEY)

HF_API_URL = "https://api-inference.huggingface.co/models/"
HF_HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
last_working_model = None

# ===== Embedding with fallback =====
def get_embedding(text: str):
    global last_working_model
    models_to_try = (
        [last_working_model] + [m for m in CANDIDATE_MODELS if m != last_working_model]
        if last_working_model else CANDIDATE_MODELS
    )
    for model_name in models_to_try:
        try:
            response = requests.post(
                HF_API_URL + model_name,
                headers=HF_HEADERS,
                json={"inputs": text},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            if isinstance(data, list):
                if isinstance(data[0], list):
                    emb = data[0]
                elif isinstance(data[0], dict) and "embedding" in data[0]:
                    emb = data[0]["embedding"]
                else:
                    emb = data
            elif isinstance(data, dict) and "embedding" in data:
                emb = data["embedding"]
            else:
                raise ValueError(f"Unexpected HF response format: {data}")
            last_working_model = model_name
            return np.array(emb).tolist()
        except Exception as e:
            continue
    raise RuntimeError("❌ All candidate Hugging Face embedding models failed.")

# ===== CV Retrieval =====
def retrieve_cv_chunks(query: str, top_k: int = 5) -> str:
    try:
        query_vec = get_embedding(query)
    except Exception as e:
        return ""
    results = index.query(vector=query_vec, top_k=top_k, include_metadata=True)
    matches = results.get("matches", [])
    if not matches:
        return ""
    chunks = []
    for i, m in enumerate(matches):
        text = m["metadata"].get("text", "")
        score = m.get("score", 0)
        chunks.append(f"[Chunk {i+1} | Score: {score:.4f}]\n{text.strip()}")
    return "\n\n".join(chunks)

# ===== Gemini Answering =====
def ask_gemini(user_input: str, context: str) -> str:
    try:
        model = genai.GenerativeModel("gemini-2.0-flash-exp")
        greetings = ["hi", "hello", "hey", "salam", "salaam"]
        if user_input.lower().strip() in greetings:
            return (
                "👋 Hi! I’m **Bilqees Shahid’s AI Assistant**.\n\n"
                "You can ask me anything about her — such as her **skills**, **education**, "
                "**projects**, or **experience**. How can I help you today?"
            )
        if not context.strip():
            return (
                "I couldn’t find relevant information in Bilqees Shahid’s CV "
                "for that question. Could you ask something else about her professional background?"
            )
        prompt = f"""
You are Bilqees Assistant — an AI that answers questions strictly using Bilqees Shahid’s CV.

Below are the 5 text chunks retrieved from her CV:
--------------------
{context}
--------------------

User question:
{user_input}

Instructions:
- Use only the information from the chunks above to answer.
- If the CV doesn’t mention something, explicitly say “not mentioned in the CV”.
- Answer in a clear, warm, and polite tone.
"""
        response = model.generate_content(prompt)
        return response.text.strip() if response.text else "⚠️ No response text from Gemini."
    except Exception as e:
        return f"⚠️ Gemini error: {e}"

# ===== FastAPI Setup =====
app = FastAPI(title="Bilqees Assistant API")

# Allow CORS for your frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # change to your frontend URL in production
    allow_methods=["*"],
    allow_headers=["*"]
)

class QueryRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QueryRequest):
    context = retrieve_cv_chunks(request.question)
    answer = ask_gemini(request.question, context)
    return {"answer": answer}

@app.get("/")
def root():
    return {"message": "Bilqees Shahid's AI Assistant API is running!"}
