
# import os
# import time
# import pdfplumber
# import requests
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from pinecone import Pinecone, ServerlessSpec
# from dotenv import load_dotenv

# if os.getenv("RAILWAY_ENVIRONMENT") is None:
#     load_dotenv()

# # CONFIG - put your keys in .env: HF_API_KEY, PINECONE_API_KEY
# HF_API_KEY = os.getenv("HF_API_KEY")
# PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# if not HF_API_KEY:
#     raise SystemExit("❌ HF_API_KEY not found in .env")
# if not PINECONE_API_KEY:
#     raise SystemExit("❌ PINECONE_API_KEY not found in .env")

# # Initialize Pinecone
# pc = Pinecone(api_key=PINECONE_API_KEY)

# # Define index name
# index_name = "data-pdf"

# # --- NEW LOGIC: Delete old index & create new one ---
# if index_name in [i.name for i in pc.list_indexes()]:
#     pc.delete_index(index_name)
#     print("✅ Deleted old index")

# pc.create_index(
#     name=index_name,
#     dimension=384,
#     metric="cosine",
#     spec=ServerlessSpec(cloud="aws", region="us-east-1")
# )
# print("✅ Created new index")

# # Get index instance
# index = pc.Index(index_name)

# # Candidate embedding models that are known to return embeddings via HF Inference API
# CANDIDATE_MODELS = [
#     "thenlper/gte-small",             # recommended, 384-dim
#     "BAAI/bge-small-en-v1.5",         # good alternative
#     "sentence-transformers/all-mpnet-base-v2"  # sometimes works via inference
# ]

# # PDF folder
# PDF_FOLDER = r"C:\Users\AHS PRINTERS\Desktop\PDF-files"

# # Helper: test model with one small request, return embedding or None + response text
# def test_model_and_get_embedding(model_id: str, text: str):
#     url = f"https://api-inference.huggingface.co/models/{model_id}"
#     headers = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}
#     payload = {"inputs": text}
#     try:
#         resp = requests.post(url, headers=headers, json=payload, timeout=30)
#     except Exception as e:
#         return None, f"Request error: {e}"

#     # Return raw text on non-200 for debug
#     if resp.status_code != 200:
#         return None, f"Status {resp.status_code}: {resp.text}"

#     # Try parsing
#     try:
#         data = resp.json()
#     except Exception as e:
#         return None, f"JSON parse error: {e} - raw: {resp.text}"

#     # Common embedding shapes
#     if isinstance(data, list):
#         first = data[0]
#         if isinstance(first, list) and all(isinstance(x, (int, float)) for x in first):
#             return first, None
#         if isinstance(first, dict) and "embedding" in first:
#             return first["embedding"], None
#         if all(isinstance(x, (int, float)) for x in data):
#             return data, None
#     if isinstance(data, dict) and "embedding" in data and isinstance(data["embedding"], list):
#         return data["embedding"], None

#     return None, f"Unexpected response format: {data}"

# # Find a working model
# print("🔎 Testing Hugging Face models to find a compatible embedding endpoint...")
# working_model = None
# sample_text = "Hello world"
# for m in CANDIDATE_MODELS:
#     emb, err = test_model_and_get_embedding(m, sample_text)
#     if emb:
#         working_model = m
#         print(f"✅ Working model found: {m}")
#         break
#     else:
#         print(f"❌ Model {m} failed: {err}")

# if not working_model:
#     print("\n❗ No embedding-capable model found. Please check your HF_API_KEY access rights.")
#     raise SystemExit("Stopping: no compatible HF embedding model available.")

# # Use the working model for the rest of the run
# HF_MODEL = working_model
# print(f"\nUsing model {HF_MODEL} for embeddings.\n")

# # Embedding function
# def get_embedding(text: str):
#     emb, err = test_model_and_get_embedding(HF_MODEL, text)
#     if err:
#         print("⚠️ Embedding request failed:", err)
#     return emb

# # PDF text extraction
# def extract_text_from_pdf(file_path: str) -> str:
#     text = ""
#     with pdfplumber.open(file_path) as pdf:
#         for page in pdf.pages:
#             page_text = page.extract_text()
#             if page_text:
#                 text += page_text + "\n"
#     return text.strip()

# # Main processing function
# def process_pdf(file_path: str):
#     print(f"\n📄 Processing: {os.path.basename(file_path)}...")
#     text = extract_text_from_pdf(file_path)
#     if not text:
#         print("⚠️ No text found in this PDF!")
#         return

#     splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
#     chunks = splitter.split_text(text)

#     vectors = []
#     for i, chunk in enumerate(chunks):
#         embedding = get_embedding(chunk)
#         if embedding:
#             vectors.append({
#                 "id": f"{os.path.basename(file_path)}_chunk_{i}",
#                 "values": embedding,
#                 "metadata": {"text": chunk}  # store preview only
#             })
#             print(f"  ✓ chunk {i} embedded (len {len(embedding)})")
#         else:
#             print(f"  ⚠️ Skipping chunk {i} (no embedding)")

#         time.sleep(1)

#     if vectors:
#         print(f"📤 Uploading {len(vectors)} chunks to Pinecone...")
#         index.upsert(vectors=vectors)
#         print("✅ Upload complete!")

# # Run the process
# if __name__ == "__main__":
#     if not os.path.exists(PDF_FOLDER):
#         raise SystemExit(f"PDF folder not found: {PDF_FOLDER}")

#     pdf_files = [f for f in os.listdir(PDF_FOLDER) if f.lower().endswith(".pdf")]
#     if not pdf_files:
#         raise SystemExit("No PDF files found in folder.")

#     for f in pdf_files:
#         process_pdf(os.path.join(PDF_FOLDER, f))
