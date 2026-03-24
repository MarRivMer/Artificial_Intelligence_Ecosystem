import logging
from transformers import logging as transformers_logging
import warnings
from dotenv import load_dotenv
import os
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss

# Load environment variables from .env file
load_dotenv()

# Set log levels
transformers_logging.set_verbosity_error()
warnings.filterwarnings("ignore")

# Retrieve OpenAI API key from environment
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OpenAI API key not found. Make sure your .env file has OPENAI_API_KEY set.")

# Modern OpenAI client
client = OpenAI(api_key=api_key)

# Read contents of Selected_Document.txt into text variable
with open("Selected_Document.txt", "r", encoding="utf-8") as file:
    text = file.read()

# -----------------------------
# Parameters
# -----------------------------
chunk_size = 500
chunk_overlap = 100
embedding_model_name = "sentence-transformers/all-distilroberta-v1"
cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
llm_model = "gpt-5.4-nano"   # change here if you want to swap models later

# Retrieve K with FAISS, then re-rank to M with a cross-encoder
top_k = 20
top_m = 8

# Split text into chunks using RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", " ", ""],
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
)
chunks = text_splitter.split_text(text)

if not chunks:
    raise ValueError("No text chunks were created from Selected_Document.txt.")

# Load model and encode chunks (bi-encoder)
embedder = SentenceTransformer(embedding_model_name)
embeddings = embedder.encode(chunks, show_progress_bar=False)
embeddings = np.array(embeddings, dtype="float32")

# Normalize for cosine-style similarity search
faiss.normalize_L2(embeddings)

# Initialize FAISS index and add embeddings
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatIP(dimension)
faiss_index.add(embeddings)

# Initialize the cross-encoder once
reranker = CrossEncoder(cross_encoder_name)

# -----------------------------
# Retrieval (bi-encoder + FAISS)
# -----------------------------
def retrieve_chunks(question: str, k: int = top_k):
    """
    Encode the question and search the FAISS index for top k similar chunks.
    """
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_arr = np.array(q_vec, dtype="float32")
    faiss.normalize_L2(q_arr)

    k = min(k, len(chunks))
    _, indices = faiss_index.search(q_arr, k)
    return [chunks[i] for i in indices[0] if i != -1]

# -----------------------------
# Re-ranking (cross-encoder)
# -----------------------------
def _dedupe_preserve_order(items):
    seen = set()
    out = []
    for it in items:
        key = " ".join(it.split())  # normalize whitespace
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out

def rerank_chunks(question: str, candidate_chunks, m: int = top_m):
    """
    Score (question, chunk) pairs with a cross-encoder and return the top-m chunks.
    """
    if not candidate_chunks:
        return []

    pairs = [(question, c) for c in candidate_chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(
        zip(candidate_chunks, scores),
        key=lambda x: float(x[1]),
        reverse=True
    )
    best = [c for c, _ in ranked[:min(m, len(ranked))]]
    return _dedupe_preserve_order(best)

# -----------------------------
# QA with LLM
# -----------------------------
def answer_question(question: str) -> str:
    candidates = retrieve_chunks(question, k=top_k)
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)
    context = "\n\n".join(relevant_chunks)

    if not context.strip():
        return "I couldn't find relevant context in the document."

    system_prompt = (
        "You are a helpful assistant answering questions ONLY from the provided context. "
        "Do not use outside knowledge. "
        "If the answer is not clearly in the context, say: "
        "'I don't know based on the provided context.'"
    )

    user_prompt = f"""Context:
{context}

Question:
{question}

Answer:
"""

    response = client.responses.create(
        model=llm_model,
        instructions=system_prompt,
        input=user_prompt,
        max_output_tokens=500,
    )

    return response.output_text.strip()

if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")
    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if not question:
            continue
        print("Answer:", answer_question(question))