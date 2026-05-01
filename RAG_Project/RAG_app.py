import logging
import warnings
import os

import numpy as np
import faiss
import openai

from dotenv import load_dotenv
from transformers import logging as hf_logging
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer, CrossEncoder


# 3.1 Suppress noisy logs
hf_logging.set_verbosity_error()
logging.getLogger("langchain_text_splitters").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# 3.2 ChatGPT API Credentials
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")


# 3.3 Parameters
chunk_size = 500
chunk_overlap = 50
model_name = "sentence-transformers/all-distilroberta-v1"
top_k = 20

cross_encoder_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
top_m = 8


# 3.4 Read document
with open("Selected_Document.txt", "r", encoding="utf-8") as file:
    text = file.read()


# 3.5 Split into Appropriately‑Sized Chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=chunk_size,
    chunk_overlap=chunk_overlap,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(text)


# 3.6 Embed and build FAISS index
embedder = SentenceTransformer(model_name)

embeddings = embedder.encode(chunks, show_progress_bar=False)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]

faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)


# 3.7 Retrieval function
def retrieve_chunks(question: str, k: int = top_k) -> list[str]:
    q_vec = embedder.encode([question], show_progress_bar=False)
    q_arr = np.array(q_vec).astype("float32")

    distances, I = faiss_index.search(q_arr, k)

    return [chunks[i] for i in I[0]]


# 3.8 Cross-encoder re-ranker
reranker = CrossEncoder(cross_encoder_name)


def dedupe_preserve_order(items: list[str]) -> list[str]:
    seen = set()
    result = []

    for item in items:
        normalized = " ".join(item.split())

        if normalized not in seen:
            seen.add(normalized)
            result.append(item)

    return result


def rerank_chunks(question: str, candidate_chunks: list[str], m: int = top_m) -> list[str]:
    pairs = [(question, chunk) for chunk in candidate_chunks]

    scores = reranker.predict(pairs)

    scored_chunks = list(zip(candidate_chunks, scores))
    scored_chunks.sort(key=lambda item: item[1], reverse=True)

    best_chunks = [chunk for chunk, score in scored_chunks[:m]]

    return dedupe_preserve_order(best_chunks)


# 3.9 Q&A with ChatGPT
def answer_question(question: str) -> str:
    """Retrieve relevant chunks, rerank them, and ask GPT to answer using only that context."""

    candidates = retrieve_chunks(question)
    relevant_chunks = rerank_chunks(question, candidates, m=top_m)

    context = "\n\n".join(relevant_chunks)

    system_prompt = (
        "You are a knowledgeable assistant that answers questions based on the provided context. "
        "If the answer is not in the context, say you don’t know."
    )

    user_prompt = f"""
Context:
{context}

Question: {question}

Answer:
"""

    resp = openai.chat.completions.create(
        model="gpt-5",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        max_completion_tokens=700
    )

    return resp.choices[0].message.content.strip()


# 3.10 Interactive loop
if __name__ == "__main__":
    print("Enter 'exit' or 'quit' to end.")

    while True:
        question = input("Your question: ")

        if question.lower() in ("exit", "quit"):
            break

        print("Answer:", answer_question(question))