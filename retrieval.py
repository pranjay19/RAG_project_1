
from pathlib import Path
import re

import torch
import chromadb

from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "ai_notes"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


# ==========================================
# 2. BASIC TEXT TOKENIZATION
# ==========================================

def tokenize(text):
    return re.findall(r"\b\w+\b", text.lower())


# ==========================================
# 3. CREATE OUR RETRIEVER
# ==========================================

class NotesRetriever:

    def __init__(self):

        # Connect to the database built by ingest.py
        if not DB_PATH.exists():
            raise FileNotFoundError(
                "ChromaDB not found. Run python ingest.py first."
            )

        client = chromadb.PersistentClient(path=str(DB_PATH))

        self.collection = client.get_collection(
            name=COLLECTION_NAME
        )

        # Load saved text chunks and metadata
        self.documents = []

        for offset in range(0, self.collection.count(), 200):

            data = self.collection.get(
                limit=200,
                offset=offset,
                include=["documents", "metadatas"]
            )

            for doc_id, text, metadata in zip(
                data["ids"],
                data["documents"],
                data["metadatas"]
            ):
                self.documents.append({
                    "id": doc_id,
                    "text": text,
                    "metadata": metadata or {}
                })

        if not self.documents:
            raise ValueError(
                "Database is empty. Run python ingest.py."
            )

        self.lookup = {
            doc["id"]: doc for doc in self.documents
        }

        print("Loaded chunks:", len(self.documents))

        # Build the BM25 keyword index
        tokenized_docs = [
            tokenize(doc["text"])
            for doc in self.documents
        ]

        self.bm25 = BM25Okapi(tokenized_docs)

        # Select hardware device (Apple MPS, NVIDIA CUDA, or standard CPU)
        if torch.backends.mps.is_available():
            device = "mps"
        elif torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"

        print(f"Loading embedding model on {device}...")

        self.embedder = SentenceTransformer(
            EMBEDDING_MODEL,
            device=device
        )

        print(f"Loading reranker on {device}...")

        self.reranker = CrossEncoder(
            RERANKER_MODEL,
            device=device
        )


    # ======================================
    # 4. BM25 KEYWORD SEARCH
    # ======================================

    def keyword_search(self, question, top_k=10):

        query_tokens = tokenize(question)

        scores = self.bm25.get_scores(query_tokens)

        ranked = sorted(
            zip(self.documents, scores),
            key=lambda item: float(item[1]),
            reverse=True
        )

        return [
            doc["id"]
            for doc, score in ranked[:top_k]
            if score > 0
        ]


    # ======================================
    # 5. VECTOR / SEMANTIC SEARCH
    # ======================================

    def vector_search(self, question, top_k=10):

        query_embedding = self.embedder.encode(
            [question],
            normalize_embeddings=True
        )

        results = self.collection.query(
            query_embeddings=query_embedding.tolist(),
            n_results=min(top_k, self.collection.count())
        )

        return results["ids"][0]


    # ======================================
    # 6. RECIPROCAL RANK FUSION (RRF)
    # ======================================

    def rrf(self, keyword_ids, vector_ids, k=60):

        scores = {}

        for ranking in [keyword_ids, vector_ids]:

            for rank, doc_id in enumerate(ranking, start=1):

                if doc_id not in scores:
                    scores[doc_id] = 0.0

                scores[doc_id] += 1 / (k + rank)

        return sorted(
            scores,
            key=lambda doc_id: scores[doc_id],
            reverse=True
        )


    # ======================================
    # 7. RERANK WITH CROSS-ENCODER
    # ======================================

    def rerank(self, question, candidate_ids, top_k=5):

        if not candidate_ids:
            return []

        # Compare question with each candidate passage
        pairs = [
            (question, self.lookup[doc_id]["text"])
            for doc_id in candidate_ids
        ]

        scores = self.reranker.predict(
            pairs,
            batch_size=8,
            show_progress_bar=False
        )

        results = []

        for doc_id, score in zip(candidate_ids, scores):

            doc = self.lookup[doc_id]

            results.append({
                "id": doc_id,
                "text": doc["text"],
                "page": doc["metadata"].get("page", "?"),
                "score": float(score)
            })

        results.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        return results[:top_k]


    # ======================================
    # 8. COMPLETE RETRIEVAL PIPELINE
    # ======================================

    def search(self, question, top_k=5):

        # Step A: Keyword retrieval
        keyword_ids = self.keyword_search(question, top_k=10)

        # Step B: Semantic retrieval
        vector_ids = self.vector_search(question, top_k=10)

        # Step C: Merge the two ranked lists
        combined_ids = self.rrf(keyword_ids, vector_ids)

        print("\nBM25 matches:", len(keyword_ids))
        print("Vector matches:", len(vector_ids))
        print("Unique hybrid candidates:", len(combined_ids))

        # Step D: Rerank candidates
        results = self.rerank(
            question,
            combined_ids,
            top_k=top_k
        )

        return results


# ==========================================
# 9. RUN THE RETRIEVER
# ==========================================

def main():

    retriever = NotesRetriever()

    while True:

        question = input(
            "\nAsk a question (or type 'exit'): "
        ).strip()

        if question.lower() in ("exit", "quit"):
            break

        if not question:
            continue

        results = retriever.search(question)

        for i, result in enumerate(results, start=1):

            print(f"\n========== RESULT {i} ==========")

            print("Page:", result["page"])

            print(
                "Reranker score:",
                round(result["score"], 4)
            )

            print("Text:")
            print(result["text"][:600])

        print("\nRetrieval completed!")


if __name__ == "__main__":
    main()
