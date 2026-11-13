
from pathlib import Path

import pymupdf
import torch
import chromadb
from sentence_transformers import SentenceTransformer


# ==========================================
# 1. PROJECT CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent

PDF_PATH = BASE_DIR / "data" / "ai_engineering_handbook.pdf"
DB_PATH = BASE_DIR / "chroma_db"

COLLECTION_NAME = "ai_notes"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

CHUNK_SIZE = 140  # Words per chunk
BATCH_SIZE = 32


# ==========================================
# 2. EXTRACT TEXT AND CREATE CHUNKS
# ==========================================

def extract_and_chunk_pdf(pdf_path):

    chunks = []
    metadata = []
    ids = []

    with pymupdf.open(pdf_path) as pdf:

        print(f"Total PDF pages: {len(pdf)}")

        for page_number, page in enumerate(pdf, start=1):

            # Extract text from the current page
            text = page.get_text("text", sort=True)

            # Split the text into individual words
            words = text.split()

            if not words:
                print(f"Skipping page {page_number}: no text found")
                continue

            # Divide words into chunks
            for start in range(0, len(words), CHUNK_SIZE):

                chunk_words = words[start:start + CHUNK_SIZE]

                chunk_text = " ".join(chunk_words)

                chunk_number = start // CHUNK_SIZE + 1

                # Store text
                chunks.append(chunk_text)

                # Store source information
                metadata.append({
                    "page": page_number,
                    "chunk": chunk_number,
                    "source": pdf_path.name
                })

                # Unique ID for each chunk
                ids.append(
                    f"page_{page_number}_chunk_{chunk_number}"
                )

    return chunks, metadata, ids


# ==========================================
# 3. GENERATE EMBEDDINGS AND STORE IN CHROMA
# ==========================================

def build_database(chunks, metadata, ids):

    # Use Apple's GPU for model inference
    if not torch.backends.mps.is_available():
        raise RuntimeError(
            "Apple MPS GPU is unavailable. Check your PyTorch installation."
        )

    print("Loading embedding model on Apple MPS GPU...")

    model = SentenceTransformer(
        MODEL_NAME,
        device="mps"
    )

    print("Generating embeddings...")

    embeddings = model.encode(
        chunks,
        batch_size=BATCH_SIZE,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Embedding matrix shape:", embeddings.shape)

    # Create/open persistent vector database
    client = chromadb.PersistentClient(
        path=str(DB_PATH)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME
    )

    # Save in batches
    for start in range(0, len(chunks), 100):

        end = start + 100

        collection.upsert(
            ids=ids[start:end],
            documents=chunks[start:end],
            metadatas=metadata[start:end],
            embeddings=embeddings[start:end].tolist()
        )

    print("Successfully stored chunks in ChromaDB!")
    print("Total database records:", collection.count())

    return model, collection


# ==========================================
# 4. TEST A SIMPLE SEMANTIC SEARCH
# ==========================================

def test_search(model, collection):

    question = "What is tokenization?"

    print("\nTesting question:", question)

    # Embed the user's question
    query_embedding = model.encode(
        [question],
        normalize_embeddings=True
    )

    # Search for the three nearest chunks
    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(3, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    for i, (text, meta, distance) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0]
        ),
        start=1
    ):

        print(f"\nResult {i}")
        print("Page:", meta["page"])
        print("Distance:", round(distance, 4))
        print("Text:", text[:300], "...")


# ==========================================
# 5. RUN EVERYTHING
# ==========================================

def main():

    if not PDF_PATH.exists():
        raise FileNotFoundError(
            f"PDF not found: {PDF_PATH}"
        )

    print("Starting PDF ingestion...")

    chunks, metadata, ids = extract_and_chunk_pdf(PDF_PATH)

    if not chunks:
        raise ValueError(
            "No extractable text was found in the PDF."
        )

    print("Total chunks created:", len(chunks))

    print("\nExample chunk:")
    print(chunks[0][:400])

    model, collection = build_database(
        chunks,
        metadata,
        ids
    )

    test_search(model, collection)

    print("\nPhase 1 completed successfully!")


if __name__ == "__main__":
    main()
