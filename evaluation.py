
import json
import argparse
from pathlib import Path
from statistics import mean

from retrieval import NotesRetriever


# ==========================================
# 1. CONFIGURATION
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
LABELS_FILE = BASE_DIR / "evaluation_labels.json"

TOP_K = 5

# Questions based on our AI Engineering notes
QUESTIONS = [
    "What is tokenization?",
    "What is the difference between bag of words and TF-IDF?",
    "What is cosine similarity?",
    "Why do we use chunking in RAG?",
    "How does backpropagation work?"
]


# ==========================================
# 2. GET RESULTS FROM EACH SEARCH METHOD
# ==========================================

def get_rankings(retriever, question):

    # Search using BM25
    bm25_ids = retriever.keyword_search(
        question, top_k=12
    )

    # Search using embeddings
    vector_ids = retriever.vector_search(
        question, top_k=12
    )

    # Combine rankings using RRF
    hybrid_ids = retriever.rrf(
        bm25_ids, vector_ids
    )

    # Rerank every hybrid candidate
    reranked = retriever.rerank(
        question,
        hybrid_ids,
        top_k=len(hybrid_ids)
    )

    reranked_ids = [
        result["id"] for result in reranked
    ]

    return {
        "BM25": bm25_ids,
        "Vector": vector_ids,
        "Hybrid": hybrid_ids,
        "Hybrid + Rerank": reranked_ids
    }


# ==========================================
# 3. LOAD AND SAVE HUMAN LABELS
# ==========================================

def load_labels():

    if LABELS_FILE.exists():
        return json.loads(
            LABELS_FILE.read_text(encoding="utf-8")
        )

    return {}


def save_labels(labels):

    LABELS_FILE.write_text(
        json.dumps(labels, indent=2),
        encoding="utf-8"
    )


# ==========================================
# 4. HUMAN ANNOTATION
# ==========================================

def label_questions(retriever):

    labels = load_labels()

    for question in QUESTIONS:

        if question in labels:
            print(f"\nAlready labelled: {question}")
            continue

        rankings = get_rankings(
            retriever, question
        )

        # Show candidates in reranked order
        candidate_ids = rankings["Hybrid + Rerank"]

        print("\n" + "=" * 60)
        print("QUESTION:", question)
        print("=" * 60)

        for i, doc_id in enumerate(candidate_ids, start=1):

            doc = retriever.lookup[doc_id]
            page = doc["metadata"].get("page", "?")

            print(f"\n{i}. {doc_id} | Page {page}")
            print(doc["text"][:200], "...")

        print("\nRead the passages carefully.")
        print("Enter 'v 3' to view candidate 3 fully.")
        print("Enter '1,3,5' to mark relevant passages.")
        print("Press Enter to skip this question.")

        while True:

            choice = input("\nYour selection: ").strip()

            if not choice:
                print("Question skipped.")
                break

            # View the entire passage
            if choice.lower().startswith("v "):

                try:
                    number = int(choice.split()[1])
                    doc_id = candidate_ids[number - 1]
                    print("\n", retriever.lookup[doc_id]["text"])
                except (ValueError, IndexError):
                    print("Invalid candidate number.")

                continue

            try:
                selected_numbers = [
                    int(x.strip())
                    for x in choice.split(",")
                ]

                if any(
                    n < 1 or n > len(candidate_ids)
                    for n in selected_numbers
                ):
                    raise ValueError

                relevant_ids = [
                    candidate_ids[n - 1]
                    for n in selected_numbers
                ]

                labels[question] = list(
                    dict.fromkeys(relevant_ids)
                )

                save_labels(labels)

                print("Saved relevant passage IDs!")
                break

            except ValueError:
                print("Use numbers like 1,3,5 or v 2.")


# ==========================================
# 5. CALCULATE RETRIEVAL METRICS
# ==========================================

def calculate_metrics(ranked_ids, relevant_ids, k=5):

    relevant = set(relevant_ids)

    top_ids = ranked_ids[:k]

    # Count correct passages in top K
    correct = sum(
        doc_id in relevant
        for doc_id in top_ids
    )

    # Precision
    precision = correct / k

    # Recall
    recall = correct / len(relevant)

    # Hit rate for this individual question
    hit = 1.0 if correct > 0 else 0.0

    # Reciprocal rank of first relevant result
    rr = 0.0

    for rank, doc_id in enumerate(ranked_ids, start=1):

        if doc_id in relevant:
            rr = 1.0 / rank
            break

    return {
        "Precision@5": precision,
        "Recall@5": recall,
        "Hit@5": hit,
        "RR": rr
    }


# ==========================================
# 6. EVALUATE ALL RETRIEVAL METHODS
# ==========================================

def run_evaluation(retriever):

    labels = load_labels()

    if not labels:
        print(
            "No human labels found. "
            "Run: python evaluation.py --label"
        )
        return

    methods = [
        "BM25",
        "Vector",
        "Hybrid",
        "Hybrid + Rerank"
    ]

    scores = {
        method: [] for method in methods
    }

    for question, relevant_ids in labels.items():

        if not relevant_ids:
            continue

        rankings = get_rankings(
            retriever, question
        )

        print("\nQUESTION:", question)

        for method in methods:

            result = calculate_metrics(
                rankings[method],
                relevant_ids,
                TOP_K
            )

            scores[method].append(result)

            print(
                f"{method:17} "
                f"Recall@5={result['Recall@5']:.2f} "
                f"RR={result['RR']:.2f}"
            )

    # Calculate averages over all labelled questions
    print("\n" + "=" * 72)
    print("AVERAGE RETRIEVAL EVALUATION")
    print("=" * 72)

    print(
        f"{'Method':19} "
        f"{'P@5':>8} "
        f"{'R@5':>8} "
        f"{'Hit@5':>8} "
        f"{'MRR':>8}"
    )

    for method in methods:

        rows = scores[method]

        if not rows:
            continue

        p = mean(r["Precision@5"] for r in rows)
        recall = mean(r["Recall@5"] for r in rows)
        hit = mean(r["Hit@5"] for r in rows)
        mrr = mean(r["RR"] for r in rows)

        print(
            f"{method:19} "
            f"{p:8.3f} "
            f"{recall:8.3f} "
            f"{hit:8.3f} "
            f"{mrr:8.3f}"
        )


# ==========================================
# 7. MAIN PROGRAM
# ==========================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--label",
        action="store_true",
        help="Manually label relevant passages"
    )

    args = parser.parse_args()

    retriever = NotesRetriever()

    if args.label:
        label_questions(retriever)
    else:
        run_evaluation(retriever)


if __name__ == "__main__":
    main()
