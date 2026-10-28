
"""Mutation Intelligence: lexical and semantic message comparison."""

from itertools import combinations

from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# A pretrained model that converts sentences into meaning-aware vectors.
# The model is loaded only when semantic comparison is requested.
_MODEL = None
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def get_semantic_model():
    """Load and cache the pretrained sentence-embedding model."""
    global _MODEL

    if _MODEL is None:
        _MODEL = SentenceTransformer(MODEL_NAME)

    return _MODEL


def compare_messages(messages):
    """
    Compare every pair using both TF-IDF and semantic similarity.

    Returns lexical and semantic scores for each pair.
    Similarity is a signal for investigation, not proof of a scam.
    """
    if len(messages) < 2:
        return []

    ids = list(messages.keys())
    texts = [str(messages[key]).strip() for key in ids]

    if any(not text for text in texts):
        raise ValueError("Messages must not be empty.")

    # Method 1: TF-IDF measures overlap in words and phrases.
    try:
        tfidf = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )
        tfidf_matrix = tfidf.fit_transform(texts)
        lexical_scores = cosine_similarity(tfidf_matrix)
    except ValueError:
        lexical_scores = [[0.0] * len(texts) for _ in texts]

    # Method 2: Sentence embeddings compare semantic meaning.
    model = get_semantic_model()
    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    semantic_scores = cosine_similarity(embeddings)

    results = []

    for i, j in combinations(range(len(ids)), 2):
        results.append({
            "message_a": ids[i],
            "message_b": ids[j],
            "tfidf_similarity": round(
                float(lexical_scores[i][j]), 4
            ),
            "semantic_similarity": round(
                float(semantic_scores[i][j]), 4
            ),
        })

    return sorted(
        results,
        key=lambda item: item["semantic_similarity"],
        reverse=True,
    )


def detect_mutation_candidates(messages, threshold=0.50):
    """
    Flag pairs whose semantic similarity reaches the chosen threshold.

    This is an experimental threshold, not a validated scam detector.
    Calibrate it using labelled examples before real-world use.
    """
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1.")

    comparisons = compare_messages(messages)

    return [
        result for result in comparisons
        if result["semantic_similarity"] >= threshold
    ]
