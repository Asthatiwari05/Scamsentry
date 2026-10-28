
"""
Scam Genome: evidence-based relationship discovery.

This module compares scam cases, identifies shared indicators,
and constructs a graph of potentially related cases.
"""

import re
from itertools import combinations

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


INDICATOR_FIELDS = ("domains", "phone_numbers", "payment_ids")


def normalize_indicator(value):
    """Normalize an indicator for consistent comparison."""
    return str(value).strip().lower()


def normalize_indicators(values):
    """Convert a list of indicators into a normalized set."""
    if not values:
        return set()

    if isinstance(values, str):
        values = [values]

    return {
        normalize_indicator(value)
        for value in values
        if value and str(value).strip()
    }


def prepare_cases(cases):
    """Validate cases and prepare their text and indicators."""
    prepared = []
    seen_ids = set()

    for case in cases:
        case_id = str(case.get("id", "")).strip()

        if not case_id:
            raise ValueError("Every case must have a non-empty 'id'.")

        if case_id in seen_ids:
            raise ValueError(f"Duplicate case ID: {case_id}")

        seen_ids.add(case_id)

        item = dict(case)
        item["id"] = case_id
        item["text"] = str(case.get("text", "")).strip()

        for field in INDICATOR_FIELDS:
            item[field] = normalize_indicators(case.get(field, []))

        prepared.append(item)

    return prepared


def find_shared_indicators(case_a, case_b):
    """Return indicators shared by two cases."""
    shared = {}

    for field in INDICATOR_FIELDS:
        matches = case_a[field] & case_b[field]

        if matches:
            shared[field] = sorted(matches)

    return shared


def build_genome(cases, similarity_threshold=0.55):
    """
    Build a graph of potentially related scam cases.

    Cases can be connected through:
    1. Shared domains, phone numbers, or payment identifiers.
    2. Sufficiently similar message text.

    Text similarity alone is only a candidate relationship,
    not proof of a shared scam campaign.
    """
    if not 0 <= similarity_threshold <= 1:
        raise ValueError("similarity_threshold must be between 0 and 1.")

    prepared = prepare_cases(cases)
    graph = nx.Graph()

    # Add every case, including cases with no connections.
    for case in prepared:
        graph.add_node(
            case["id"],
            text=case["text"],
            timestamp=case.get("timestamp", ""),
            source=case.get("source", ""),
            label=case.get("label", "unknown"),
            domains=sorted(case["domains"]),
            phone_numbers=sorted(case["phone_numbers"]),
            payment_ids=sorted(case["payment_ids"]),
        )

    # Create TF-IDF vectors for message comparison.
    texts = [case["text"] for case in prepared]

    non_empty_indices = [
        index for index, text in enumerate(texts) if text
    ]

    similarities = {}

    if non_empty_indices:
        non_empty_texts = [texts[i] for i in non_empty_indices]

        vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            stop_words="english",
        )

        try:
            matrix = vectorizer.fit_transform(non_empty_texts)
            similarity_matrix = cosine_similarity(matrix)

            for row, original_i in enumerate(non_empty_indices):
                for col, original_j in enumerate(non_empty_indices):
                    if original_i < original_j:
                        similarities[(original_i, original_j)] = float(
                            similarity_matrix[row, col]
                        )

        except ValueError:
            # For example, all messages may be empty or contain
            # only words removed by preprocessing.
            pass

    # Compare each pair of cases.
    for i, j in combinations(range(len(prepared)), 2):
        case_a = prepared[i]
        case_b = prepared[j]

        shared = find_shared_indicators(case_a, case_b)
        text_score = similarities.get((i, j), 0.0)

        reasons = []

        for field, matches in shared.items():
            reasons.append(
                f"Shared {field}: {', '.join(matches)}"
            )

        if text_score >= similarity_threshold:
            reasons.append(
                f"Similar message text: {text_score:.2f}"
            )

        # Only create an edge when at least one signal exists.
        if reasons:
            graph.add_edge(
                case_a["id"],
                case_b["id"],
                text_similarity=round(text_score, 4),
                shared_indicators=shared,
                evidence=reasons,
                relationship_type=(
                    "shared_indicator_and_text"
                    if shared and text_score >= similarity_threshold
                    else "shared_indicator"
                    if shared
                    else "text_similarity_candidate"
                ),
            )

    return graph


def describe_relationship(graph, case_id_a, case_id_b):
    """Explain why two cases are connected."""
    if case_id_a not in graph or case_id_b not in graph:
        return "One or both case IDs were not found."

    if not graph.has_edge(case_id_a, case_id_b):
        return "No direct relationship was found by the current rules."

    edge = graph.edges[case_id_a, case_id_b]

    lines = [
        f"Case {case_id_a} and Case {case_id_b}",
        f"Relationship type: {edge['relationship_type']}",
        f"Text similarity: {edge['text_similarity']:.2f}",
        "Evidence:",
    ]

    lines.extend(f"- {reason}" for reason in edge["evidence"])

    lines.append(
        "Interpretation: This is a potential relationship, "
        "not proof that both cases share the same operator."
    )

    return "\n".join(lines)


def get_genome_summary(graph):
    """Return basic statistics about the discovered case network."""
    components = list(nx.connected_components(graph))

    connected_groups = [
        sorted(component)
        for component in components
        if len(component) > 1
    ]

    return {
        "total_cases": graph.number_of_nodes(),
        "total_relationships": graph.number_of_edges(),
        "isolated_cases": list(nx.isolates(graph)),
        "potential_groups": connected_groups,
    }
