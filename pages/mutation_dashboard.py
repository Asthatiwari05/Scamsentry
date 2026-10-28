
import streamlit as st
import pandas as pd

from mutation import compare_messages, detect_mutation_candidates
st.markdown(
    """
    <style>
    [data-testid="stSidebarNav"] {
        display: none;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧬 ScamSentry AI")
st.subheader("Mutation Intelligence")
st.write(
    "Explore how suspicious messages change their wording "
    "while potentially preserving similar meanings."
)

st.info(
    "This is an experimental analysis tool. Similarity scores "
    "do not prove a message is fraudulent or that two messages "
    "come from the same scam campaign."
)

with st.sidebar:
    st.header("Analysis settings")
    threshold = st.slider(
        "Semantic similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Pairs at or above this score are flagged for review.",
    )

    st.caption(
        "Higher thresholds flag fewer, more similar pairs. "
        "The threshold has not yet been validated on labelled data."
    )

st.markdown("### Enter messages to compare")
st.write(
    "Add at least two messages. Use one message per box. "
    "You can paste real messages after removing personal information."
)

default_messages = [
    "Your parcel is delayed. Pay a delivery fee.",
    "Delivery failed. Submit the required charge to receive your package.",
    "Your electricity bill is ready. View your monthly usage.",
]

message_count = st.number_input(
    "Number of messages",
    min_value=2,
    max_value=10,
    value=3,
    step=1,
)

messages = {}

for i in range(int(message_count)):
    default_value = default_messages[i] if i < len(default_messages) else ""

    messages[f"MESSAGE-{i + 1}"] = st.text_area(
        f"Message {i + 1}",
        value=default_value,
        height=100,
        key=f"message_{i + 1}",
        placeholder="Paste a message here...",
    )

submitted = st.button(
    "Analyze message mutations",
    type="primary",
    use_container_width=True,
)

if submitted:
    cleaned_messages = {
        message_id: message.strip()
        for message_id, message in messages.items()
        if message.strip()
    }

    if len(cleaned_messages) < 2:
        st.error("Please enter at least two non-empty messages.")
    else:
        try:
            with st.spinner(
                "Generating embeddings and comparing messages..."
            ):
                comparisons = compare_messages(cleaned_messages)
                candidates = detect_mutation_candidates(
                    cleaned_messages,
                    threshold=threshold,
                )

            st.markdown("---")
            st.markdown("### Analysis overview")

            col1, col2, col3 = st.columns(3)

            col1.metric("Messages analyzed", len(cleaned_messages))
            col2.metric("Pairs compared", len(comparisons))
            col3.metric("Candidates flagged", len(candidates))

            if comparisons:
                st.markdown("### Message similarity ranking")

                df = pd.DataFrame(comparisons)
                df = df.rename(
                    columns={
                        "message_a": "Message A",
                        "message_b": "Message B",
                        "tfidf_similarity": "TF-IDF score",
                        "semantic_similarity": "Semantic score",
                    }
                )

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True,
                )

                st.markdown("### Similarity comparison")

                chart_df = df.copy()
                chart_df["Message Pair"] = (
                    chart_df["Message A"]
                    + " ↔ "
                    + chart_df["Message B"]
                )

                chart_df = chart_df.set_index("Message Pair")[
                    ["TF-IDF score", "Semantic score"]
                ]

                st.bar_chart(chart_df)

                st.markdown("### Mutation candidates")

                st.caption(
                    f"Candidates have a semantic similarity score "
                    f"of at least {threshold:.2f}."
                )

                if candidates:
                    for candidate in candidates:
                        with st.expander(
                            f"{candidate['message_a']} ↔ "
                            f"{candidate['message_b']} "
                            f"({candidate['semantic_similarity']:.2f})"
                        ):
                            st.write(
                                "**TF-IDF similarity:** "
                                f"{candidate['tfidf_similarity']:.4f}"
                            )
                            st.write(
                                "**Semantic similarity:** "
                                f"{candidate['semantic_similarity']:.4f}"
                            )
                            st.write(
                                "These messages may express similar "
                                "ideas. Review their contents and any "
                                "shared indicators before drawing conclusions."
                            )
                else:
                    st.success(
                        "No pairs reached the selected threshold."
                    )

                st.warning(
                    "A high similarity score is not a scam verdict. "
                    "Legitimate messages can also be similar, and "
                    "related scam messages can receive low scores."
                )

                csv_data = df.to_csv(index=False).encode("utf-8")

                st.download_button(
                    "Download comparison report (CSV)",
                    data=csv_data,
                    file_name="scamsentry_mutation_report.csv",
                    mime="text/csv",
                )

        except Exception as error:
            st.error(
                "Analysis failed. Check your internet connection "
                "on the first model download and review the terminal "
                "for details."
            )
            st.exception(error)

st.markdown("---")
st.caption(
    "ScamSentry AI • Experimental research prototype • "
    "Review results before taking action."
)
