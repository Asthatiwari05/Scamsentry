
import json

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from genome import build_genome, get_genome_summary, describe_relationship

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
st.subheader("Scam Genome — Relationship Intelligence")

st.write(
    "Explore potential connections between suspicious messages "
    "using shared indicators and message-text similarity."
)

st.warning(
    "Demo cases are fictional. A connection indicates a potential "
    "relationship, not proof of fraud or a shared scam operator."
)

# Fictional demonstration data only.
sample_cases = [
    {
        "id": "CASE-001",
        "text": "Your parcel is delayed. Pay a delivery fee using this link.",
        "domains": ["fake-delivery.example"],
        "phone_numbers": ["9000000001"],
        "payment_ids": [],
        "timestamp": "2026-10-20",
        "label": "synthetic_test",
    },
    {
        "id": "CASE-002",
        "text": "Delivery failed. Complete the fee payment to receive your parcel.",
        "domains": ["fake-delivery.example"],
        "phone_numbers": ["9000000001"],
        "payment_ids": [],
        "timestamp": "2026-10-21",
        "label": "synthetic_test",
    },
    {
        "id": "CASE-003",
        "text": "Your electricity bill is ready. View your monthly usage.",
        "domains": ["electricity-info.example"],
        "phone_numbers": ["9000000002"],
        "payment_ids": [],
        "timestamp": "2026-10-22",
        "label": "synthetic_test",
    },
    {
        "id": "CASE-004",
        "text": "Your parcel is waiting. Pay the delivery fee to receive it.",
        "domains": ["parcel-help.example"],
        "phone_numbers": ["9000000003"],
        "payment_ids": [],
        "timestamp": "2026-10-23",
        "label": "synthetic_test",
    },
]


def parse_list(value):
    """Read a JSON list or a comma-separated list."""
    if value is None or pd.isna(value) or not str(value).strip():
        return []

    value = str(value).strip()

    if value.startswith("["):
        parsed = json.loads(value)
        if not isinstance(parsed, list):
            raise ValueError("Indicator columns must contain lists.")
        return parsed

    return [item.strip() for item in value.split(",") if item.strip()]


def read_uploaded_cases(uploaded_file):
    """Convert an uploaded CSV into the format expected by genome.py."""
    frame = pd.read_csv(uploaded_file)

    required = {"id", "text"}
    missing = required - set(frame.columns)

    if missing:
        raise ValueError(
            "CSV must include these columns: id, text. "
            f"Missing: {', '.join(sorted(missing))}"
        )

    cases = []

    for _, row in frame.iterrows():
        case = {
            "id": str(row["id"]).strip(),
            "text": str(row["text"]).strip(),
            "timestamp": str(row.get("timestamp", "")),
            "label": str(row.get("label", "unknown")),
        }

        for field in ("domains", "phone_numbers", "payment_ids"):
            case[field] = parse_list(row.get(field, ""))

        cases.append(case)

    return cases


def draw_network(graph):
    """Draw cases and their discovered relationships."""
    positions = nx.spring_layout(graph, seed=42, k=1.1)

    edge_x, edge_y = [], []

    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]

        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edges = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line={"width": 1.5, "color": "#8993A4"},
        hoverinfo="none",
        name="Potential relationships",
    )

    node_x, node_y, node_text, node_colors = [], [], [], []

    for node in graph.nodes():
        x, y = positions[node]
        details = graph.nodes[node]

        node_x.append(x)
        node_y.append(y)
        node_text.append(
            f"Case: {node}<br>"
            f"Label: {details.get('label', 'unknown')}<br>"
            f"Message: {details.get('text', '')}"
        )
        node_colors.append(
            "#16A34A" if graph.degree(node) > 0 else "#64748B"
        )

    nodes = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=list(graph.nodes()),
        textposition="bottom center",
        hovertext=node_text,
        hoverinfo="text",
        marker={
            "size": 24,
            "color": node_colors,
            "line": {"width": 1, "color": "white"},
        },
        name="Cases",
    )

    fig = go.Figure(data=[edges, nodes])
    fig.update_layout(
        height=520,
        margin={"l": 10, "r": 10, "t": 20, "b": 10},
        showlegend=False,
        xaxis={"visible": False},
        yaxis={"visible": False},
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )

    return fig


with st.sidebar:
    st.header("Genome settings")

    similarity_threshold = st.slider(
        "Text similarity threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.55,
        step=0.05,
    )

    st.caption(
        "This threshold applies to TF-IDF text similarity. "
        "It is experimental and needs validation."
    )

    uploaded_file = st.file_uploader(
        "Upload case data (CSV)",
        type=["csv"],
        help=(
            "Required columns: id and text. Optional columns: "
            "domains, phone_numbers, payment_ids, timestamp, label. "
            "Indicator lists can be comma-separated or JSON arrays."
        ),
    )

try:
    if uploaded_file is not None:
        cases = read_uploaded_cases(uploaded_file)
        st.caption("Data source: uploaded CSV")
    else:
        cases = sample_cases
        st.caption("Data source: fictional demonstration cases")

    graph = build_genome(
        cases,
        similarity_threshold=similarity_threshold,
    )
    summary = get_genome_summary(graph)

    st.markdown("### Genome overview")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Cases", summary["total_cases"])
    col2.metric("Relationships", summary["total_relationships"])
    col3.metric("Isolated cases", len(summary["isolated_cases"]))
    col4.metric("Potential groups", len(summary["potential_groups"]))

    st.markdown("### Case relationship network")

    st.plotly_chart(
        draw_network(graph),
        use_container_width=True,
    )

    st.markdown("### Case register")

    case_rows = []

    for case_id, details in graph.nodes(data=True):
        case_rows.append({
            "Case ID": case_id,
            "Label": details["label"],
            "Date": details["timestamp"],
            "Domains": ", ".join(details["domains"]),
            "Phone numbers": ", ".join(details["phone_numbers"]),
            "Connections": graph.degree(case_id),
            "Message": details["text"],
        })

    case_frame = pd.DataFrame(case_rows)

    st.dataframe(
        case_frame,
        use_container_width=True,
        hide_index=True,
    )

    st.download_button(
        "Download case register (CSV)",
        data=case_frame.to_csv(index=False).encode("utf-8"),
        file_name="scamsentry_case_register.csv",
        mime="text/csv",
    )

    st.markdown("### Explain a relationship")

    connected_pairs = list(graph.edges())

    if connected_pairs:
        pair_labels = [
            f"{source} ↔ {target}"
            for source, target in connected_pairs
        ]

        selected_pair = st.selectbox(
            "Select a connected pair",
            pair_labels,
        )

        selected_index = pair_labels.index(selected_pair)
        source, target = connected_pairs[selected_index]

        st.code(describe_relationship(graph, source, target))

    else:
        st.info(
            "No relationships were found at this threshold. "
            "Try a lower threshold or check the case indicators."
        )

    st.caption(
        "Relationships are candidates for investigation. "
        "Shared indicators can be reused or spoofed, and text similarity "
        "can produce false matches."
    )

except Exception as error:
    st.error("Could not analyze the case data.")
    st.exception(error)
