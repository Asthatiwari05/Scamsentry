
import streamlit as st

st.set_page_config(
    page_title="ScamSentry | Scam Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {
        max-width: 1240px;
        padding-top: 1.8rem;
        padding-bottom: 3rem;
    }

   /* Hide Streamlit's automatic page list.
   Keep our custom product navigation below. */
[data-testid="stSidebarNav"] {
    display: none;
}

[data-testid="stSidebar"] {
    border-right: 1px solid rgba(128, 128, 128, 0.18);
}

    .brand {
        font-size: 1.05rem;
        font-weight: 750;
        letter-spacing: -0.3px;
    }

    .eyebrow {
        color: #60A5FA;
        font-size: 0.78rem;
        font-weight: 750;
        letter-spacing: 1.8px;
        text-transform: uppercase;
    }

    .hero {
        background: linear-gradient(120deg, #101D32 0%, #183858 65%, #174765 100%);
        border: 1px solid #294A6C;
        border-radius: 22px;
        padding: 2.5rem;
        margin: 0.6rem 0 1.5rem 0;
        color: white;
    }

    .hero h1 {
        color: white;
        font-size: clamp(2rem, 4vw, 3.25rem);
        line-height: 1.12;
        letter-spacing: -1.5px;
        margin: 0.8rem 0;
    }

    .hero p {
        color: #D5E4F5;
        font-size: 1.05rem;
        line-height: 1.7;
        max-width: 720px;
    }

    .hero-note {
        color: #A9C6E5;
        font-size: 0.82rem;
        margin-top: 1.1rem;
    }

    .section-label {
        font-size: 0.76rem;
        font-weight: 750;
        letter-spacing: 1.5px;
        text-transform: uppercase;
        color: #60A5FA;
        margin-bottom: 0.3rem;
    }

    .module-card {
        border: 1px solid rgba(128,128,128,0.27);
        border-radius: 17px;
        padding: 1.3rem;
        min-height: 205px;
        margin-bottom: 0.65rem;
        background: rgba(128,128,128,0.025);
    }

    .module-card h3 {
        font-size: 1.15rem;
        margin: 0.65rem 0 0.45rem 0;
    }

    .module-card p {
        font-size: 0.93rem;
        line-height: 1.6;
        color: #9CA3AF;
        margin-bottom: 0;
    }

    .module-icon {
        font-size: 1.65rem;
    }

    .footer {
        border-top: 1px solid rgba(128,128,128,0.2);
        margin-top: 2.2rem;
        padding-top: 1.1rem;
        color: #9CA3AF;
        font-size: 0.82rem;
    }

    div.stButton > button,
    div[data-testid="stPageLink"] a {
        border-radius: 10px;
    }

    @media (max-width: 700px) {
        .hero {
            padding: 1.5rem;
            border-radius: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown(
        '<div class="brand">🛡️ ScamSentry</div>',
        unsafe_allow_html=True,
    )
    st.caption("SCAM INTELLIGENCE PLATFORM")
    st.divider()

    st.markdown("**WORKSPACE**")
    st.page_link("app.py", label="Overview", icon="🏠")
    st.page_link(
        "pages/1_Message_Analyzer.py",
        label="Message Safety",
        icon="🔎",
    )

    st.markdown("**INTELLIGENCE**")
    st.page_link(
        "pages/mutation_dashboard.py",
        label="Mutation Intelligence",
        icon="🧬",
    )
    st.page_link(
        "pages/genome_dashboard.py",
        label="Scam Genome",
        icon="🕸️",
    )

    st.markdown("**PROTECTION**")
    st.page_link(
        "pages/early_warning.py",
        label="Safety Center",
        icon="🛡️",
    )

    st.divider()
    st.caption("ScamSentry AI · Research prototype")

st.markdown(
    """
    <div class="hero">
        <div class="eyebrow">YOUR SCAM INTELLIGENCE CENTER</div>
        <h1>Spot the scam.<br>Understand the risk.</h1>
        <p>
            Check suspicious messages, explore potential scam patterns,
            and find practical guidance to help you make safer decisions.
            One workspace. Clear explanations. Evidence before conclusions.
        </p>
        <div class="hero-note">
            Built for awareness, analysis and informed decision-making.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

left, right = st.columns([1.45, 1], gap="large")

with left:
    st.markdown(
        '<div class="section-label">START HERE</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Not sure about a message?")
    st.write(
        "Start with a message check. See the model's prediction, "
        "which words influenced it, and warning signs worth reviewing."
    )

    st.page_link(
        "pages/1_Message_Analyzer.py",
        label="Check a suspicious message  →",
        icon="🔎",
        use_container_width=True,
    )

with right:
    st.markdown(
        '<div class="section-label">OUR APPROACH</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Explain, don't just label")
    st.write(
        "ScamSentry combines text classification, similarity analysis "
        "and relationship discovery to help you investigate suspicious "
        "patterns instead of relying on a single score."
    )

st.divider()

st.markdown(
    '<div class="section-label">EXPLORE THE PLATFORM</div>',
    unsafe_allow_html=True,
)
st.header("Your security workspace")
st.write(
    "Choose the tool that matches what you want to investigate."
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-icon">🧬</div>
            <h3>Mutation Intelligence</h3>
            <p>
                Compare message wording and meaning to find potential
                variations that may deserve a closer look.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/mutation_dashboard.py",
        label="Explore message patterns  →",
        use_container_width=True,
    )

with col2:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-icon">🕸️</div>
            <h3>Scam Genome</h3>
            <p>
                Explore connections between cases using shared indicators
                and text similarity, with explanations for each link.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/genome_dashboard.py",
        label="Explore case relationships  →",
        use_container_width=True,
    )

col3, col4 = st.columns(2, gap="large")

with col3:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-icon">🛡️</div>
            <h3>Consumer Safety Center</h3>
            <p>
                Read practical safety guidance and review warning drafts
                with their evidence notes and review status.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/early_warning.py",
        label="Open the Safety Center  →",
        use_container_width=True,
    )

with col4:
    st.markdown(
        """
        <div class="module-card">
            <div class="module-icon">📊</div>
            <h3>Model Transparency</h3>
            <p>
                Inspect model evaluation, spam precision and recall,
                cross-validation results, and the limits of predictions.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.page_link(
        "pages/1_Message_Analyzer.py",
        label="View model performance  →",
        use_container_width=True,
    )

st.divider()

st.markdown(
    '<div class="section-label">HOW IT WORKS</div>',
    unsafe_allow_html=True,
)
st.header("From message to insight")

step1, step2, step3 = st.columns(3, gap="large")

with step1:
    st.markdown("### 01 · Analyze")
    st.write(
        "Review the message and the signals identified by the "
        "classification model."
    )

with step2:
    st.markdown("### 02 · Investigate")
    st.write(
        "Compare messages and examine potential links between "
        "cases using available evidence."
    )

with step3:
    st.markdown("### 03 · Act safely")
    st.write(
        "Use practical guidance and verify suspicious requests "
        "through official channels."
    )

st.warning(
    "ScamSentry is an experimental research prototype. Similarity "
    "scores and model predictions do not prove fraud. Demo cases and "
    "unverified warning drafts must not be treated as confirmed alerts."
)

st.markdown(
    """
    <div class="footer">
        <strong>ScamSentry AI</strong> · Scam intelligence and consumer awareness
        <br><br>
        Built to support safer decisions, not replace independent verification.
    </div>
    """,
    unsafe_allow_html=True,
)
