
import streamlit as st
import pandas as pd
from datetime import date

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

st.title("🛡️ ScamSentry AI")
st.subheader("Consumer Early-Warning Intelligence Feed")

st.write(
    "Understand suspicious scam patterns, learn their warning signs, "
    "and discover practical steps to protect yourself."
)

st.info(
    "DEMO MODE: The initial warnings below are fictional examples. "
    "They are not live alerts or verified reports of current scams."
)

# Fictional examples for demonstrating the interface.
DEMO_WARNINGS = [
    {
        "id": "DEMO-001",
        "title": "Parcel delivery fee request",
        "category": "Delivery",
        "risk": "High caution",
        "status": "Fictional demonstration",
        "summary": (
            "A message claims that a parcel cannot be delivered "
            "until a small additional fee is paid."
        ),
        "signals": [
            "Unexpected payment request",
            "Pressure to act before delivery",
            "A link supplied in an unsolicited message",
        ],
        "action": (
            "Check your order through the courier's official website "
            "or app. Do not rely on the link in the message."
        ),
    },
    {
        "id": "DEMO-002",
        "title": "Electricity bill payment threat",
        "category": "Utilities",
        "risk": "High caution",
        "status": "Fictional demonstration",
        "summary": (
            "A message threatens service disconnection unless an "
            "urgent payment is made using a supplied link."
        ),
        "signals": [
            "Urgent disconnection threat",
            "Unexpected payment link",
            "Pressure to pay immediately",
        ],
        "action": (
            "Check your bill using your electricity provider's "
            "official app or website and contact its published support."
        ),
    },
    {
        "id": "DEMO-003",
        "title": "Unexpected account verification",
        "category": "Account security",
        "risk": "Caution",
        "status": "Fictional demonstration",
        "summary": (
            "A message asks you to verify an account using a link "
            "you were not expecting."
        ),
        "signals": [
            "Unexpected verification request",
            "A link to an unfamiliar domain",
            "A request for passwords or one-time codes",
        ],
        "action": (
            "Open the service through its official app or website. "
            "Never share passwords, PINs, or one-time verification codes."
        ),
    },
]


if "warning_drafts" not in st.session_state:
    st.session_state.warning_drafts = []


def show_warning(warning, is_draft=False):
    with st.container(border=True):
        st.markdown(f"### 🛡️ {warning['title']}")

        st.caption(
            f"ID: {warning['id']} · "
            f"Category: {warning['category']} · "
            f"Status: {warning['status']}"
        )

        st.write(warning["summary"])

        st.markdown("**Warning signs**")
        for signal in warning["signals"]:
            st.markdown(f"- {signal}")

        st.markdown("**What you should do**")
        st.write(warning["action"])

        if is_draft:
            st.warning(
                "This is a user-created draft. It has not been "
                "independently verified and must not be presented "
                "as a confirmed public alert."
            )


with st.sidebar:
    st.header("Feed filters")

    selected_category = st.selectbox(
        "Category",
        [
            "All categories",
            "Delivery",
            "Utilities",
            "Account security",
            "Other",
        ],
    )

    show_demo = st.checkbox(
        "Show fictional demo warnings",
        value=True,
    )

    show_drafts = st.checkbox(
        "Show my warning drafts",
        value=True,
    )

    st.divider()
    st.caption(
        "This prototype does not monitor the internet or "
        "automatically verify real-world scam reports."
    )


st.markdown("### Consumer warning feed")

feed = []

if show_demo:
    feed.extend(DEMO_WARNINGS)

if show_drafts:
    feed.extend(st.session_state.warning_drafts)

if selected_category != "All categories":
    feed = [
        warning for warning in feed
        if warning["category"] == selected_category
    ]

if not feed:
    st.info(
        "No warnings match these filters. Try another category "
        "or enable the demonstration warnings."
    )
else:
    st.caption(f"{len(feed)} warning(s) shown")

    for warning in feed:
        show_warning(
            warning,
            is_draft=warning["status"] == "Draft — pending review",
        )


st.divider()
st.markdown("### Create a warning draft")

st.write(
    "Use this form to turn an observed pattern into a draft for review. "
    "Do not include victims' names, phone numbers, passwords, or other "
    "personal information."
)

with st.form("warning_draft_form", clear_on_submit=True):
    title = st.text_input(
        "Warning title",
        placeholder="Example: Unexpected parcel payment requests",
    )

    category = st.selectbox(
        "Category",
        [
            "Delivery",
            "Utilities",
            "Account security",
            "Banking",
            "Job offers",
            "Other",
        ],
    )

    summary = st.text_area(
        "What is the suspected tactic?",
        placeholder="Describe the observed message or behaviour.",
    )

    signals_text = st.text_area(
        "Warning signs (one per line)",
        placeholder=(
            "Unexpected payment request\n"
            "Pressure to act immediately"
        ),
    )

    action = st.text_area(
        "Recommended consumer action",
        placeholder="Explain how a person can check safely.",
    )

    evidence = st.text_area(
        "Evidence notes and source",
        placeholder=(
            "Record where this pattern came from. "
            "Do not include private victim information."
        ),
    )

    submitted = st.form_submit_button(
        "Save warning draft",
        type="primary",
        use_container_width=True,
    )

    if submitted:
        if not all(
            field.strip()
            for field in [title, summary, signals_text, action, evidence]
        ):
            st.error("Please complete every field, including evidence notes.")
        else:
            draft = {
                "id": f"DRAFT-{len(st.session_state.warning_drafts) + 1:03d}",
                "title": title.strip(),
                "category": category,
                "risk": "Unassessed",
                "status": "Draft — pending review",
                "summary": summary.strip(),
                "signals": [
                    line.strip()
                    for line in signals_text.splitlines()
                    if line.strip()
                ],
                "action": action.strip(),
                "evidence": evidence.strip(),
                "created": str(date.today()),
            }

            st.session_state.warning_drafts.insert(0, draft)
            st.success(
                "Draft saved for this session. It is not a verified alert."
            )
            st.rerun()


st.divider()
st.markdown("### Safety reminder")

st.write(
    "Never pay or share sensitive information solely because a message "
    "pressures you to act quickly. Verify requests independently using "
    "official contact details. In India, suspected cybercrime can be "
    "reported through the official National Cyber Crime Reporting Portal "
    "at https://cybercrime.gov.in/."
)

st.caption(
    "ScamSentry AI · Early-warning prototype · "
    "A warning is not proof that a particular message or sender is fraudulent."
)
