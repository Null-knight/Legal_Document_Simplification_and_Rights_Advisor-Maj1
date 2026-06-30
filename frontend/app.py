from pathlib import Path
from collections import defaultdict
import html
import tempfile

import streamlit as st

from utils.api_client import ApiClient


st.set_page_config(page_title="Legal Advisor", page_icon="⚖", layout="wide")

ACCENTS = {
    "Blue": {"primary": "#2563eb", "secondary": "#dbeafe"},
    "Emerald": {"primary": "#059669", "secondary": "#d1fae5"},
    "Purple": {"primary": "#7c3aed", "secondary": "#ede9fe"},
    "Orange": {"primary": "#ea580c", "secondary": "#ffedd5"},
}

THEMES = {
    "Light": {
        "app_bg": "#f8fafc",
        "sidebar_bg": "#eef2f7",
        "surface": "#ffffff",
        "card": "#ffffff",
        "text": "#0f172a",
        "muted": "#64748b",
        "border": "#d9e2ef",
    },
}


def init_preferences() -> None:
    st.session_state.setdefault("accent", "Blue")


def apply_theme() -> None:
    theme = THEMES["Light"]
    accent = ACCENTS[st.session_state.accent]
    css = f"""
    <style>
    :root {{
      --app-bg: {theme["app_bg"]};
      --sidebar-bg: {theme["sidebar_bg"]};
      --surface-color: {theme["surface"]};
      --card-color: {theme["card"]};
      --text-color: {theme["text"]};
      --muted-color: {theme["muted"]};
      --border-color: {theme["border"]};
      --primary-color: {accent["primary"]};
      --secondary-color: {accent["secondary"]};
    }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)


def render_hero() -> None:
    st.markdown(
        """
        <section class="app-hero">
          <div class="hero-kicker">&#9878; Legal Intelligence Platform</div>
          <h1 class="hero-title">Legal Document Simplification & Rights Advisor</h1>
          <p class="hero-subtitle">
            Local-first legal AI workspace for document explanation, rule-backed rights reports,
            contract analysis, risk detection, and legal knowledge search.
          </p>
          <div class="hero-grid">
            <div class="hero-card"><strong>Document Simplification</strong><span>Convert legal text into plain language.</span></div>
            <div class="hero-card"><strong>Rights Advisory</strong><span>Generate rule-backed eligibility reports.</span></div>
            <div class="hero-card"><strong>Contract Analysis</strong><span>Extract clauses and compare versions.</span></div>
            <div class="hero-card"><strong>Risk Detection</strong><span>Flag liability, renewal, penalty, and exit risks.</span></div>
            <div class="hero-card"><strong>Legal Research</strong><span>Search a local structured knowledge base.</span></div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def pct(value: float | int | None) -> int:
    if value is None:
        return 0
    value = float(value)
    if value <= 1:
        value *= 100
    return max(0, min(100, round(value)))


def render_confidence(value: float | int | None, label: str = "Confidence") -> None:
    percent = pct(value)
    if percent >= 80:
        color = "#16a34a"
        status = "High Confidence"
    elif percent >= 55:
        color = "#f59e0b"
        status = "Medium Confidence"
    else:
        color = "#dc2626"
        status = "Low Confidence"
    st.markdown(
        f"""
        <div class="confidence-wrap">
          <div class="confidence-label"><span>{html.escape(label)}</span><span>{status} - {percent}%</span></div>
          <div class="confidence-bar"><div class="confidence-fill" style="width:{percent}%; background:{color};"></div></div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sources(sources: list[dict], title: str = "Sources") -> None:
    if not sources:
        return
    st.markdown(f"#### {title}")
    for source in sources:
        metadata = source.get("metadata", {}) or {}
        source_title = str(
            source.get("title")
            or metadata.get("title")
            or source.get("source")
            or metadata.get("filename")
            or source.get("filename")
            or "Source"
        )
        confidence = source.get("confidence") or source.get("similarity")
        confidence_text = f" - {pct(confidence)}%" if confidence is not None else ""
        excerpt = str(source.get("excerpt") or source.get("content") or "")[:420]
        with st.container(border=True):
            st.markdown(f"✓ **{source_title}**{confidence_text}")
            if excerpt:
                st.caption(excerpt)

def render_loading_steps(title: str, steps: list[str]) -> None:
    lines = "".join(f"<div>✓ {html.escape(step)}</div>" for step in steps)
    st.markdown(
        f"""
        <div class="loading-steps">
          <div>{html.escape(title)}</div>
          {lines}
        </div>
        """,
        unsafe_allow_html=True,
    )


def risk_visual(severity: str) -> tuple[str, int, str]:
    severity = severity.lower()
    if severity == "high":
        return "High Risk", 88, "#dc2626"
    if severity == "medium":
        return "Medium Risk", 58, "#f59e0b"
    return "Low Risk", 28, "#16a34a"


def render_risk_card(risk: dict) -> None:
    label, score, color = risk_visual(risk.get("severity", "Low"))
    risk_type = str(risk.get("risk_type", "risk")).replace("_", " ").title()
    st.markdown(
        f"""
        <div class="risk-card">
          <strong>{html.escape(label)} - {html.escape(risk_type)}</strong>
          <div class="risk-bar" style="margin:0.55rem 0;"><div class="risk-fill" style="width:{score}%; background:{color};"></div></div>
          <p>{html.escape(str(risk.get("explanation", "")))}</p>
          <div class="source-excerpt">{html.escape(str(risk.get("clause", "")))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_status_badges(client: ApiClient, online: bool) -> None:
    if not online:
        st.markdown(
            """
            <div class="status-grid">
              <div class="status-badge">● Backend Offline<span>Start FastAPI to enable features</span></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return
    try:
        status = client.system_status()
    except Exception:
        status = {}
    badges = [
        (
            "Ollama Connected" if status.get("ollama_connected") else "Ollama Offline",
            status.get("ollama_connected"),
            "Local LLM provider",
        ),
        (
            "ChromaDB Active" if status.get("chromadb_active") else "ChromaDB Offline",
            status.get("chromadb_active"),
            f"{status.get('chunks_indexed', 0)} vector chunks",
        ),
        (
            "Rights Engine Loaded" if status.get("rights_engine_loaded") else "Rights Engine Missing",
            status.get("rights_engine_loaded"),
            "Rule + evidence engine",
        ),
        (
            "Knowledge Base Ready" if status.get("knowledge_base_ready") else "Knowledge Base Empty",
            status.get("knowledge_base_ready"),
            f"{status.get('documents_indexed', 0)} documents indexed",
        ),
    ]
    html_badges = []
    for label, ok, subtitle in badges:
        icon = "●"
        status_class = "status-green" if ok else "status-yellow"
        html_badges.append(
            f'<div class="status-badge"><b class="{status_class}">{icon}</b> {html.escape(label)}<span>{html.escape(str(subtitle))}</span></div>'
        )
    st.markdown(f'<div class="status-grid">{"".join(html_badges)}</div>', unsafe_allow_html=True)


init_preferences()
apply_theme()

styles = Path(__file__).parent / "assets" / "styles.css"
if styles.exists():
    st.markdown(f"<style>{styles.read_text()}</style>", unsafe_allow_html=True)

st.sidebar.markdown(
    """
    <div class="sidebar-brand">
      <div class="sidebar-brand-title">&#9878; Legal Advisor</div>
      <div class="sidebar-brand-subtitle">Local legal intelligence</div>
    </div>
    """,
    unsafe_allow_html=True,
)
api_base = st.sidebar.text_input("Backend URL", value="http://127.0.0.1:8601")
client = ApiClient(api_base)
online = client.health()
st.sidebar.markdown(
    '<span class="status-ok">Backend online</span>' if online else '<span class="status-offline">Backend offline</span>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")
NAV_ITEMS = {
    "🏠 Dashboard": "Dashboard",
    "💬 Legal Chat": "Chat",
    "📄 Simplify Document": "Simplify",
    "⚖ Rights Advisor": "Rights Eligibility",
    "📊 Risk Analysis": "Risk Analyzer",
    "🔍 Compare Documents": "Compare",
    "🧩 Clause Extraction": "Clause Extraction",
    "📚 Knowledge Base": "Documents",
    "📈 Analytics": "Analytics",
    "❓ Rights Q&A": "Rights Q&A",
    "⬆ Upload": "Upload",
    "⚙ Settings": "Settings",
}

page_label = st.sidebar.radio("Navigation", list(NAV_ITEMS.keys()), label_visibility="collapsed")
page = NAV_ITEMS[page_label]

render_hero()
render_status_badges(client, online)

if page == "Dashboard":
    st.subheader("Dashboard")
    if online:
        data = client.analytics()
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents", data["total_documents"])
        col2.metric("Knowledge Chunks", data["total_chunks"])
        col3.metric("Queries", data["total_queries"])
        st.markdown("#### Workspace Shortcuts")
        c1, c2, c3 = st.columns(3)
        c1.markdown('<div class="feature-card"><strong>💬 Legal Chat</strong><span>Ask sourced questions over the local legal knowledge base.</span></div>', unsafe_allow_html=True)
        c2.markdown('<div class="feature-card"><strong>⚖ Rights Advisor</strong><span>Generate rule-backed eligibility reports with supporting sources.</span></div>', unsafe_allow_html=True)
        c3.markdown('<div class="feature-card"><strong>📊 Risk Analysis</strong><span>Detect liability, renewal, penalty, and termination risks.</span></div>', unsafe_allow_html=True)
        with st.expander("Feature guide and example prompts", expanded=True):
            st.markdown(
                """
                - **Dashboard:** system overview, status badges, and quick project metrics.
                - **Legal Chat:** ask sourced legal questions. Example: `My landlord refuses to return my security deposit. What records should I keep?`
                - **Simplify Document:** paste legal text and get a plain-language explanation. Example: paste a lease termination clause.
                - **Rights Advisor:** answer structured facts to generate an eligibility report. Example: Tenant Rights with `deposit withheld` checked.
                - **Risk Analysis:** paste a contract and detect risky clauses. Example: `The vendor shall indemnify the client for all losses.`
                - **Compare Documents:** paste old and new contract versions. Example: compare `30 days notice` vs `7 days notice`.
                - **Clause Extraction:** extract termination, payment, liability, indemnity, and renewal clauses. Example: `This agreement shall cease after 30 days.`
                - **Knowledge Base:** browse indexed legal documents by domain.
                - **Analytics:** show indexed documents, chunks, queries, and domain statistics.
                - **Rights Q&A:** ask open-ended rights questions. Example: `How do I report UPI fraud?`
                - **Upload:** index your own PDF, DOCX, or TXT legal document.
                - **Settings:** Customize the application accent color.
                """
            )
    else:
        st.warning("Backend is offline. Start the FastAPI server to use the workspace.")

elif page == "Settings":
    st.subheader("Settings")
    with st.container():
        st.markdown('<div class="settings-panel">', unsafe_allow_html=True)
        accent_choice = st.selectbox(
            "Accent color",
            list(ACCENTS.keys()),
            index=list(ACCENTS.keys()).index(st.session_state.accent),
            format_func=lambda item: {
                "Blue": "Blue (Default)",
                "Emerald": "Emerald",
                "Purple": "Purple",
                "Orange": "Orange",
            }[item],
        )
        if accent_choice != st.session_state.accent:
            st.session_state.accent = accent_choice
            st.rerun()
        st.caption("Accent color preference is stored for this browser session.")
        st.markdown("</div>", unsafe_allow_html=True)

elif page == "Chat":
    st.subheader("Ask from your local legal knowledge base")
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    prompt = st.chat_input("Ask about a document, clause, or right")
    if prompt:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Analyzing legal question..."):
                data = client.chat(prompt)
            st.markdown(data["answer"])
        st.session_state.messages.append({"role": "assistant", "content": data["answer"]})

elif page == "Rights Eligibility":
    st.subheader("Rights eligibility engine")
    modules = client.rights_modules() if online else {}
    labels = {value: key for key, value in modules.items()}
    selected_label = st.selectbox("Rights module", list(labels.keys()) or ["Tenant Rights"])
    domain = labels.get(selected_label, "tenant")

    facts = {}
    if domain == "tenant":
        facts["tenant_months"] = st.number_input("Months as tenant", min_value=0, value=12)
        facts["eviction_notice_received"] = st.checkbox("Eviction notice received")
        facts["deposit_withheld"] = st.checkbox("Security deposit withheld")
        facts["repair_issue"] = st.checkbox("Repair or habitability issue")
    elif domain == "employment":
        facts["terminated"] = st.checkbox("Terminated or forced to resign")
        facts["unpaid_wages"] = st.checkbox("Unpaid salary or wage issue")
    elif domain == "consumer":
        facts["defective_product"] = st.checkbox("Defective product")
        facts["service_deficiency"] = st.checkbox("Service deficiency")
    elif domain == "cyber":
        facts["money_lost"] = st.checkbox("Money lost in cyber incident")
    elif domain == "women":
        facts["harassment"] = st.checkbox("Workplace harassment")
        facts["domestic_violence"] = st.checkbox("Domestic violence or household abuse")
    elif domain == "senior":
        facts["maintenance_needed"] = st.checkbox("Maintenance support needed")
        facts["abuse"] = st.checkbox("Abuse, neglect, or property pressure")
    elif domain == "rti":
        facts["government_information"] = st.checkbox("Information is held by a public authority")

    if st.button("Generate rights report", type="primary"):
        with st.spinner("Building rights report..."):
            data = client.rights_eligibility(domain, facts)
        render_loading_steps(
            "Rights advisory pipeline",
            ["Evaluating eligibility rules", "Retrieving supporting sources", "Preparing explainable report"],
        )
        st.markdown(f"### {data['module']}")
        render_confidence(data.get("confidence", 0))
        for right in data["rights"]:
            if right["applies"]:
                st.success(right["title"])
            else:
                st.info(right["title"])
            st.write(right["explanation"])
            st.caption(f"Source: {right['source']}")
            if right.get("supporting_sources"):
                render_sources(right["supporting_sources"], "Supporting Legal Knowledge")
        if data.get("missing_facts"):
            st.warning("Missing facts: " + ", ".join(data["missing_facts"]))

elif page == "Clause Extraction":
    st.subheader("Automatic clause extraction")
    text = st.text_area("Paste contract or legal document text", height=320)
    if st.button("Extract clauses", type="primary", disabled=not text.strip()):
        with st.spinner("Extracting clauses..."):
            data = client.extract_clauses(text)
        render_loading_steps("Clause extraction pipeline", ["Splitting document text", "Running regex detectors", "Running semantic clause matching"])
        if not data["clauses"]:
            st.info("No known clause types detected.")
        for clause_type, clauses in data["clauses"].items():
            with st.expander(f"{clause_type.replace('_', ' ').title()} ({len(clauses)})", expanded=True):
                for item in clauses:
                    st.write(item["clause"])

elif page == "Risk Analyzer":
    st.subheader("Contract risk analyzer")
    text = st.text_area("Paste contract text", height=320)
    if st.button("Analyze risk", type="primary", disabled=not text.strip()):
        with st.spinner("Detecting contract risks..."):
            data = client.analyze_risk(text)
        render_loading_steps("Risk analysis pipeline", ["Extracting legal clauses", "Scoring risk severity", "Preparing risk summary"])
        high_count = sum(1 for risk in data["risks"] if risk["severity"] == "High")
        medium_count = sum(1 for risk in data["risks"] if risk["severity"] == "Medium")
        overall = "High" if high_count else "Medium" if medium_count else "Low"
        label, score, _ = risk_visual(overall)
        st.markdown(f"### Overall: {label}")
        st.progress(score)
        st.write("Clause summary")
        st.json(data["clause_summary"])
        for risk in data["risks"]:
            render_risk_card(risk)

elif page == "Compare":
    st.subheader("Legal document comparison")
    col1, col2 = st.columns(2)
    with col1:
        old_text = st.text_area("Old version", height=360)
    with col2:
        new_text = st.text_area("New version", height=360)
    if st.button("Compare versions", type="primary", disabled=not old_text.strip() or not new_text.strip()):
        with st.spinner("Comparing legal documents..."):
            data = client.compare_documents(old_text, new_text)
        render_loading_steps("Comparison pipeline", ["Reading old version", "Reading new version", "Detecting clause-level changes"])
        st.caption(f"Old clauses: {data['old_clause_count']} | New clauses: {data['new_clause_count']}")
        if not data["changes"]:
            st.success("No clause-level changes detected.")
        for change in data["changes"]:
            with st.expander(f"{change['change_type'].title()} - {change['impact_hint']}", expanded=True):
                st.markdown("Old")
                st.write(change["old"] or "None")
                st.markdown("New")
                st.write(change["new"] or "None")

elif page == "Analytics":
    st.subheader("Usage and knowledge base analytics")
    if online:
        data = client.analytics()
        col1, col2, col3 = st.columns(3)
        col1.metric("Documents", data["total_documents"])
        col2.metric("Chunks", data["total_chunks"])
        col3.metric("Queries", data["total_queries"])
        st.markdown("Documents by domain")
        st.dataframe(data["documents_by_domain"], use_container_width=True, hide_index=True)
        st.markdown("Documents by source")
        st.dataframe(data["documents_by_source"], use_container_width=True, hide_index=True)
        st.markdown("Most asked questions")
        st.dataframe(data["most_asked_questions"], use_container_width=True, hide_index=True)
    else:
        st.info("Backend is offline.")

elif page == "Simplify":
    st.subheader("Simplify legal text")
    text = st.text_area("Paste legal text", height=320)
    if st.button("Simplify", type="primary", disabled=not text.strip()):
        with st.spinner("Simplifying legal document..."):
            data = client.simplify(text)
        render_loading_steps("Simplification pipeline", ["Reading legal text", "Finding obligations and risks", "Writing plain-language explanation"])
        st.markdown(data["summary"])

elif page == "Upload":
    st.subheader("Upload a legal document")
    st.markdown(
        """
        <div class="upload-card">
          <strong>📄 Drag & Drop Documents</strong>
          <span>Supported formats: PDF - DOCX - TXT</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    uploaded = st.file_uploader("TXT, PDF, or DOCX", type=["txt", "pdf", "docx"])
    if uploaded and st.button("Index document", type="primary"):
        suffix = Path(uploaded.name).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp:
            temp.write(uploaded.getbuffer())
            temp_path = Path(temp.name)
        try:
            with st.spinner("Indexing uploaded document..."):
                result = client.upload_document(temp_path)
            render_loading_steps("Document indexing pipeline", ["Extracting text", "Chunking document", "Creating embeddings", "Saving to ChromaDB"])
            st.success(f"Indexed {result['chunks_indexed']} chunks from {result['filename']}")
        finally:
            temp_path.unlink(missing_ok=True)

elif page == "Rights Q&A":
    st.subheader("Rights explainer")
    topic = st.text_input("Topic", placeholder="Example: tenant security deposit")
    if st.button("Explain rights", type="primary", disabled=not topic.strip()):
        with st.spinner("Retrieving rights sources..."):
            data = client.rights(topic)
        render_loading_steps("Rights Q&A pipeline", ["Searching knowledge base", "Selecting citations", "Preparing answer"])
        st.markdown(data["answer"])
        if data.get("citations"):
            render_sources(data["citations"])

else:
    st.subheader("Indexed documents")
    docs = client.documents() if online else []
    if docs:
        grouped = defaultdict(list)
        for doc in docs:
            grouped[doc.get("domain") or "UNKNOWN"].append(doc)
        for domain, items in sorted(grouped.items()):
            with st.expander(f"📚 {domain.replace('_', ' ').title()} ({len(items)})", expanded=domain in {"TENANT", "CONSUMER"}):
                for doc in items:
                    st.markdown(
                        f"""
                        <div class="kb-card">
                          <strong>{html.escape(str(doc.get("title") or doc.get("filename")))}</strong>
                          <div class="source-excerpt">{html.escape(str(doc.get("category") or doc.get("source_type")))} - {doc.get("chunk_count", 0)} chunks</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        with st.expander("Raw document table"):
            st.dataframe(docs, use_container_width=True, hide_index=True)
    else:
        st.info("No documents indexed yet.")

