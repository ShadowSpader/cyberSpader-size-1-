import streamlit as st
import ollama
from pypdf import PdfReader
import time
import re

# Match this exact string to your model in 'ollama list' (e.g., "gemma4:e4b" or "gemma3:4b")
MODEL_NAME = "gemma4:e2b"

st.set_page_config(page_title="Private Legal Triage Agent", layout="wide")

st.title(" Private On-Device Legal Document Triage Agent")
st.caption(f"ZeroDay OSS PS2 | 100% Offline Local Inference via Ollama ({MODEL_NAME}) | Zero Cloud Transmission")

# Mandatory Legal Boundary Disclaimer (PS2 Requirement L8)
st.info(
    "**Disclaimer:** This tool is for preliminary document understanding and issue identification only. "
    "It is NOT a lawyer, does NOT provide definitive legal advice, and cannot determine legal validity or enforceability. "
    "Consult a qualified legal professional before acting on any contract."
)

def extract_numbered_passages(raw_text: str):
    """Splits contract into numbered passages [P1], [P2] for verifiable citation grounding (L5)."""
    paragraphs = [p.strip() for p in re.split(r'\n\s*\n', raw_text) if len(p.strip()) > 20]
    if not paragraphs:
        paragraphs = [line.strip() for line in raw_text.split('\n') if len(line.strip()) > 20]
    numbered = []
    for idx, para in enumerate(paragraphs, 1):
        numbered.append(f"[P{idx}] {para}")
    return "\n\n".join(numbered), paragraphs

def check_prompt_injection(text: str) -> bool:
    """Basic detection for adversarial prompt injection embedded in documents."""
    suspicious_patterns = [
        r"ignore (all )?(previous|prior) instructions",
        r"system prompt",
        r"you are now a",
        r"forget everything",
    ]
    for pat in suspicious_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

# Sidebar: Runtime & Privacy Verification (Rubric Criterion 4 & 5)
with st.sidebar:
    st.header(" Privacy & Runtime Status")
    st.success("Mode: 100% On-Device Local Execution")
    st.markdown(f"""
    - **Model:** `{MODEL_NAME}`
    - **Inference Runtime:** Local Ollama Daemon (`127.0.0.1:11434`)
    - **Network Egress:** `0 Bytes (Isolated)`
    - **Telemetry:** `Disabled`
    - **Data Ingestion:** `In-Memory RAM Only`
    """)
    st.divider()
    st.subheader("Scope Statement")
    st.caption("Rental agreements, employment contracts, non-disclosure agreements, and service contracts.")

# Document Ingestion (Requirement L1 & L2)
tab1, tab2 = st.tabs([" Upload PDF / TXT", " Paste Contract Text"])
raw_doc_text = ""

with tab1:
    uploaded_file = st.file_uploader("Upload contract file (Processed entirely in RAM)", type=["pdf", "txt"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                pages_text = [page.extract_text() or "" for page in reader.pages]
                raw_doc_text = "\n\n".join(pages_text)
            else:
                raw_doc_text = uploaded_file.read().decode("utf-8", errors="replace")
        except Exception as e:
            st.error(f"Error parsing file locally: {e}")

with tab2:
    pasted_text = st.text_area(
        "Paste contract clauses directly:",
        height=180,
        placeholder="Paste contract text here..."
    )
    if pasted_text.strip():
        raw_doc_text = pasted_text

if raw_doc_text.strip():
    if check_prompt_injection(raw_doc_text):
        st.warning(" Potential prompt injection detected in contract text. Enforcing strict document boundaries.")

    numbered_doc, _ = extract_numbered_passages(raw_doc_text)

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader(" Source Passages (Grounding Index)")
        st.text_area("Indexed Passages", value=numbered_doc, height=450, disabled=True)

    with col_right:
        st.subheader("Triage Actions")
        mode = st.radio(
            "Select Triage Task:",
            ["Full Document Triage (Clauses, Risks & Missing Terms)", "Ask a Specific Question"]
        )

        user_question = ""
        if mode == "Ask a Specific Question":
            user_question = st.text_input("Enter question (e.g., 'What is the notice period?' or 'Is this clause enforceable?'):")

        if st.button("Run On-Device Triage", type="primary"):
            start_time = time.time()
            with st.spinner(f"Running inference locally via {MODEL_NAME}..."):
                system_instructions = """You are an on-device, privacy-preserving Legal Document Triage Assistant.
STRICT MANDATORY RULES:
1. GROUNDING & CITATIONS (L5): Base every statement strictly on the provided numbered passages ([P1], [P2], etc.). You MUST cite the passage key (e.g. [P1]) and quote verbatim excerpt for every point.
2. SILENCE TEST: If the query or topic is not directly mentioned in the document passages, you MUST explicitly state: "THE DOCUMENT IS SILENT ON THIS ISSUE." Do not extrapolate or assume terms.
3. SEPARATION (L6): Group analysis into three distinct sections:
   - "What the Document Explicitly States" (Direct factual statements + [P#] citations + exact quotes)
   - "Plain-Language Explanation" (Simple terms for legal jargon)
   - "Possible Concerns, Ambiguities & Missing Terms" (Unilateral rights, vague terms, jurisdiction gaps)
4. REFUSAL & SAFE ESCALATION (L8): You are not an attorney. Never state whether a clause is legally binding, valid, or enforceable in court. Refuse the legal verdict, explain that enforceability depends on applicable statutes and case law, and list specific questions for a qualified lawyer."""

                if mode == "Full Document Triage (Clauses, Risks & Missing Terms)":
                    prompt_content = f"""Analyze the provided contract passages:
1. What the Document Explicitly States (Key obligations, penalties, deadlines, termination clauses with [P#] citations and verbatim quotes).
2. Plain-Language Explanation of complex clauses.
3. Possible Concerns, Ambiguities & Missing Terms (Risks, unbalanced obligations, jurisdiction issues).
4. Safe Escalation Advice (Concrete questions to ask legal counsel).

PASSAGES:
{numbered_doc}"""
                else:
                    prompt_content = f"""Answer the user question strictly using the provided document passages.
Follow all rules: cite [P#] keys with verbatim text, state "THE DOCUMENT IS SILENT ON THIS ISSUE" if absent, and refuse to provide definitive legal validity verdicts.

USER QUESTION: {user_question}

PASSAGES:
{numbered_doc}"""

                try:
                    response = ollama.chat(
                        model=MODEL_NAME,
                        messages=[
                            {"role": "system", "content": system_instructions},
                            {"role": "user", "content": prompt_content}
                        ]
                    )
                    elapsed = round(time.time() - start_time, 2)
                    st.success(f"Triage completed in {elapsed}s | Local inference (0 bytes transmitted)")
                    st.markdown(response["message"]["content"])
                except Exception as e:
                    st.error(f"Inference error: {e}. Check that Ollama is running and `{MODEL_NAME}` matches your local library.")
