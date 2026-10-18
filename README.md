# DocuShield: Private On-Device Legal Document Triage Agent
**ZERODAY OSS Hacktoberfest 2026 — Track PS2 (Agentic AI)**

An open-source, 100% offline legal document triage assistant powered by Google's Gemma edge-class model running on local GPU via Ollama. It translates complex legal contracts into plain language, verifies every claim with passage-anchored quotes, flags omitted clauses, and guarantees zero cloud egress.

---

## Technical Contribution Statement
DocuShield eliminates cloud data exposure in legal review by pairing an in-memory document ingestion buffer (`io.BytesIO` + `pypdf`) with an on-device BM25 context filter and a local edge-class Gemma LLM. It guarantees zero-byte network transmission, prevents disk persistence, and enforces calibrated silence on missing clauses without hallucination.

---

## Key Capabilities (PS2 Mandatory Criteria)
- **100% On-Device Privacy (L9 / Rubric 4)**: 0 bytes of network transmission. No cloud API calls, no analytics, no external telemetry.
- **Traceable Grounding (L5 / Rubric 2)**: All document paragraphs are partitioned and tagged (`[P1]`, `[P2]`). Factual claims are tied directly to verifiable verbatim quotes.
- **Calibrated Silence (L7 / Rubric 2)**: Outputs `"THE DOCUMENT IS SILENT ON THIS ISSUE"` when requested provisions are absent, avoiding hallucinated boilerplate.
- **Ethical Boundaries & Safe Escalation (L8 / Rubric 3)**: Explicitly refuses to offer binding legal validity rulings and produces sharp, actionable questions for licensed attorneys.
- **Adversarial Prompt Defense (Rubric 6)**: Treats contract text strictly as untrusted data, resisting embedded prompt overrides.

---

## Architecture & Data Flow
[User Contract (PDF/TXT)]
│
▼
[Ephemeral Memory Buffer: io.BytesIO] (Zero disk writes)
│
▼
[Partition Engine: parse_and_index_document] -> Generates [P1], [P2] tags
│
▼
[Local Context Filter (BM25 in RAM)] -> Isolates top matching passages (<2ms)
│
▼
[Ollama Daemon (127.0.0.1:11434)] -> Local loopback execution
│
▼
[Google Gemma Edge Model (GPU VRAM)] -> Streams plain-language triage


---

## Hardware Sizing & Benchmarks
- **Test Machine**: Windows 11 Laptop, NVIDIA GeForce RTX 3050 (4GB VRAM)
- **Local Model**: Google Gemma edge-class (`gemma4:e2b` / `gemma2:2b`)
- **Quantization**: 4-bit edge quant (`q4_K_M`)
- **VRAM Utilization**: ~1.8 GB (Leaves 2.2 GB headroom for OS display)
- **Retrieval Latency**: <2 ms (In-memory BM25 filtering)
- **RAM Overhead**: ~140 MB (No temporary files cached on disk)

---

## Quickstart & Setup (Reproducibility)

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com) installed and running locally

### 2. Pull Edge Model
```bash
ollama run gemma4:e2b
# Fallback edge size: ollama run gemma2:2b
