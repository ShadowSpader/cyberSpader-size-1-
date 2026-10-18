# Problem Statement 2: Private On-Device Legal Document Triage Agent

## Overview
A 100% offline, privacy-first legal document analysis and triage tool powered by Google's Gemma edge-class model running locally via Ollama. It enables users to inspect contracts (rental, employment, service agreements) without transmitting sensitive legal documents over the internet.

## Key Capabilities
- **100% On-Device Processing (L9):** Zero cloud API calls, zero telemetry, and complete network isolation.
- **Traceable Grounding (L5):** Automatically indexes documents into numbered passages (`[P1]`, `[P2]`) with verbatim clause citations.
- **Calibrated Silence (L7):** Explicitly outputs `"THE DOCUMENT IS SILENT ON THIS ISSUE"` when clauses are absent, eliminating hallucinated terms.
- **Ethical Legal Guardrail (L8):** Strictly refuses to issue legal enforceability or validity rulings; escalates with actionable questions for legal counsel.
- **In-Memory Ingestion (L1):** Extracts text from PDFs and text files directly in RAM using `pypdf`.

## Architecture & Technology Stack
- **Model:** Google Gemma Edge Class (`gemma4:e2b`)
- **Local Engine:** Ollama Daemon (`http://localhost:11434`)
- **Application Interface:** Streamlit
- **PDF Parser:** `pypdf` (local in-memory processing)

## Local Setup & Reproduction
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
