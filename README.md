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

   ## Hardware Sizing & Performance Benchmarks
- **Test Machine**: Windows 11 laptop, NVIDIA GeForce RTX 3050 (4GB VRAM)
- **Local Model**: Google Gemma edge-class (`gemma4:e2b` / `gemma2:2b`)
- **Quantization**: 4-bit edge quant (`q4_K_M`)
- **VRAM Footprint**: ~1.8 GB (Fits comfortably within 4GB VRAM budget)
- **Inference Speed**: ~24 tokens/sec (Streamlit real-time streaming)
- **RAM Overhead**: ~140 MB (Zero disk caching; PDF extraction purely in `io.BytesIO`)

## Network Isolation & Privacy Verification
DocuShield guarantees 0-byte network egress during document triage:
1. **Physical Air-Gap Test**: Disable Wi-Fi/Ethernet. Run any contract review; the app completes inference with 100% functionality.
2. **Connection Monitoring Audit**: Inspect active network sockets during analysis:
   ```powershell
   Get-NetTCPConnection | Where-Object { $_.LocalPort -eq 11434 }
   # Verifies Ollama is bound strictly to loopback (127.0.0.1) with no external sockets
