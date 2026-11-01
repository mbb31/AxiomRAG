# AxiomRAG 🏛️

<div align="center">
<img width="3584" height="1184" alt="AxiomRAG logo" src="https://github.com/user-attachments/assets/ad66179f-d98e-4447-9686-ade0293f0f3f" />

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-red.svg)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-Integration-green.svg)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A privacy-first, local Retrieval-Augmented Generation (RAG) engine designed for high-density financial documents, corporate valuations, and zero-leakage technical research.

</div>

---

## 🌟 Overview

In quantitative finance, corporate valuation, and econometric research, data privacy is non-negotiable. Uploading sensitive financial models, unreleased earnings reports, or proprietary filings to third-party cloud LLM APIs introduces severe compliance and security risks. 

**AxiomRAG** solves this by providing a completely local, zero-cost RAG pipeline running entirely on local hardware via **Ollama (Llama 3.1)** and **ChromaDB**. It ensures absolute data confidentiality while delivering a modern, ChatGPT-grade user interface for multi-document querying.

---

## 🚀 Key Features

* **🔒 100% Local & Privacy-First:** Built to run entirely on-device with zero external data transmission. Powered by Ollama (`Llama 3.1` + `nomic-embed-text`).
* **📂 Multi-Document Ingestion & Unique Collections:** Automatically parses complex PDF datasets via `pdfplumber` and assigns unique collection UUIDs to prevent context bleeding across files.
* **⚡ Intelligent Chunking & Vectorization:** Utilizes recursive text splitting and ChromaDB for high-speed similarity searches and precise context retrieval.
* **🎨 Modern SaaS UI:** Custom-styled Streamlit interface featuring ChatGPT-style pill inputs, dark-themed aesthetics, and an embedded PDF viewer.

---

## 📊 System Architecture & Tech Stack

* **Frontend & UI:** Streamlit (Custom CSS, Dark Mode, Split-column layout)
* **Orchestration:** LangChain (`RecursiveCharacterTextSplitter`, `Chroma`)
* **Local Inference & Embeddings:** Ollama (`Llama 3.1` for text generation, `nomic-embed-text` for vector embeddings)
* **Data Parsing:** `pdfplumber` for text and layout extraction

---

## 🛠️ Quick Start Guide

### Prerequisites
* Ensure you have Python 3.10+ installed.
* Ensure you have [Ollama](https://ollama.com) installed and running locally with the required models pulled:
  ```bash
  ollama pull llama3.1
  ollama pull nomic-embed-text
  
  ---

## 🎥 Demo

> Watch AxiomRAG ingest financial documents and query data locally in real-time below:


https://github.com/user-attachments/assets/43d7cb9c-f261-49f4-83be-f963062c8f68



---

## 📜 License

This project is open-source under the [MIT License](LICENSE).
