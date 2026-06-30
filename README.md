# ⚖️ Legal Document Simplification & Rights Advisor

> A Local-First AI Legal Intelligence Platform powered by **FastAPI**, **Streamlit**, **ChromaDB**, **SQLite**, **Retrieval-Augmented Generation (RAG)** and **Ollama**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)
![Streamlit](https://img.shields.io/badge/Streamlit-Latest-red)
![SQLite](https://img.shields.io/badge/SQLite-Database-blue)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-purple)
![Ollama](https://img.shields.io/badge/Ollama-LocalLLM-orange)
![License](https://img.shields.io/badge/License-Educational-lightgrey)

---

# 📖 Overview

The **Legal Document Simplification & Rights Advisor** is a local-first Legal Intelligence Platform designed to help users understand legal documents, explore legal rights, simplify complex legal language, compare contracts, extract clauses, analyse legal risks and answer legal questions using a Retrieval-Augmented Generation (RAG) pipeline.

Unlike general-purpose AI chatbots, this application retrieves information from a curated local legal knowledge base before generating responses, improving transparency and reducing hallucinations.

---

# ✨ Features

## AI Assistant
- Legal Chat
- Conversation Memory
- Query Validation
- Intent Routing
- Prompt Engineering
- Confidence Estimation
- Source Attribution

## Legal Intelligence
- Rights Advisor
- Legal Document Simplification
- Clause Extraction
- Contract Comparison
- Risk Analysis
- Knowledge Graph
- Analytics Dashboard

## AI Pipeline
- Retrieval-Augmented Generation (RAG)
- Semantic Search
- ChromaDB Vector Retrieval
- SQLite Operational Database
- Ollama Local LLM
- Explainable AI Responses

---

# 🏗 Architecture

```text
                 User
                   │
                   ▼
          Streamlit Frontend
                   │
                   ▼
            FastAPI Backend
                   │
      ┌────────────┼─────────────┐
      ▼            ▼             ▼
   SQLite      ChromaDB       Ollama
      │            │             │
      └────────────┼─────────────┘
                   ▼
          AI Generated Response
```

---

# 📁 Project Structure

```text
LEGAL_ADVISOR/
│
├── backend/
│   ├── app/
│   │   ├── api/
│   │   ├── core/
│   │   ├── legal/
│   │   ├── rag/
│   │   ├── db/
│   │   └── data/
│   └── requirements.txt
│
├── frontend/
│   ├── components/
│   ├── pages/
│   └── app.py
│
├── scripts/
│
├── tools/
│   └── ollama/
│
├── README.md
└── .gitignore
```

---

# 💻 Technology Stack

| Layer | Technology |
|-------|------------|
| Language | Python |
| Backend | FastAPI |
| Frontend | Streamlit |
| Database | SQLite |
| Vector Database | ChromaDB |
| Embeddings | Sentence Transformers |
| Local LLM | Ollama |
| AI Pipeline | Retrieval-Augmented Generation |

---

# 📋 Prerequisites

- Python 3.11+
- Git
- Ollama
- 8 GB RAM (recommended)

---

# 🤖 Installing Ollama (Required for AI Features)

1. Download Ollama from:

https://ollama.com/download

2. Install Ollama normally.

3. Create the following directory inside the project if it does not already exist:

```text
tools/
└── ollama/
```

4. Install or place the Ollama executable inside **tools/ollama/** (for project organization).

5. Verify the installation:

```powershell
ollama --version
```

6. Download the recommended model:

```powershell
ollama pull llama3.1:8b
```

(Replace with the model configured by your necessity if different.)

7. Start the Ollama service:

```powershell
ollama serve
```

**Keep this terminal running while using the AI features.**

---

## Setup

```powershell
cd C:\Users\Rahul\Desktop\Legal_advisor
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
pip install -r frontend\requirements.txt
Copy-Item backend\.env.example backend\.env
```

## Index Seed Knowledge Base

```powershell
python scripts\init_knowledge_base.py
```

This indexes both `backend\app\data\knowledge_base` and `backend\app\data\legal_knowledge_base`.

To rebuild legal-source SQLite rows and ChromaDB from scratch:

```powershell
python scripts\init_knowledge_base.py --reset
```

---

# ▶ Running the Project

## Step 1 — Start Ollama

```powershell
ollama serve
```

## Step 2 — Start Backend

```powershell
cd backend

uvicorn app.main:app --reload
```

Backend:

http://127.0.0.1:8000

Swagger:

http://127.0.0.1:8000/docs

## Step 3 — Start Frontend
Open a second terminal:

```powershell
cd C:\Users\Rahul\Desktop\Legal_advisor
.\.venv\Scripts\Activate.ps1
streamlit run frontend\app.py
```

Frontend:

http://localhost:8501

---

# 🔄 Startup Order

1. Start Ollama
2. Start FastAPI Backend
3. Launch Streamlit Frontend
4. Open the browser

---

# 💬 Example Questions

- What is RTI?
- Explain a contract.
- What are my consumer rights?
- My employer has not paid my salary.
- Simplify this agreement.
- Compare these contracts.
- Extract important clauses.
- Analyse legal risks.
- How do I report cyber fraud?

---

# 🔌 Major API Endpoints

| Endpoint | Purpose |
|----------|---------|
| /chat | Legal AI Chat |
| /simplify | Document Simplification |
| /compare | Contract Comparison |
| /risk | Risk Analysis |
| /rights | Rights Advisor |
| /api/intelligence/dashboard | Dashboard |
| /api/intelligence/knowledge-graph | Knowledge Graph |

---

# 🧪 Testing

The project includes:

- Manual Testing
- Automated Testing
- Retrieval Testing
- API Testing
- RAG Validation
- Knowledge Base Verification
- End-to-End Workflow Testing

---

# 🛠 Troubleshooting

| Problem | Solution |
|----------|----------|
| Backend won't start | Install requirements again |
| No AI response | Ensure `ollama serve` is running |
| Wrong retrieval | Rebuild embeddings |
| Frontend cannot connect | Start backend first |
| Knowledge updates missing | Re-index the knowledge base |

---

# 🔒 Security & Privacy

- Local-first execution
- No mandatory cloud AI
- SQLite local storage
- ChromaDB local vector database
- Local Ollama inference
- Query validation
- Source attribution
- Confidence estimation

---

# ⚠ Disclaimer

This project is intended for educational and informational purposes only.

It does **not** replace professional legal advice. Always consult a qualified legal professional for decisions relating to legal matters.

---

# 🚀 Future Roadmap

- Multilingual Support
- OCR Integration
- Voice Assistant
- Mobile Application
- Fine-tuned Legal LLM
- Government Portal Integration
- Cloud Synchronization (Optional)

---


# 🙏 Acknowledgements

Built using:

- Python
- FastAPI
- Streamlit
- ChromaDB
- SQLite
- Sentence Transformers
- Ollama

Special thanks to the open-source community for making these technologies available.

---
