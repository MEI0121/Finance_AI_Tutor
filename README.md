# Finance AI Tutor: Adaptive Learning OS for CFA Equity Valuation

> **AIDF FinTech Research Intern Task Delivery**
> 
> *Target Module: CFA Level II Equity Valuation - Reading 22 (Discounted Dividend Valuation)*

This repository contains the end-to-end Proof of Concept (PoC) for an AI-driven, multi-modal teaching platform. Moving beyond traditional "reactive chatbots," this system implements a user-led discovery flow where a LangGraph orchestrator provides contextual guidance, predictive suggestions, and RAG-grounded support while learners navigate topics, Slides, and Quizzes at their own pace.

---

## 🚀 Quick Start

### Prerequisites
- Docker & Docker Compose (for containerized deployment)
- Node.js 18+ and Python 3.10+ (for local development)
- OpenAI API Key

### Environment Setup
1. Clone this repository.
2. Create a `.env` file in the root directory (you can copy from `.env.example` if provided):
   ```env
   OPENAI_API_KEY=your_actual_openai_api_key_here
   ```

### How to Run (Docker)
The entire application (Frontend + Backend APIs + ChromaDB vector store + PDF Ingestion process) is containerized for zero-config execution.

```bash
# Build and start the entire stack
docker compose up --build
```
Once the containers are running and the ingestion script finishes building the persistent vector database, access the application at:
👉 **http://localhost:3000**

*(To shut down the system and remove volumes, run `docker compose down -v`)*

---

## 🏗️ Architecture

### System Description
The platform is architected around a decoupled **Generative UI Client** and a **Supportive LangGraph Orchestrator Backend**. 

When a learner interacts with the platform (e.g., highlighting a PDF text, answering a generated quiz, or asking a direct question), the request is sent to the backend's LangGraph orchestrator. Instead of a single LLM call, the graph manages conversation memory and RAG context retrieval across specialized nodes (`planning`, `qna`, `micro_teach`, `assess`, `remediate`, `feynman`). In this Research PoC, the graph is intentionally guidance-first (not rigidly enforcement-first), while stricter progression controls are reserved for roadmap enterprise modes. The graph communicates with a persistent Chroma vector database, which utilizes a high-fidelity PyMuPDF ingestion pipeline to ensure mathematical and structural integrity of the CFA textbook.

### Tech Stack & Rationale
| Layer | Technology | Rationale |
| :--- | :--- | :--- |
| **Frontend** | Next.js (React), Tailwind, `react-pdf` | Next.js provides robust API routing and server-side capabilities if needed later. `react-pdf` enables the crucial "Two-Page Spread" native document rendering, keeping the context anchored. |
| **Backend API** | FastAPI, Uvicorn | High-performance, asynchronous Python framework, perfectly suited for heavy LLM I/O and streaming responses. |
| **Orchestration** | LangGraph, LangChain | Essential for moving from a "chatbot" to a "tutor." LangGraph allows explicit modeling of pedagogical states (cycles of teaching, assessing, and remediating). |
| **LLM Engine** | OpenAI `gpt-4o` | Superior reasoning for complex algebraic derivations required in the Gordon Growth Model and Multistage DDMs. |
| **RAG & DB** | ChromaDB, PyMuPDF (`fitz`) | PyMuPDF extracts text block-by-block, preserving formula structures that standard PDF loaders destroy. |

---

## 📁 Project Structure

```text
finance-ai-tutor/
├── backend/                  # Python/FastAPI Backend Services
│   ├── chroma_db/            # Persistent Vector Database storage
│   ├── main.py               # FastAPI entry point & API route definitions
│   ├── tutor_graph.py        # Core LangGraph state machine & adaptive routing logic
│   ├── generative_content.py # Prompts for dynamic Slides, Quizzes, and CoT evaluation
│   ├── schemas.py            # Pydantic models enforcing Structured Output (e.g., Implicit Scratchpad)
│   └── ingest_pdf.py         # PyMuPDF ingestion pipeline and chunking logic
├── frontend/                 # Next.js Frontend Application
│   ├── app/
│   │   ├── layout.tsx
│   │   └── page.tsx          # Main workspace layout (Video, Sidebar, Tabs)
│   ├── components/
│   │   ├── TextbookPdf.tsx   # Custom Two-Page Spread PDF reader with boundary clamping
│   │   ├── AiSlides.tsx      # Markdown/KaTeX slide renderer
│   │   └── AiQuiz.tsx        # Interactive assessment UI
│   └── package.json
├── docs/                     # Architectural and Pedagogical Documentation
│   ├── design.md             # Core Pedagogical Design (MUST READ)
│   └── deployment.md         # Deployment & Containerization strategies
├── docker-compose.yml        # Multi-container orchestration
├── Dockerfile.backend        
├── Dockerfile.frontend       
└── README.md                 
```

---

## 🧠 Design Decisions

The core engineering effort of this PoC was spent solving critical LLM failure modes in educational contexts (e.g., mathematical hallucinations, premature JSON guessing, and passive UX), while centering learner agency in a multi-modal workspace (Slides, Quiz, Video) tightly connected to textbook-grounded chat.

Please refer to the comprehensive pedagogical and architectural design document here:
👉 **[docs/design.md](./docs/design.md)** *Includes detailed breakdowns of:*
* *The Adaptive State Machine Logic*
* *Implicit Scratchpad (Generation-Time CoT)*
* *Socratic Remediation Mechanisms*