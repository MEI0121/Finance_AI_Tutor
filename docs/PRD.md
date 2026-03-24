# Product Requirements Document (PRD): Finance AI Tutor (PoC)

## 1. Product Overview
The Finance AI Tutor is a multi-modal, context-aware educational platform designed for advanced financial learners (e.g., CFA candidates). This Proof of Concept (PoC) specifically targets the CFA Level II Equity Valuation - Reading 22 (Discounted Dividend Valuation) module. Moving away from rigid, node-based Learning Management Systems (LMS) and reactive chatbots, this product implements a **User-Led Discovery Flow** orchestrated by LangGraph, providing contextual guidance and hallucination-free RAG support for an immersive learning experience.

## 2. Core Vision & Design Principles (The 8 Core Innovations)
The product architecture and UX are driven by eight fundamental design choices:
1. **"Open Classroom" Exploration Model:** The AI acts as a professional tutor in an always-open environment, supporting non-linear, self-directed learning rather than forced linear progression.
2. **Strict Anti-Hallucination Engineering:** Utilizes a closed-book RAG architecture combined with an "Implicit Calculation Scratchpad" (Chain of Thought) to guarantee the AI generates accurate, mathematically sound financial explanations strictly tied to the curriculum.
3. **Seamless Multi-Modal Workspace:** Users can instantly toggle between Textbook, AI Slides, AI Quizzes, and Video components without losing conversational context or interrupting their learning flow.
4. **Optimized "Two-Page Spread" Reading UX:** Features a custom "split two-page, single-page turn" design to prevent cognitive breaks when reading complex financial concepts that span across pages.
5. **Socratic Remediation & The Feynman Technique:** Incorrect quiz answers trigger Socratic hints to guide the user's algebraic logic rather than revealing the correct option. Slides conclude with a "Feynman Node," challenging the user to summarize concepts to prove mastery.
6. **Interactive "Highlight-to-Action" Tooling:** Users can highlight text within the textbook or slides to instantly query the AI tutor, translate, or add to notes, maximizing active learning.
7. **Context-Aware Predictive Suggestions:** The chat UI automatically generates contextual follow-up prompts (`---SUGGESTIONS---`) after every interaction to cure "prompt blank paralysis" and drive the conversation forward.
8. **Total Data-to-Logic Decoupling:** The pedagogical orchestrator is entirely decoupled from the source material, achieving a highly scalable "teach what it is fed" capability.

## 3. Core Features & State Machine Requirements

### 3.1 LangGraph Pedagogical Orchestrator
The backend relies on LangGraph to construct a Directed Cyclic Graph (DCG) that manages the pedagogical state and intent routing.
* **Required Graph Nodes:** `greeting`, `planning`, `micro_teach`, `qna` (Interactive Sandbox), `assess` (MCQ generation), `remediate` (Socratic feedback), and `feynman` (capstone summary).
* **Socratic Evaluation (`remediate`):** When a user fails an assessment, the system must trigger a Diagnostic Chain-of-Thought. The AI steps through the algebraic logic to help the user locate their specific error without exposing the correct answer.
* **Circuit Breaker Mechanism:** The state machine must track `remediation_attempts`. If a user fails the same concept $\ge 3$ times, the AI breaks the Socratic loop and provides a clear, step-by-step resolution to prevent extreme frustration.

### 3.2 Generative UI & Multi-Modality
* **Decoupled Utilities:** Slides (Markdown/KaTeX) and Quizzes must operate as on-demand, stateless APIs to prevent state bloat within the LangGraph orchestrator.
* **PDF Interactivity:** The frontend must integrate a custom `react-pdf` reader with boundary clamping and text-selection listeners, bridging the static textbook with the dynamic AI tutor.

## 4. Anti-Hallucination & Engineering Constraints
Given the strict requirements of financial mathematics, the system must aggressively mitigate LLM hallucinations.
* **Implicit Scratchpad Pattern:** All Pydantic schemas for mathematical generation must prioritize a hidden `calculation_scratchpad` field. The LLM is forced to explicitly write out its algebraic derivations *before* outputting the final `correct_answer`. This field is hidden from the frontend UI.
* **Ironclad System Prompt (`IRONCLAD_TEMPLATE`):** The global prompt must enforce strict "Closed-Book" rules. The LLM must gracefully decline questions that fall outside the retrieved RAG context.
* **High-Fidelity Ingestion:** Document parsing must utilize PyMuPDF (`fitz`) with specific chunking parameters (`CHUNK_SIZE = 4000`, `CHUNK_OVERLAP = 800`) to prevent multi-step valuation formulas from being severed during vectorization.

## 5. Architecture & Deployment Specifications

### 5.1 Tech Stack
* **Frontend:** Next.js (React), Tailwind CSS, `react-pdf`.
* **Backend:** Python FastAPI, Uvicorn, LangGraph, LangChain.
* **AI & Data Layer:** OpenAI `gpt-4o`, ChromaDB, PyMuPDF.

### 5.2 Containerization (Docker)
The system is deployed as a multi-container stack via `docker-compose.yml`:
* **Frontend Service (`Dockerfile.frontend`):** Multi-stage build based on `node:20-alpine`, exposing port `3000`.
* **Backend Service (`Dockerfile.backend`):** Built on `python:3.11-slim`. 
* **Startup Sequence:** The backend container **must** execute the data ingestion script (`python -m backend.ingest_pdf`) to build/rebuild the Chroma index *before* starting the FastAPI uvicorn server.
* **Data Persistence:** ChromaDB storage must be mapped to a host volume (`./backend/chroma_db`) to prevent embedding loss between container restarts.

## 6. Enterprise Scaling Roadmap
To evolve this PoC into a production-grade enterprise platform, the following milestones are planned:
1. **Enterprise Multi-Modal Parsing:** Transition to LlamaParse or Unstructured.io to accurately extract nested financial tables and charts from CFA PDFs.
2. **GraphRAG Integration:** Upgrade from standard semantic ChromaDB similarity to an entity-based Knowledge Graph (GraphRAG) to facilitate cross-chapter financial insights.
3. **Progressive Scaffolding & Gating:** Introduce a "Prerequisite Gating" mechanism (e.g., Mandatory Video $\rightarrow$ Baseline Quiz $\rightarrow$ Unlock Free Chat) to solve the "cold start" problem for absolute beginners.
4. **Persistent Analytics (PostgreSQL):** Implement a relational database to store user telemetry and LangGraph checkpoints, allowing educators to identify systemic curriculum gaps based on aggregate performance data.
