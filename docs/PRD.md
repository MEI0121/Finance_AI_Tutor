# Product Requirements Document (PRD): Finance AI Tutor (PoC)

## 1. Executive Summary & Objective
The Finance AI Tutor is a Proof of Concept (PoC) designed to transform static financial curriculum (specifically CFA Level II Equity Valuation) into an interactive, multi-modal learning experience. 

**Objective:** To build an AI-driven, hallucination-free educational platform that prioritizes learner agency, allowing users to seamlessly transition between reading, chatting, and self-assessment within a unified workspace.

## 2. Target Audience & User Personas
* **Primary Persona:** Advanced Financial Learner (e.g., CFA Candidate).
  * *Pain Points:* Traditional LMS pathways are too rigid. LLMs hallucinate complex financial formulas. Switching between PDF textbooks, video lectures, and chat interfaces breaks learning flow.
  * *Goals:* Needs mathematically accurate explanations, immediate contextual help while reading, and active recall testing.

## 3. Scope & Assumptions
* **In Scope for PoC:** Local deployment, Single PDF ingestion (CFA Reading 22), LangGraph stateful chat orchestration, dynamic generation of Markdown slides and MCQs, custom UI workspace.
* **Out of Scope for PoC:** User authentication/login, persistent cloud database for cross-session history, enterprise-grade OCR for nested tables, mobile responsiveness.

## 4. User Stories & Acceptance Criteria (AC)

### Epic 1: Multi-Modal Learning Workspace
**User Story 1.1:** As a learner, I want to view my textbook, slides, and quizzes in the same window as my chat tutor so that I don't lose context.
* **AC 1:** The UI must implement a split-pane layout (Content on left, Chat on right).
* **AC 2:** Users can toggle between Textbook, Slides, and Quiz tabs with zero latency and without losing their chat history.

**User Story 1.2:** As a learner, I want the textbook reading experience to feel natural and uninterrupted.
* **AC 1:** The PDF viewer must display a "Two-Page Spread" layout.
* **AC 2:** Pagination must progress one page at a time to prevent cognitive breaks for concepts spanning across pages.

### Epic 2: Pedagogical AI Orchestration (LangGraph)
**User Story 2.1:** As a learner, I want the AI to suggest what I should ask next so that I don't experience "prompt paralysis."
* **AC 1:** After every AI response, the system must generate 2-3 clickable predictive suggestion pills (`---SUGGESTIONS---`).

**User Story 2.2:** As a learner, I want the AI to guide me when I get a question wrong, rather than just giving me the answer.
* **AC 1:** When a user selects a wrong answer in the Assessment mode, the AI must provide Socratic hints (Diagnostic CoT) to help them find the algebraic error.
* **AC 2:** The AI must *never* reveal the correct MCQ option directly upon a failure.

### Epic 3: High-Fidelity Content Generation
**User Story 3.1:** As a learner, I want to highlight text in the PDF to instantly ask the tutor to explain it.
* **AC 1:** Highlighting text triggers a pop-up action menu.
* **AC 2:** Clicking "Ask Tutor" injects the highlighted text into the chat and generates a context-aware response.

**User Story 3.2:** As a learner, I want the AI to generate summary slides of the chapter I am studying.
* **AC 1:** The system provides an on-demand API to generate slides.
* **AC 2:** Slides must support KaTeX rendering for financial formulas.
* **AC 3:** The final slide must include a "Feynman Node" prompt, challenging the user to summarize the concept.

## 5. Non-Functional Requirements (NFRs)

### 5.1 Anti-Hallucination & Accuracy (Critical)
* **Strict RAG Grounding:** The AI must operate in a "Closed-Book" mode, strictly utilizing the ingested ChromaDB vector store.
* **Mathematical Accuracy:** To prevent algebraic hallucinations, the generative APIs must utilize an **Implicit Scratchpad Pattern**. The LLM must output a `calculation_scratchpad` (hidden from the user) before populating the final answer options.

### 5.2 Performance & Architecture
* **Decoupled State:** Chat orchestration (Stateful) must be separated from Slide/Quiz generation (Stateless) to prevent LangGraph state bloat and reduce latency.
* **Containerization:** The entire application (Next.js frontend, FastAPI backend, ChromaDB) must be deployable via a single `docker-compose up` command to ensure environmental consistency.

## 6. Success Metrics (KPIs for Future Beta)
* **Engagement:** Average session length and number of chat turns per session.
* **Accuracy:** Rate of reported AI hallucinations or math errors (Target: < 1%).
* **Pedagogical Effectiveness:** Completion rate of the "Feynman Node" summaries.
