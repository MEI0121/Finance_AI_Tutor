# Pedagogical Design: Finance AI Tutor

This document outlines the pedagogical and architectural decisions behind the Finance AI Tutor. Rather than a reactive "Q&A chatbot," this system implements a **User-Led Discovery Flow with a Supportive LangGraph Orchestrator** to emulate guided, context-rich learning while preserving learner agency.

---

## 💡 Design Highlights (The 8 Core Innovations)

During the development of this PoC, eight key design choices were made to differentiate this tool from traditional LMS platforms and generic AI chatbots:

1. **"Open Classroom" Exploration Model:** Breaking away from rigid, node-based progression, we designed an "always-open classroom." The AI acts as a professional tutor waiting for the student, allowing for non-linear, self-directed learning.
2. **Strict Anti-Hallucination Engineering:** Financial education requires absolute precision. We implemented a closed-book RAG architecture and a "Calculation Scratchpad" (Chain of Thought) mechanism. The AI generates content *strictly* based on the vectorized textbook curriculum, preventing out-of-bounds or fabricated financial advice.
3. **Seamless Multi-Modal Workspace:** The UI allows instant toggling between Textbook, AI Slides, and AI Quiz modes. The Chat Tutor and Video components can be invoked or dismissed without losing context or interrupting the learning flow. Generative AI continuously supplies fresh slides and quizzes for ongoing reinforcement.
4. **Optimized "Two-Page Spread" Reading UX:** The textbook interface features a "split two-page, single-page turn" design. This ensures that unless a concept spans three continuous pages, the user maintains a fluid, uninterrupted reading experience without cognitive breaks.
5. **Socratic Remediation & The Feynman Technique:** In the Quiz module, wrong answers trigger Socratic feedback—analyzing the misconception and guiding the thought process *without* revealing the correct answer. Correct answers provide full analytical breakdowns. Furthermore, Slides conclude with a "Feynman Node," challenging users to summarize concepts in their own words to prove mastery.
6. **Interactive "Highlight-to-Action" Tooling:** Users can highlight text in the Textbook and Slides to instantly ask questions, translate, or add to notes (mocked). Quizzes offer the ability to save valuable questions to a workbook (mocked), maximizing self-directed learning capabilities.
7. **Context-Aware Predictive Suggestions:** To cure "prompt blank paralysis," the Chat interface generates contextual follow-up suggestions (`---SUGGESTIONS---`) after every interaction. Clicking these pills instantly sends the query, boosting chat efficiency and stimulating continuous inquiry.
8. **Total Data-to-Logic Decoupling:** The AI tutoring logic is entirely decoupled from the source material. The system achieves a "teach what it is fed" capability. The finer the raw textbook parsing and the cleaner the data preprocessing, the higher the quality of the resulting AI classroom.

---

## 1. Lesson Flow (User-Led Discovery with LangGraph)

The pedagogical flow is modeled as a Directed Cyclic Graph (DCG) using `LangGraph`, with guidance-first orchestration rather than hard gating. The tutor provides contextual explanations, memory-aware follow-ups, and predictive suggestions, while the learner can move across topics and workspace surfaces at their own pace.

**Core Nodes in the State Machine:**
1. **`greeting` & `planning`:** The AI assesses the user's initial intent and generates a structured learning plan based on the CFA Reading 22 curriculum.
2. **`micro_teach`:** Delivers bite-sized, RAG-grounded explanations of specific concepts (e.g., the Gordon Growth Model).
3. **`qna` (Interactive Sandbox):** Allows the learner to ask clarifying questions, branch into direct exploration, and continue through contextual guidance.
4. **`assess`:** Generates dynamic Multiple Choice Questions (MCQs) when requested, enabling optional comprehension checks.
5. **`remediate`:** Triggered upon an incorrect assessment. Provides Socratic hints.
6. **`feynman`:** The capstone node. Upon completing the chapter, the learner must explain the core concept in their own words to prove mastery.

---

## 2. Adaptive Logic & Remediation

The system's routing logic separates conversational exploration from evaluation while remaining learner-directed.

- **Advancing vs. Remediating:** When in the `assess` node, the AI evaluates the learner response against the generated question and retrieved context.
  - *Correct Answer:* Provides the full analytical breakdown and suggests moving to the next `micro_teach` step, or to the final `feynman` checkpoint.
  - *Wrong Answer:* The graph routes to `remediate` for Socratic hinting.
- **Socratic Diagnostic CoT:** During remediation, the AI does *not* reveal the correct option. It applies a **Diagnostic Chain-of-Thought (CoT)**, stepping through the algebra to help the learner locate their specific mathematical error (e.g., confusing $D_0$ with $D_1$).
- **The "Circuit Breaker":** Tracks `remediation_attempts`. If `attempts >= 3` on a single concept, the AI triggers a "Circuit Breaker"—it gently breaks the Socratic loop, provides a concise step-by-step resolution, and returns to a flexible exploration state.

---

## 3. Content Integration & Multi-Modality

The workspace is divided into a **Static/Generative UI Pane** (Left) and an **Adaptive Tutor Pane** (Right), forming a unified multi-modal learning surface.

- **High-Fidelity RAG Pipeline:** We utilized **PyMuPDF (`fitz`)** with a large `CHUNK_SIZE = 4000` and `CHUNK_OVERLAP = 800`. This allows the LLM to ingest entire multi-step mathematical derivations in a single context window.
- **Dynamic AI Slides & Quizzes:** Content is not hardcoded. The system synthesizes retrieved RAG chunks into Markdown/KaTeX-formatted presentation slides and rigorous MCQs on the fly.
- **Multi-Modal Workspace Core:** Slides, Quiz, Video, and textbook chat coexist in one workspace so learners can switch modes fluidly without losing context.
- **PDF-to-Chat Interactivity:** The custom Two-Page Spread PDF reader plus chat highlight/Q&A loop is the primary bridge between textbook evidence and tutor guidance.

---

## 4. AI Prompt Design (Combating LLM Hallucinations)

In financial education, a fabricated valuation formula is unacceptable. The prompt engineering strategy focused entirely on **Anti-Hallucination** and **Mathematical Rigor**:

1. **`IRONCLAD_TEMPLATE` (Closed-Book RAG):** The global system prompt enforces a strict "Closed-Book" rule. The LLM is instructed to gracefully decline queries outside the retrieved context rather than hallucinate general financial knowledge.
2. **Implicit Scratchpad Pattern (Generation-Time CoT):**
   *The Problem:* Forcing an LLM to output a `correct_answer` JSON key *before* performing the math leads to "premature guessing" and severe algebra hallucinations.
   *The Solution:* We injected a mandatory `calculation_scratchpad` field at the very top of our Pydantic schemas. This forces the LLM to write out the full mathematical proof *internally* before it is allowed to populate the final options. The frontend explicitly hides this scratchpad from the UI.
3. **Split Evaluation Prompts:**
   The `evaluate_quiz_feedback` prompt splits logic. For wrong answers, it applies a strict "Socratic Override," forbidding the LLM from outputting the final calculated percentage, ensuring the learner must still perform the final step themselves.

---

## 5. What I'd Do Next (Given More Time)

While this PoC demonstrates a robust pedagogical framework, scaling it to a production environment would require:

1. **Enterprise Multi-Modal Parsing:** Transitioning to an enterprise OCR solution (like LlamaParse or Unstructured.io) to perfectly extract nested tabular data and financial charts from the CFA curriculum.
2. **GraphRAG Integration:** Upgrading from standard semantic ChromaDB similarity to entity-based Knowledge Graphs (GraphRAG) to draw cross-chapter connections.
3. **Persistent Analytics (PostgreSQL):** Storing user performance and LangGraph checkpoint states to identify systemic content gaps based on aggregated learner telemetry.
4. **Progressive Scaffolding & Gating:** Implementing a "Prerequisite Gating" mechanism (e.g., Video -> Flashcards -> Unlock Chat) to solve the "cold start" problem for complex topics, while retaining the current flexible exploration as the endgame state.

---

## 6. Content Sources

* **Primary Textbook:** *Discounted Dividend Valuation.pdf* (Provided by task owner via internal link).
* **Video Content:** Embedded YouTube educational content via iframe.
* **Slide Content:** Dynamically generated via LLM based strictly on the retrieved PDF chunks, rendered locally using `react-markdown` and `rehype-katex`.
