# Pedagogical Design: Finance AI Tutor

This document outlines the pedagogical and architectural decisions behind the Finance AI Tutor. Rather than a reactive "Q&A chatbot," this system implements a **State-Machine-Driven Progressive Scaffolding** architecture to emulate a human tutor's cognitive sequencing.

---

## 1. Lesson Flow (The LangGraph State Machine)

The pedagogical flow is explicitly modeled as a Directed Cyclic Graph (DCG) using `LangGraph`. This ensures the AI maintains strict control over the lesson sequence, preventing the learner from skipping foundational concepts.

**Core Nodes in the State Machine:**
1. **`greeting` & `planning`:** The AI assesses the user's initial intent and generates a structured learning plan (sub-topic breakdown) based on the CFA Reading 22 curriculum.
2. **`micro_teach`:** Delivers bite-sized, RAG-grounded explanations of specific concepts (e.g., the Gordon Growth Model).
3. **`qna` (Interactive Sandbox):** Allows the learner to ask clarifying questions about the current micro-step without advancing the curriculum. Includes **Predictive Follow-up UX** (`---SUGGESTIONS---`) to eliminate "prompt blank paralysis."
4. **`assess`:** Generates dynamic Multiple Choice Questions (MCQs) to verify comprehension before allowing progression.
5. **`remediate`:** Triggered upon an incorrect assessment. Provides Socratic hints.
6. **`feynman`:** The capstone node. Upon completing the chapter, the learner must explain the core concept in their own words to prove mastery.

---

## 2. Adaptive Logic & Remediation

The system's routing logic strictly separates "exploration" from "evaluation".

- **Advancing vs. Remediating:** When in the `assess` node, the AI strictly evaluates the user's answer against a mathematical proof. 
  - *Correct Answer:* The graph advances the state to the next `micro_teach` step, or the final `feynman` node if the lesson is complete.
  - *Wrong Answer:* The graph forcibly routes the user to the `remediate` node.
- **Socratic Diagnostic CoT:** During remediation, the AI does *not* reveal the correct option. It applies a **Diagnostic Chain-of-Thought (CoT)**, stepping through the algebra to help the learner locate their specific mathematical error (e.g., confusing $D_0$ with $D_1$).
- **The "Circuit Breaker" (Edge Case Handling):** To prevent an infinite loop of frustration, the state machine tracks `remediation_attempts`. If `attempts >= 3` on a single concept, the AI triggers a "Circuit Breaker"—it gently breaks the Socratic loop, provides the full step-by-step resolution, and moves the learner forward to maintain momentum.

---

## 3. Content Integration & Multi-Modality

The workspace is divided into a **Static/Generative UI Pane** (Left) and an **Adaptive Tutor Pane** (Right).

- **High-Fidelity RAG Pipeline:** CFA textbooks rely heavily on complex formulas. Standard PDF chunkers destroy semantic formatting. We utilized **PyMuPDF (`fitz`)** with a large `CHUNK_SIZE = 4000` and `CHUNK_OVERLAP = 800`. This allows the LLM to ingest entire multi-step mathematical derivations in a single context window.
- **Dynamic AI Slides & Quizzes:** Content is not hardcoded. The system synthesizes retrieved RAG chunks into Markdown/KaTeX-formatted presentation slides and rigorous MCQs on the fly.
- **Two-Page Spread & Video:** The UI natively embeds YouTube video players and a custom Two-Page Spread PDF reader to emulate a premium desktop learning environment, minimizing context switching.

---

## 4. AI Prompt Design (Combating LLM Hallucinations)

In financial education, a fabricated valuation formula is unacceptable. The prompt engineering strategy focused entirely on **Anti-Hallucination** and **Mathematical Rigor**:

1. **`IRONCLAD_TEMPLATE` (Closed-Book RAG):** The global system prompt enforces a strict "Closed-Book" rule. If the ChromaDB retrieval yields no context for a user's question, the LLM is instructed to gracefully decline rather than hallucinate general financial knowledge.
2. **Implicit Scratchpad Pattern (Generation-Time CoT):**
   *The Problem:* Because LLMs are auto-regressive, forcing them to output a `correct_answer` JSON key *before* performing the math leads to "premature guessing" and severe algebra hallucinations.
   *The Solution:* We injected a mandatory `calculation_scratchpad` field at the very top of our Pydantic schemas. This forces the LLM to write out the full mathematical proof *internally* before it is allowed to populate the final options. The frontend explicitly hides this scratchpad from the UI, using it purely as an internal cognitive anchor for the LLM.
3. **Split Evaluation Prompts:**
   The `evaluate_quiz_feedback` prompt splits logic. For wrong answers, it applies a strict "Socratic Override," forbidding the LLM from outputting the final calculated percentage, ensuring the learner must still perform the final step themselves.

---

## 5. What I'd Do Next (Given More Time)

While this PoC demonstrates a robust pedagogical framework, scaling it to a production environment would require:

1. **Enterprise Multi-Modal Parsing:** Transitioning from PyMuPDF to an enterprise OCR solution (like LlamaParse or Unstructured.io) to perfectly extract nested tabular data and financial charts from the CFA curriculum.
2. **GraphRAG Integration:** Upgrading from standard semantic ChromaDB similarity to entity-based Knowledge Graphs (GraphRAG). This would allow the AI Tutor to draw cross-chapter connections (e.g., linking Reading 22 DDMs to Reading 24 Free Cash Flow models).
3. **Persistent Analytics (PostgreSQL):** Storing user performance, frequent misconceptions, and LangGraph checkpoint states in a managed database. This would allow curriculum designers to identify systemic content gaps based on aggregated learner telemetry.