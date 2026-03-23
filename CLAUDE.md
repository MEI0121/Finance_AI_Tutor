# AI Collaboration Directives

This project was developed utilizing AI-assisted coding tools (Cursor / Claude 3.5 Sonnet). Treating AI as a junior engineering pair, the following strict architectural directives were enforced during generation:

1.  **Pedagogy Over Chatbots:** AI outputs must not simply "answer the user." The AI must build finite state machines (LangGraph) to orchestrate teaching, assessment, and remediation cycles.
2.  **Zero Hallucination Tolerance:** When generating prompt templates for the AI Tutor, enforce strict "Closed-Book" rules relying solely on ChromaDB vector retrieval.
3.  **Algorithmic Rigor:** Enforce the "Implicit Scratchpad" pattern in Pydantic schemas to force Generation-Time Chain-of-Thought (CoT) for all financial math equations, preventing premature JSON generation errors.
4.  **Component Modularity:** Next.js frontend code must decouple UI components (PDF Viewer, Video Player, Chat) from the backend API state.