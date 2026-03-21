# Pedagogical Design: Finance AI Tutor

## Lesson Flow
The tutor follows a state-machine-driven "Progressive Scaffolding" flow, completely distinct from a reactive search engine.
1. **The Intuition Hook (Node A):** Starts with a real-world business scenario (e.g., valuing a cash-generating asset) to anchor the concept before introducing formulas.
2. **Multi-modal Delivery (Node B):** Presents the core concept using a split-view UI (Blackboard + Tutor Chat), embedding a relevant YouTube video and extracting text strictly from the provided PDF.
3. **Diagnostic Assessment (Node C):** Generates a dynamic quiz. The options are deliberately designed to catch classic financial misconceptions (e.g., confusing D0 with D1).
4. **Socratic Remediation (Node D):** If the user chooses a misconception trap, the AI retrieves the exact PDF context and guides the user step-by-step without giving the final answer.
5. **The Feynman Checkpoint (Node E):** Upon solving the calculation, the user must explain the concept in plain English to unlock the next module.

## Adaptive Logic
- **Misconception-Based Routing:** The system doesn't just evaluate "Right/Wrong". It identifies *why* the user is wrong based on the specific distractor chosen, adapting the Socratic prompt accordingly.
- **The "Frustration Circuit Breaker":** To prevent infinite, frustrating loops, the state machine tracks `remediation_attempts`. If `attempts >= 3`, the AI breaks the Socratic loop, provides a step-by-step resolution, and offers encouragement before moving on.

## Content Integration
- **Split-View UI:** A Next.js frontend divides the screen. The left acts as a "Dynamic Blackboard" rendering YouTube iframes and PDF diagrams, while the right maintains the conversation flow.
- **RAG via ChromaDB:** The textbook PDF is chunked and stored locally. The AI strictly retrieves content from this database to prevent financial hallucination. 

## AI Prompt Design
- **Content-Agnostic Engine:** The core System Prompt contains NO hardcoded financial terms. It relies entirely on Dependency Injection. 
- **Prompt Structure:** `[System Rules (Socratic method, strict boundary)]` + `[Current Lesson JSON Payload]` + `[ChromaDB Retrieved Context]` + `[User Input]`.
- This ensures the engine can pivot to teaching "Fixed Income" tomorrow just by swapping the PDF and JSON payload.

## What I'd Do Next
Given more than 16 hours, I would implement:
1. **Silent Telemetry:** Log chosen misconceptions into an `analytics.csv` to provide the platform's curriculum team with data on common student pitfalls.
2. **3D Avatar Integration:** Integrate HeyGen/ElevenLabs API to provide a real-time, emotionally responsive voice and video avatar for the tutor.
3. **Production Database:** Migrate the lightweight `user_state.json` to a managed PostgreSQL instance for persistent, cross-device user sessions.