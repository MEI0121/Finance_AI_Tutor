"""LLM-backed generation for slide decks and quiz sets (RAG-grounded)."""

import os
from typing import Tuple

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import ValidationError

from backend.schemas import QuizSet, SlideDeck
from backend.tutor_graph import retrieve_context

load_dotenv()

# Deeper retrieval for generative UI (V12)
RAG_TOP_K = 10

_SUB_CHAPTER_FOCUS = {
    "single": "single-stage Gordon growth DDM, constant growth, V0 = D1 / (r - g), convergence condition r > g",
    "multi": "multi-stage dividend discount models, explicit forecast horizon, transition to mature growth",
    "terminal": "terminal value, perpetual growth in the final stage, consistency with long-run sustainable g",
}


def _llm() -> ChatOpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key == "":
        raise ValueError("OPENAI_API_KEY is not set")
    return ChatOpenAI(model="gpt-4o", temperature=0.3)


def _ironclad_generative_system(topic: str, retrieved_context: str) -> str:
    return f"""You are a strict, content-agnostic Finance AI Tutor generating structured teaching materials.
CURRENT TOPIC: {topic}
REFERENCE KNOWLEDGE (retrieved textbook segments):
{retrieved_context if retrieved_context.strip() else "(empty — if empty, say so briefly; do not fabricate issuer-specific facts.)"}

CRITICAL RULES:
1. ZERO HALLUCINATION: When REFERENCE KNOWLEDGE is non-empty, base every slide and quiz item STRICTLY on it. Quote or paraphrase technical detail, definitions, and numeric examples that appear in the reference.
2. QUIZ / PLAIN TEXT FIELDS: For quiz questions, options, and explanations, use readable plain text (you may write formulas inline as plain text such as V0 = D1 / (r - g)). Do not use LaTeX $ delimiters in those quiz strings.
3. Prefer depth over brevity where the schema allows (explanations, slide content_md).
"""


def generate_slide_deck(topic: str) -> Tuple[SlideDeck | None, str | None]:
    """
    Returns (deck, error). On success error is None.
    """
    try:
        query = f"{topic} discounted dividend model DDM Gordon growth valuation equity intrinsic value"
        ctx = retrieve_context(query, k=RAG_TOP_K)
        if not ctx.strip():
            return (
                None,
                "No reference text found in the knowledge base. Ingest your curriculum PDF into ChromaDB, then retry.",
            )

        system = _ironclad_generative_system(topic, ctx)
        system += """

SYSTEM: You are a rigorous CFA Level 2 Professor. The user provided a textbook segment. You MUST extract deep, highly technical details. Do not summarize superficially. You MUST generate at least 8 to 12 slides. Include exact formulas, specific data examples from the text, and in-depth conceptual breakdowns.

SLIDE content_md (all slide types): Write comprehensive, highly detailed body text using standard GitHub-Flavored Markdown.
- Use Markdown tables whenever comparing numbers, scenarios, or definitions.
- Use $...$ for inline LaTeX math and $$...$$ for display (block) equations.
- Use headings (##, ###), bullet or numbered lists, and bold emphasis for structure.
- Every slide MUST fill content_md with substantive material grounded in REFERENCE KNOWLEDGE.

OUTPUT FORMAT: Produce a SlideDeck with between 8 and 12 slides total. Use only these slide types: "concept", "mini-quiz", "feynman".
- Most slides should be type "concept" with an optional short title plus rich content_md (dense paragraphs, tables, and LaTeX as needed).
- Include at least two "mini-quiz" slides. Each must have rich content_md (context/setup may repeat key formulas), plus question, exactly four options in order A-D, and correct_answer matching the reference.
- Include at least one "feynman" slide with optional title, content_md explaining what to teach back, and a reflection prompt field.
- Order logically: build concepts, place mini-quizzes after related concepts, use feynman toward the end.
- Ground every formula, table cell, and example in the REFERENCE KNOWLEDGE when possible."""

        llm = _llm()
        structured = llm.with_structured_output(SlideDeck)
        deck: SlideDeck = structured.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content="Generate the SlideDeck now. Meet the slide count and rigor requirements exactly."
                ),
            ]
        )
        if not deck.slides or len(deck.slides) < 8:
            return None, "Model returned an incomplete slide deck (expected 8-12 slides)."
        return deck, None
    except ValidationError as e:
        return None, f"Structured output validation failed: {e!s}"
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Generation failed: {e!s}"


def generate_quiz_set(topic: str, sub_chapter: str) -> Tuple[QuizSet | None, str | None]:
    try:
        focus = _SUB_CHAPTER_FOCUS.get(sub_chapter, _SUB_CHAPTER_FOCUS["single"])
        query = f"{topic} {focus} DDM quiz multiple choice calculation intrinsic value"
        ctx = retrieve_context(query, k=RAG_TOP_K)
        if not ctx.strip():
            return (
                None,
                "No reference text found in the knowledge base. Ingest your curriculum PDF into ChromaDB, then retry.",
            )

        system = _ironclad_generative_system(topic, ctx)
        system += f"""

SYSTEM: Generate 5 highly difficult, CFA-style multiple-choice questions. Include calculation questions if formulas are present in the reference. The explanations must be exhaustively detailed, using Socratic reasoning. Tie each question to specific ideas or numbers in the reference when available.

SUB-CHAPTER FOCUS: {focus}

OUTPUT FORMAT: Exactly 5 QuizQuestion objects. Each has question, four options (A-D order), correct_answer (A/B/C/D), and a long, detailed explanation that teaches."""

        llm = _llm()
        structured = llm.with_structured_output(QuizSet)
        qs: QuizSet = structured.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content="Generate the QuizSet of exactly 5 MCQs now."
                ),
            ]
        )
        if not qs.questions or len(qs.questions) != 5:
            return None, "Model returned an incomplete quiz set (expected 5 questions)."
        return qs, None
    except ValidationError as e:
        return None, f"Structured output validation failed: {e!s}"
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Generation failed: {e!s}"


def evaluate_quiz_feedback(
    question_text: str,
    selected_option: str,
    is_correct: bool,
) -> tuple[str | None, str | None]:
    """
    Returns (feedback_markdown, error). Feedback is GFM + optional $ / $$ KaTeX.
    """
    try:
        q = (question_text or "").strip()
        sel = (selected_option or "").strip()
        if not q or not sel:
            return None, "question_text and selected_option are required."

        query = f"{q}\n{sel}"
        ctx = retrieve_context(query, k=RAG_TOP_K)

        base_rules = f"""You are a strict, content-agnostic Finance AI Tutor.
REFERENCE KNOWLEDGE (retrieved textbook segments):
{ctx if ctx.strip() else "(empty — reason from general finance pedagogy only; do not invent issuer-specific facts.)"}

FORMATTING: Respond in GitHub-Flavored Markdown only. Use **bold** and lists for structure.
Use $...$ for inline LaTeX math and $$...$$ for display equations when formulas help.
Do not wrap the entire answer in a code fence."""

        if is_correct:
            system = (
                base_rules
                + """

MODE: CORRECT ANSWER — EXPLANATION
The learner chose the correct option. Your job:
1. Give a clear, highly detailed, step-by-step breakdown of WHY this answer is correct.
2. Connect each step to definitions or logic from REFERENCE KNOWLEDGE when it supports a claim.
3. When the topic involves formulas, show the relevant relationships explicitly using LaTeX in Markdown.
4. You may name the correct reasoning path fully (the learner already succeeded).
5. Do not be terse — aim for teaching depth suitable for a motivated student."""
            )
            human = (
                f"Question:\n{q}\n\nChosen option (correct):\n{sel}\n\n"
                "Write the markdown explanation now."
            )
        else:
            system = (
                base_rules
                + """

MODE: SOCRATIC REMEDIATION (WRONG ANSWER)
The learner chose an incorrect option. Your job:
1. Act as a Socratic tutor. Be warm and concise but rigorous.
2. You MUST NOT reveal which option is correct (no letter, no paraphrase of the right choice that identifies it).
3. You MUST NOT perform the final numerical or symbolic answer to the quiz question for them.
4. Briefly analyze why the chosen option is logically or conceptually flawed (misapplied formula, wrong sign, wrong cash-flow timing, etc.) using REFERENCE KNOWLEDGE only when it helps.
5. End with exactly one focused Socratic question that nudges them to rethink the key idea — not a giveaway."""
            )
            human = (
                f"Question:\n{q}\n\nChosen option (incorrect):\n{sel}\n\n"
                "Write the markdown Socratic feedback now."
            )

        llm = _llm()
        reply = llm.invoke(
            [SystemMessage(content=system), HumanMessage(content=human)]
        )
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        text = text.strip()
        if not text:
            return None, "Model returned empty feedback."
        return text, None
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Evaluation failed: {e!s}"
