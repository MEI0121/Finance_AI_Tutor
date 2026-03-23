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

# V15.1: slide decks need broad context—retrieve many chunks for exhaustive coverage
RAG_SLIDE_DECK_K = 30

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


def _llm_slide_deck() -> ChatOpenAI:
    """Higher output budget for large structured SlideDeck (6–15 slides)."""
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key == "":
        raise ValueError("OPENAI_API_KEY is not set")
    return ChatOpenAI(
        model="gpt-4o",
        temperature=0.25,
        max_tokens=16384,
    )


def _ironclad_generative_system(topic: str, retrieved_context: str) -> str:
    return f"""You are a strict, content-agnostic Finance AI Tutor generating structured teaching materials.
CURRENT TOPIC: {topic}
REFERENCE KNOWLEDGE (retrieved textbook segments):
{retrieved_context if retrieved_context.strip() else "(empty — if empty, say so briefly; do not fabricate issuer-specific facts.)"}

CRITICAL ANTI-HALLUCINATION RULE: You are a strict closed-book AI. You MUST rely EXCLUSIVELY on the retrieved context provided below. If the context does not contain enough information to generate the requested content, you MUST scale back your output. DO NOT invent formulas. DO NOT use external knowledge to fill in gaps. Stick strictly to the boundaries of the provided text.

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
        query = (
            f"{topic} discounted dividend model DDM Gordon growth valuation "
            "equity intrinsic value multi-stage terminal value assumptions limitations "
            "applications examples"
        )
        ctx = retrieve_context(query, k=RAG_SLIDE_DECK_K)
        if not ctx.strip():
            return (
                None,
                "No reference text found in the knowledge base. Ingest your curriculum PDF into ChromaDB, then retry.",
            )

        system = _ironclad_generative_system(topic, ctx)
        system += """

PERSONA: You are a pedantic, detail-obsessed senior CFA Level II / Level III instructor. **Summarization is FORBIDDEN.** Shallow overviews, “in summary,” and lazy condensation are not allowed.

CRITICAL RULES FOR GENERATION:
1. DO NOT REPEAT CONTENT. Every slide MUST contain unique, non-redundant knowledge.
2. If you have exhaustively covered all the mathematical formulas, concepts, and examples in the provided context, you MUST STOP generating slides.
3. DO NOT pad the slide deck with duplicates or paraphrased versions of earlier slides just to reach a higher slide count.
4. Your progression must be logical: Introduction -> Deep Concepts & Formulas -> Socratic Mini-Quiz -> Edge Cases/Limitations -> Feynman Summary.

MISSION — COMPREHENSIVE EXTRACTION (“carpet-bombing”):
You MUST perform an **exhaustive, detail-by-detail extraction** of *all* distinct knowledge points present in REFERENCE KNOWLEDGE: every definition, assumption, condition, edge case, formula, sign convention, timing nuance, and **worked or numeric example** the text provides. If the reference presents multiple phrasings of the same idea, reconcile them rigorously. If something is absent from the reference, state that gap in one sentence—never invent facts.

CONCEPT SLIDES (type "concept") — DEEP DIVE ONLY:
- Create **highly detailed** concept slides. **Do not fear content density**—each slide should pack multiple paragraphs, sub-headings, and technical substance.
- **Tables:** Use GitHub-Flavored Markdown **tables** to contrast **every** method, stage, or scenario the reference distinguishes (e.g., single-stage vs. multi-stage DDM, forecast vs. terminal phase). Where the text compares approaches, **table it**.
- **KaTeX:** Use $...$ inline and $$...$$ for **every** formula, rearrangement, and derivation logic the reference supports—including limits, prerequisites (e.g., $r > g$), and algebraic steps.
- **Limitations, prerequisites, real-world application:** As in CFA readings, dedicate slides (or substantial sections) to **model limitations**, **required inputs and sensitivities**, **when the model misleads practitioners**, and **real-world / institutional applications** wherever the reference supplies them.
- Prefer dense prose with ## / ###; **ban casual bullet lists** except for genuinely distinct enumerated factors (e.g., explicit assumption lists).

MINI-QUIZ SLIDES (type "mini-quiz"):
- At least **two** mini-quiz slides distributed through the deck. content_md = dense stem; question + four options (plain text in options—no LaTeX in option strings). correct_answer from REFERENCE KNOWLEDGE only.
- CRITICAL WORKFLOW FOR MATH: Fill `calculation_scratchpad` with step-by-step work FIRST. Only AFTER the exact final value is in the scratchpad, set `options` and `correct_answer`. `correct_answer` MUST match the scratchpad result. For non-calculation mini-quizzes, use a brief rationale in `calculation_scratchpad`.

FEYNMAN SLIDE (type "feynman"):
- Exactly **one** closing slide: content_md frames mastery; prompt = sharp teach-back question.

DECK SIZE: Produce **6 to 15 slides inclusive** (stop earlier if the reference is exhausted—never duplicate to fill a quota). The majority must be type "concept". Order: logical progression, mini-quizzes after related concept clusters, **feynman last**.

OUTPUT: SlideDeck JSON only, satisfying slide count and types."""

        llm = _llm_slide_deck()
        structured = llm.with_structured_output(SlideDeck)
        deck: SlideDeck = structured.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content="Generate the SlideDeck now: 6-15 slides max, exhaustive but non-redundant extraction from the reference, "
                    "no summarization—pedantic CFA Academy depth, full type mix, feynman last. Stop when material is exhausted. "
                    "For mini-quiz slides, complete calculation_scratchpad before options and correct_answer."
                ),
            ]
        )
        if not deck.slides or len(deck.slides) < 6:
            return None, "Model returned an incomplete slide deck (expected 6-15 slides)."
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

MATH ACCURACY RULE: If generating a calculation question, you MUST internally calculate the exact answer step-by-step BEFORE defining the options. The designated `correct_answer` letter MUST identify your exact mathematical result. DO NOT round arbitrarily (e.g., do not round 11.86% to 12%). Your options must include the precise calculated value.

CRITICAL WORKFLOW FOR MATH: You must use the `calculation_scratchpad` field to solve the math step-by-step FIRST. Only AFTER you have the exact final number in your scratchpad, you may populate the `options` and `correct_answer` fields. The `correct_answer` MUST identically match the result from your scratchpad (same meaning as `correct_option` in plain language).

SYSTEM: Generate 5 highly difficult, CFA-style multiple-choice questions. Include calculation questions if formulas are present in the reference. The explanations must be exhaustively detailed, using Socratic reasoning. Tie each question to specific ideas or numbers in the reference when available.

SUB-CHAPTER FOCUS: {focus}

OUTPUT FORMAT: Exactly 5 QuizQuestion objects, fields in this order: `calculation_scratchpad` (required first—full work for calculation items, brief rationale for concept-only items), then `question`, four `options` (A-D order), `correct_answer` (A/B/C/D), and `explanation`. The `correct_answer` letter MUST identify the option whose numeric value equals your exact calculation from the scratchpad and MATH ACCURACY RULE."""

        llm = _llm()
        structured = llm.with_structured_output(QuizSet)
        qs: QuizSet = structured.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(
                    content="Generate the QuizSet of exactly 5 MCQs now. Fill calculation_scratchpad before options and correct_answer on every row."
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
{ctx if ctx.strip() else "(empty — scale back: do not invent formulas or issuer-specific facts; ground only in the question text.)"}

CRITICAL ANTI-HALLUCINATION RULE: You are a strict closed-book AI. You MUST rely EXCLUSIVELY on the retrieved context provided below. If the context does not contain enough information to generate the requested content, you MUST scale back your output. DO NOT invent formulas. DO NOT use external knowledge to fill in gaps. Stick strictly to the boundaries of the provided text.

CRITICAL MATHEMATICAL RULE: If the question involves calculations, you MUST use 'Chain of Thought' reasoning.
1. Write down the exact base formula (e.g., P0 = D0(1+g)/(r-g)).
2. Substitute the known variables strictly based on the provided context.
3. Perform the algebraic steps ONE BY ONE (e.g., cross-multiplication, expanding brackets, isolating the variable).
4. DO NOT skip steps. DO NOT use simplified approximation formulas unless explicitly stated in the textbook.
5. Your final calculated math MUST perfectly logically lead to the correct option. If it doesn't match, re-calculate your algebraic steps. Do not invent excuses.
6. ABSOLUTELY NO ARBITRARY ROUNDING: Do not invent fake rounding rules (e.g., rounding 11.86% to 12%) just to force your math to match a designated correct option. Trust your exact algebraic calculation. If your exact math yields 11.86%, then the answer is exactly 11.86%.

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
3. When the topic involves formulas, show the relevant relationships explicitly using LaTeX in Markdown. Follow CRITICAL MATHEMATICAL RULE above: your worked arithmetic must be correct and must land on the chosen option without arbitrary rounding (item 6); if your numbers drift, fix the algebra before publishing.
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
For this mode ONLY: Do not use item 5 of CRITICAL MATHEMATICAL RULE to derive or state the correct option's final value (that would spoil the quiz). Still apply items 1–4 and 6 rigorously for any partial algebra you show—no skipped steps, no wrong formulas, no fake rounding, no excuses if intermediate arithmetic is inconsistent.

The learner chose an incorrect option. Your job:
1. Act as a Socratic tutor. Be warm and concise but rigorous.
2. You MUST NOT reveal which option is correct (no letter, no paraphrase of the right choice that identifies it).
3. You MUST NOT perform the final numerical or symbolic answer to the quiz question for them.
4. Briefly analyze why the chosen option is logically or conceptually flawed (misapplied formula, wrong sign, wrong cash-flow timing, etc.) using REFERENCE KNOWLEDGE only when it helps. If you discuss any algebra or numbers, follow CRITICAL MATHEMATICAL RULE (show formulas and steps explicitly, no skipped shortcuts, no bogus arithmetic) — but only for intermediate checks or to expose an error in their line of reasoning; do not derive the full solution that would spoil the quiz.
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
