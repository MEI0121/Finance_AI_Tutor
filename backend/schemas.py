"""Structured output schemas for generative UI (slides, quizzes)."""

from typing import Literal

from pydantic import BaseModel, Field


class Slide(BaseModel):
    """One slide; fields used depend on ``type``."""

    type: Literal["concept", "mini-quiz", "feynman"]
    title: str | None = None
    content_md: str = Field(
        ...,
        description=(
            "Full slide body in GitHub-Flavored Markdown: headings, tables, lists. "
            "Use $ for inline LaTeX and $$ for block LaTeX. Be comprehensive and detailed."
        ),
    )
    question: str | None = None
    options: list[str] | None = Field(
        default=None,
        description="For mini-quiz: exactly four option strings in order A through D.",
    )
    correct_answer: Literal["A", "B", "C", "D"] | None = None
    prompt: str | None = Field(
        default=None,
        description="For feynman: self-reflection prompt for the learner.",
    )


class SlideDeck(BaseModel):
    slides: list[Slide] = Field(
        ...,
        min_length=8,
        max_length=12,
        description="8-12 slides: dense concept slides, mini-quizzes, feynman checkpoints.",
    )


class QuizQuestion(BaseModel):
    question: str
    options: list[str] = Field(
        ...,
        min_length=4,
        max_length=4,
        description="Four strings: option A, B, C, D in order.",
    )
    correct_answer: Literal["A", "B", "C", "D"]
    explanation: str = Field(
        ...,
        description="Exhaustively detailed explanation with Socratic reasoning, grounded in reference.",
    )


class QuizSet(BaseModel):
    questions: list[QuizQuestion] = Field(
        ...,
        min_length=5,
        max_length=5,
        description="Exactly five CFA-style multiple-choice questions.",
    )


class GenerateSlidesRequest(BaseModel):
    topic: str = Field(
        default="Discounted Dividend Valuation",
        description="Curriculum topic (e.g. DDM).",
    )


class GenerateQuizRequest(BaseModel):
    topic: str = Field(
        default="Discounted Dividend Valuation",
        description="Curriculum topic.",
    )
    sub_chapter: Literal["single", "multi", "terminal"] = Field(
        default="single",
        description="Sub-chapter focus for question targeting.",
    )


class GenerativeErrorResponse(BaseModel):
    error: str
    detail: str | None = None


class EvaluateQuizRequest(BaseModel):
    question_text: str = Field(..., description="Full quiz question stem.")
    selected_option: str = Field(
        ...,
        description="The option the learner chose (e.g. label + text).",
    )
    is_correct: bool = Field(
        ...,
        description="Whether that option matches the keyed correct answer.",
    )
