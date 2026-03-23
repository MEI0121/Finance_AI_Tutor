from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.generative_content import (
    evaluate_quiz_feedback,
    generate_quiz_set,
    generate_slide_deck,
)
from backend.schemas import (
    EvaluateQuizRequest,
    GenerateQuizRequest,
    GenerateSlidesRequest,
)
from backend.tutor_graph import run_tutor_flow

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok"}


class ChatRequest(BaseModel):
    message: str = Field(default="", description="User message for this turn.")
    state: dict = Field(default_factory=dict, description="Tutor state from the previous response.")


@app.post("/chat")
def chat(payload: ChatRequest):
    outcome = run_tutor_flow(payload.message, payload.state)
    if outcome.get("error") == "missing_api_key":
        return outcome
    return {
        "state": outcome.get("state"),
        "reply": outcome.get("reply"),
    }


@app.post("/api/generate_slides")
def api_generate_slides(payload: GenerateSlidesRequest):
    deck, err = generate_slide_deck(payload.topic.strip() or "Discounted Dividend Valuation")
    if err:
        return {"ok": False, "error": err, "detail": None, "deck": None}
    return {"ok": True, "error": None, "detail": None, "deck": deck.model_dump()}


@app.post("/api/generate_quiz")
def api_generate_quiz(payload: GenerateQuizRequest):
    topic = payload.topic.strip() or "Discounted Dividend Valuation"
    quiz, err = generate_quiz_set(topic, payload.sub_chapter)
    if err:
        return {"ok": False, "error": err, "detail": None, "quiz": None}
    return {"ok": True, "error": None, "detail": None, "quiz": quiz.model_dump()}


@app.post("/api/evaluate_quiz")
def api_evaluate_quiz(payload: EvaluateQuizRequest):
    md, err = evaluate_quiz_feedback(
        payload.question_text.strip(),
        payload.selected_option.strip(),
        payload.is_correct,
    )
    if err:
        return {"ok": False, "error": err, "feedback_md": None}
    return {"ok": True, "error": None, "feedback_md": md}
