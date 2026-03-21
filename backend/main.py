from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from backend.tutor_graph import run_tutor_flow

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
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
