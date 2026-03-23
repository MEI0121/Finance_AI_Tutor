import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Literal, TypedDict

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

load_dotenv()

logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BACKEND_DIR / "chroma_db"
COLLECTION_NAME = "knowledge_base"

# ChromaDB 1.x shares one System per persist path; creating a new PersistentClient on
# every query can drive refcount/stop logic incorrectly and leave RustBindingsAPI
# without initialized bindings (tenant validation then fails with AttributeError).
_chroma_client: object | None = None
_chroma_lock = threading.Lock()

# Default embed for the MVP curriculum topic (Discounted Dividend Valuation); overridable via state.
DEFAULT_CURRENT_VIDEO_URL = "https://www.youtube.com/embed/-mQJ7a4U9Z8?si=xDlg2zON0SiUOP7-"

SUGGESTIONS_MARKER = "---SUGGESTIONS---"


def _suggestion_footer(*questions: str) -> str:
    """Append programmatic follow-up pills (same marker/format as LLM PREDICTIVE FOLLOW-UP)."""
    lines = "\n".join(f"* {q.strip()}" for q in questions if q.strip())
    if not lines:
        return ""
    return f"\n\n{SUGGESTIONS_MARKER}\n{lines}"


GENERIC_SESSION_GREETING = (
    "Hello! I'm your AI tutor for this session. I'm here to help you understand "
    "today's topic. Feel free to ask any specific questions, explore an idea, or let me know "
    "if you'd like a full walkthrough. No pressure—I'm here to help!"
    + _suggestion_footer(
        "Can you walk me through the full topic from the beginning?",
        "What is the single most important idea I should understand first?",
        "I have a specific question about a formula or example—can we dig in?",
    )
)

IRONCLAD_TEMPLATE = """You are a strict, content-agnostic Finance AI Tutor.
CURRENT TOPIC: {payload_topic}
REFERENCE KNOWLEDGE: {retrieved_context}

CRITICAL ANTI-HALLUCINATION RULE: You are a strict closed-book AI. You MUST rely EXCLUSIVELY on the retrieved context provided below. If the context does not contain enough information to generate the requested content, you MUST scale back your output. DO NOT invent formulas. DO NOT use external knowledge to fill in gaps. Stick strictly to the boundaries of the provided text.

CRITICAL RULES:
1. ZERO HALLUCINATION: You must base your teachings, explanations, and quizzes STRICTLY and ONLY on the {retrieved_context}. Do not use your pre-trained internet knowledge to invent formulas or facts.
2. OUT-OF-DOMAIN REJECTION: If the user asks about ANYTHING unrelated to the CURRENT TOPIC (e.g., weather, coding, unrelated stocks like TSLA, or general advice), you MUST immediately intercept and reply: "That is outside the scope of our current lesson on [CURRENT TOPIC]. Let's refocus on the material." Exception: requests to TRANSLATE course materials or text drawn from REFERENCE KNOWLEDGE (e.g., into Chinese or another language) are ALWAYS valid, on-topic pedagogical requests. Fulfill them using the quoted or retrieved material; do not reject translation requests that contain lesson-related text.
3. NO SPOILERS: During the Socratic Remediation phase, NEVER give the final mathematical answer directly. Provide a hint based on the {retrieved_context} and ask the user to try again.
4. ABSOLUTELY NO LaTeX formatting. DO NOT use \\[, \\], \\(, or \\). Format all formulas using simple plain text, e.g., Vt = D / (r - g).
5. EXPLAIN LIKE I'M 5: Whenever you introduce a financial concept or formula, you MUST pair it with a simple, real-world analogy (e.g., renting an apartment, a lemonade stand, splitting a pizza) so it feels intuitive. Avoid academic jargon unless you briefly define it in plain words.

TONE: Be warm, patient, and human. Sound like a friendly tutor in conversation, not a textbook. Hard cap: every assistant message to the learner MUST be at most 100 words unless a node explicitly overrides for internal grading only. Never nag the learner to take a quiz or test unless they explicitly ask for one.

PREDICTIVE FOLLOW-UP RULE: At the very end of EVERY response you generate, you MUST provide 2 or 3 highly relevant follow-up questions the user might want to ask next to deepen their understanding.
You MUST format them EXACTLY like this at the very bottom of your markdown output:
---SUGGESTIONS---
* [Follow-up Question 1]
* [Follow-up Question 2]

Replace the bracketed placeholders with real, specific questions (add a third line * ... if it helps). Nothing may appear after the last suggestion bullet. Word caps in the node instructions apply only to the main body above ---SUGGESTIONS---; the suggestion block does not count toward those caps."""

PREDICTIVE_FOLLOW_UP_RULE = """
PREDICTIVE FOLLOW-UP RULE: At the very end of EVERY response you generate, you MUST provide 2 or 3 highly relevant follow-up questions the user might want to ask next to deepen their understanding.
You MUST format them EXACTLY like this at the very bottom of your markdown output:
---SUGGESTIONS---
* [Follow-up Question 1]
* [Follow-up Question 2]

Replace the bracketed placeholders with real, specific questions (add a third line * ... if it helps). Nothing may appear after the last suggestion bullet. Word caps in the node instructions apply only to the main body above ---SUGGESTIONS---; the suggestion block does not count toward those caps.
""".strip()


class TutorState(TypedDict, total=False):
    messages: list
    current_node: str
    current_topic: str
    teaching_plan: list[str]
    current_step_index: int
    is_off_topic: bool
    remediation_attempts: int
    concept_mastered: bool
    quiz_asked: bool
    assessment_result: str
    circuit_breaker_triggered: bool
    active_quiz: str
    greeting_shown: bool
    awaiting_micro_reply: bool
    greeting_next: str
    route_after_assess_correct: str
    awaiting_plan_confirmation: bool
    is_adhoc_session: bool
    current_video_url: str


class MicroRouteIntent(BaseModel):
    route: Literal["qna", "assess"] = Field(
        description=(
            'Use "qna" for any teaching conversation: questions, confusion, acknowledgments like thanks or got it, '
            'follow-ups, or anything that is not an explicit request to be quizzed. '
            'Use "assess" ONLY when the learner explicitly asks for a quiz, test, practice question, or MCQ, '
            "or when a quiz is already on screen and they reply with a single letter A-D."
        )
    )


class PlanStartIntent(BaseModel):
    route: Literal["micro_teach", "qna"] = Field(
        description=(
            'Use "micro_teach" if they want to BEGIN the plan and do step 1 now: yes, let us go, take me through the course, '
            "walk me through it, go through the whole course, I'm ready, sounds good, start step 1, or similar agreement. "
            'Use "qna" only if they ask a clarifying question, want to change the plan, say not yet, or express doubt.'
        )
    )


class CurriculumPathIntent(BaseModel):
    route: Literal["planning", "adhoc_qna"] = Field(
        description=(
            'Use "adhoc_qna" when the learner asks a direct question, wants a definition, formula, or one concept, '
            'or is exploring something specific. Use "planning" ONLY when they clearly want a full guided path '
            'from scratch (e.g. teach me everything, learn from scratch, walk me through the whole topic, structured beginner course).'
        )
    )


def _get_persistent_chroma_client():
    """Return a single PersistentClient for this process (thread-safe lazy init)."""
    global _chroma_client
    with _chroma_lock:
        if _chroma_client is None:
            _chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        return _chroma_client


def _get_collection():
    embedding_fn = DefaultEmbeddingFunction()
    try:
        client = _get_persistent_chroma_client()
        names = [c.name for c in client.list_collections()]
    except Exception as exc:
        logger.warning("ChromaDB client unavailable: %s", exc)
        return None
    if COLLECTION_NAME not in names:
        return None
    try:
        return client.get_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    except Exception as exc:
        logger.warning("ChromaDB get_collection failed: %s", exc)
        return None


def retrieve_context(query: str, k: int = 5) -> str:
    collection = _get_collection()
    if collection is None:
        return ""
    result = collection.query(query_texts=[query], n_results=k)
    documents = result.get("documents")
    if documents is None:
        return ""
    first = documents[0]
    if not first:
        return ""
    return "\n\n".join(first)


def _messages_from_state(raw: list) -> list:
    out: list = []
    index = 0
    while index < len(raw):
        item = raw[index]
        role = item.get("role", "")
        content = item.get("content", "")
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
        index += 1
    return out


def _append_message(raw: list | None, role: str, content: str) -> list:
    base = list(raw) if raw else []
    base.append({"role": role, "content": content})
    return base


def _last_user_text(raw: list | None) -> str:
    if not raw:
        return ""
    idx = len(raw) - 1
    while idx >= 0:
        m = raw[idx]
        if m.get("role") == "user":
            return str(m.get("content", ""))
        idx -= 1
    return ""


def _last_assistant_text(raw: list | None) -> str:
    if not raw:
        return ""
    idx = len(raw) - 1
    while idx >= 0:
        m = raw[idx]
        if m.get("role") == "assistant":
            return str(m.get("content", ""))
        idx -= 1
    return ""


def _llm() -> ChatOpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key == "":
        raise ValueError("OPENAI_API_KEY is not set")
    return ChatOpenAI(model="gpt-4o", temperature=0.2)


def _base_prompt(topic: str, context: str) -> str:
    return IRONCLAD_TEMPLATE.format(
        payload_topic=topic,
        retrieved_context=context,
    )


def _learner_base_prompt(topic: str, context: str) -> str:
    """Alias kept for compatibility; base prompt now carries predictive rule globally."""
    return _base_prompt(topic, context)


def _is_adhoc_session(state: TutorState) -> bool:
    """Ad-hoc Q&A path: single-step placeholder plan from greeting, not a formal 3-step curriculum."""
    if bool(state.get("is_adhoc_session")):
        return True
    plan = state.get("teaching_plan") or []
    if isinstance(plan, list) and len(plan) == 1 and str(plan[0]).strip() == "Your question":
        return True
    return False


def _reset_session_for_new_learning(state: TutorState) -> TutorState:
    """
    After formal module completion (Feynman pass), allow infinite continuation in the same thread.
    Preserves message history; clears curriculum completion flags and returns to open Q&A.
    """
    out = dict(state)
    out["teaching_plan"] = ["Your question"]
    out["current_step_index"] = 0
    out["concept_mastered"] = False
    out["current_node"] = "micro_teach"
    out["awaiting_micro_reply"] = True
    out["awaiting_plan_confirmation"] = False
    out["quiz_asked"] = False
    out["active_quiz"] = ""
    out["assessment_result"] = ""
    out["route_after_assess_correct"] = ""
    out["remediation_attempts"] = 0
    out["is_adhoc_session"] = True
    out["greeting_shown"] = True
    out["greeting_next"] = ""
    out["circuit_breaker_triggered"] = False
    v = out.get("current_video_url")
    if v is None or str(v).strip() == "":
        out["current_video_url"] = DEFAULT_CURRENT_VIDEO_URL
    else:
        out["current_video_url"] = str(v).strip()
    return out


def _parse_json_array(text: str) -> list[str]:
    cleaned = text.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        index = 0
        while index < len(parts):
            block = parts[index].strip()
            if block.startswith("json"):
                block = block[4:].strip()
            if block.startswith("["):
                cleaned = block
                break
            index += 1
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            out = []
            for item in data:
                if isinstance(item, str) and item.strip():
                    out.append(item.strip())
            if len(out) >= 3:
                return out[:3]
            if len(out) > 0:
                while len(out) < 3:
                    out.append(f"Step {len(out) + 1}")
                return out[:3]
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return ["Concept intro", "Core mechanics", "Application / examples"]


def _normalize_teaching_plan_three(plan: list[str]) -> list[str]:
    """Formal curriculum always uses exactly three micro-steps for pacing."""
    out = list(plan) if plan else []
    while len(out) < 3:
        out.append(f"Step {len(out) + 1}")
    return out[:3]


def _signals_plan_start_agreement(user_text: str) -> bool:
    """Strong signals that the learner wants to begin step 1 (override flaky LLM routing)."""
    if not user_text or not str(user_text).strip():
        return False
    t = user_text.lower().strip()
    if re.match(r"^\s*(yes|yep|yeah|ok|okay|sure|go)\s*[!?.]?\s*$", t):
        return True
    if re.match(r"^let['']s\s+do\s+it\s*[!?.]?\s*$", t):
        return True
    patterns = (
        r"\btake\s+me\s+(through|along)\b",
        r"\bwalk\s+me\s+through\b",
        r"\bgo\s+through\s+the\s+(whole\s+)?(course|plan|steps)\b",
        r"\bthrough\s+the\s+whole\s+course\b",
        r"\bstart\s+step\s*1\b",
        r"\bbegin\s+(the\s+)?(course|plan)\b",
        r"\blet['']?s\s+go\b",
        r"\bsounds\s+good\b",
        r"\bi['']?m\s+ready\b",
        r"\bready\s+to\s+start\b",
        r"\b(full|whole)\s+course\b",
        r"\blead\s+me\s+through\b",
        r"\bguide\s+me\s+through\b",
    )
    index = 0
    while index < len(patterns):
        if re.search(patterns[index], t, re.I):
            return True
        index += 1
    return False


def _classify_greeting_intent(topic: str, user_text: str, context: str) -> str:
    """Return 'domain' or 'off_topic'."""
    if re.search(r"\btranslate\b", user_text, re.I):
        return "domain"
    system = _base_prompt(topic, context) + """

CLASSIFIER (INTERNAL)
The learner replied after your welcome. Decide if they want to learn something aligned with the CURRENT TOPIC and REFERENCE KNOWLEDGE.
INTERNAL OVERRIDE: For this classifier task only, DO NOT output any ---SUGGESTIONS--- block.
Output exactly one line:
INTENT: DOMAIN
or
INTENT: OFF_TOPIC

Use OFF_TOPIC only if they clearly want something unrelated to this lesson or refuse to engage with the topic. Requests to translate lesson text or course materials (any target language) are INTENT: DOMAIN."""
    llm = _llm()
    reply = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content=f"Learner message: {user_text}"),
        ]
    )
    raw = reply.content if isinstance(reply.content, str) else str(reply.content)
    upper = raw.upper()
    if "INTENT: OFF_TOPIC" in upper:
        return "off_topic"
    return "domain"


def _explicit_quiz_request(user_text: str) -> bool:
    """True if the learner clearly asked to be quizzed (not already answering a visible quiz)."""
    if not user_text or not str(user_text).strip():
        return False
    lower = user_text.lower().strip()
    if re.match(r"^\s*[ABCDabcd]\s*$", user_text.strip()):
        return False
    patterns = (
        r"\bquizzes\b",
        r"\bquiz\b",
        r"\btest\s+me\b",
        r"\bgive\s+me\s+a\s+question\b",
        r"\bpractice\s+question\b",
        r"\bmcq\b",
        r"\bmultiple\s+choice\b",
        r"\bask\s+me\s+a\s+question\b",
        r"\bpop\s+quiz\b",
    )
    index = 0
    while index < len(patterns):
        if re.search(patterns[index], lower):
            return True
        index += 1
    return False


def _classify_curriculum_vs_adhoc(topic: str, user_text: str, context: str) -> str:
    """Return 'planning' (full curriculum) or 'adhoc_qna' (direct question / one-off)."""
    system = (
        _base_prompt(topic, context)
        + """

ROUTER (INTERNAL — STRUCTURED OUTPUT)
After the welcome message, classify what the learner wants next.
INTERNAL OVERRIDE: For this router task only, DO NOT output any ---SUGGESTIONS--- block.

STRICT RULES:
- Return route "adhoc_qna" if they ask a direct question, want a formula, definition, or explanation of one idea, or sound like they want help on something specific right now.
- Return route "planning" ONLY if they clearly want a full guided journey from scratch (e.g. teach me everything, learn from scratch, walk me through the whole topic, structured path, start from zero, beginner overview of everything).

If unsure, prefer "adhoc_qna" so they are never forced into a long curriculum."""
    )
    llm = _llm().with_structured_output(CurriculumPathIntent)
    try:
        result = llm.invoke(
            [
                SystemMessage(content=system),
                HumanMessage(content=f"Learner message: {user_text}"),
            ]
        )
        if result.route == "planning":
            return "planning"
    except Exception:
        pass
    return "adhoc_qna"


def _route_micro_vs_assess(state: TutorState) -> str:
    """
    Structured router for turns after micro_teach or qna (same state: micro_teach + awaiting_micro_reply).
    Returns graph edge name: 'qna' or 'assess'.
    """
    topic = state.get("current_topic", "the lesson")
    user_text = _last_user_text(state.get("messages")).strip()
    ctx = retrieve_context(f"{topic}\n{user_text}")
    last_assistant = _last_assistant_text(state.get("messages"))
    preview = last_assistant[:500] if last_assistant else ""

    quiz_shown = bool(state.get("quiz_asked")) and str(
        state.get("active_quiz") or ""
    ).strip() != ""

    system_rules = (
        _base_prompt(topic, ctx)
        + f"""

ROUTER (INTERNAL — STRUCTURED OUTPUT ONLY)
Classify the learner's LATEST message for routing after a micro-lesson or after your Q&A reply.
INTERNAL OVERRIDE: For this router task only, DO NOT output any ---SUGGESTIONS--- block.

Context snapshot:
- CURRENT TOPIC: {topic}
- Quiz already on screen (learner should answer A/B/C/D): {quiz_shown}
- Last assistant message (excerpt): {preview!r}
- Learner latest message: {user_text!r}

STRICT RULES:
- Return route "qna" for almost everything: questions, confusion, thanks, "got it", follow-ups, requests for formulas, or any normal conversation about the topic.
- Return route "assess" ONLY if they explicitly ask you to quiz or test them (quiz, test me, practice question, MCQ, give me a question), OR when quiz_shown is True and they submit a single letter A/B/C/D as their quiz answer.

Do NOT route to assess just because they say they understand or are ready to move on — that stays in qna unless they explicitly ask for a quiz. If unsure, choose "qna"."""
    )

    llm = _llm().with_structured_output(MicroRouteIntent)
    route = "qna"
    try:
        result = llm.invoke(
            [
                SystemMessage(content=system_rules),
                HumanMessage(
                    content="Classify the learner's latest message for routing."
                ),
            ]
        )
        route = result.route
    except Exception:
        route = "qna"

    if quiz_shown:
        if re.match(r"^\s*[ABCDabcd]\s*$", user_text):
            route = "assess"
        else:
            route = "qna"
    else:
        if re.match(r"^\s*[ABCDabcd]\s*$", user_text):
            route = "qna"
        elif route == "assess" and not _explicit_quiz_request(user_text):
            route = "qna"

    if route == "assess":
        return "assess"
    return "qna"


def _route_plan_vs_micro(state: TutorState) -> str:
    """
    After planning_node, wait for the learner: agree -> micro_teach, else -> qna.
    """
    plan = state.get("teaching_plan") or []
    if not isinstance(plan, list) or len(plan) == 0:
        return "qna"
    step_index = int(state.get("current_step_index", 0))
    if step_index != 0:
        return "micro_teach"
    topic = state.get("current_topic", "the lesson")
    user_text = _last_user_text(state.get("messages")).strip()
    if _signals_plan_start_agreement(user_text):
        return "micro_teach"
    ctx = retrieve_context(f"{topic}\n{user_text}\nplan_review")
    plan_lines = "\n".join(f"{i + 1}. {plan[i]}" for i in range(len(plan)))
    system_rules = (
        _base_prompt(topic, ctx)
        + f"""

ROUTER (INTERNAL — STRUCTURED OUTPUT)
The learner is replying after you showed a step-by-step plan and asked if they are ready for step 1. They have NOT started micro-teaching yet.
INTERNAL OVERRIDE: For this router task only, DO NOT output any ---SUGGESTIONS--- block.

Proposed plan ({len(plan)} steps):
{plan_lines}

Learner latest message: {user_text!r}

STRICT RULES:
- Return route "micro_teach" if they agree to start, want you to walk them through the course/plan, say yes/let us go/take me through/go through the course, or express readiness for step 1.
- Return route "qna" ONLY if they ask a specific question about the plan, want changes, say not yet, or express doubt without agreeing to start.

When in doubt between the two, prefer "micro_teach" if they sound eager to begin the structured path."""
    )
    llm = _llm().with_structured_output(PlanStartIntent)
    route = "qna"
    try:
        result = llm.invoke(
            [
                SystemMessage(content=system_rules),
                HumanMessage(
                    content="Classify whether to start micro-teaching step 1 or answer questions."
                ),
            ]
        )
        route = result.route
    except Exception:
        route = "qna"
    if route == "micro_teach":
        return "micro_teach"
    if _signals_plan_start_agreement(user_text):
        return "micro_teach"
    return "qna"


def greeting_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    context = retrieve_context(topic)
    greeting_shown = bool(state.get("greeting_shown"))

    if not greeting_shown:
        new_messages = _append_message(
            state.get("messages"), "assistant", GENERIC_SESSION_GREETING
        )
        return {
            "messages": new_messages,
            "greeting_shown": True,
            "current_node": "greeting",
            "greeting_next": "",
            "is_off_topic": False,
        }

    last_user = _last_user_text(state.get("messages"))
    intent = _classify_greeting_intent(topic, last_user, context)
    if intent == "off_topic":
        decline = (
            f"I hear you. That is outside the scope of our current lesson on {topic}. "
            "When you are ready, tell me what you would like to learn about this topic and we will continue."
            + _suggestion_footer(
                f"What should I focus on first for {topic}?",
                "Can you give me a quick overview of what this lesson covers?",
                "I'd like to get back on topic—where should we start?",
            )
        )
        new_messages = _append_message(state.get("messages"), "assistant", decline)
        return {
            "messages": new_messages,
            "is_off_topic": True,
            "current_node": "greeting",
            "greeting_next": "",
        }

    path = _classify_curriculum_vs_adhoc(topic, last_user, context)
    if path == "planning":
        return {
            "greeting_next": "planning",
            "is_off_topic": False,
            "current_node": "greeting",
            "is_adhoc_session": False,
        }
    return {
        "greeting_next": "adhoc_qna",
        "is_off_topic": False,
        "current_node": "greeting",
        "teaching_plan": ["Your question"],
        "current_step_index": 0,
        "awaiting_plan_confirmation": False,
        "is_adhoc_session": True,
    }


def planning_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    user_bit = _last_user_text(state.get("messages"))
    context = retrieve_context(f"{topic}\n{user_bit}")
    base_prompt = _base_prompt(topic, context)
    llm = _llm()
    extra = """

NODE: TEACHING PLAN
INTERNAL OVERRIDE: For this planning task only, DO NOT output any ---SUGGESTIONS--- block.
- Propose exactly THREE short step titles for a micro-learning path tailored to the learner's goal.
- Each title must be teachable using ONLY REFERENCE KNOWLEDGE.
- Output ONLY a JSON array of 3 strings, no other text. Example: ["Concept intro", "Key formula", "Worked angle"]"""
    system = base_prompt + extra
    reply = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(
                content="Build the 3-step teaching plan for this learner now."
            ),
        ]
    )
    raw = reply.content if isinstance(reply.content, str) else str(reply.content)
    plan = _normalize_teaching_plan_three(_parse_json_array(raw))
    summary_lines = "\n".join(f"{i + 1}. {plan[i]}" for i in range(len(plan)))
    closing = "How does this plan look to you? Are you ready to start step 1?"
    assistant_text = (
        "Here is a simple 3-step path we can follow together:\n"
        f"{summary_lines}\n\n"
        f"{closing}"
        + _suggestion_footer(
            "What will we cover in step 1?",
            "I'm not sure I'm ready—can we adjust the plan?",
            "Yes—let's start step 1.",
        )
    )
    new_messages = _append_message(state.get("messages"), "assistant", assistant_text)
    return {
        "messages": new_messages,
        "teaching_plan": plan,
        "current_step_index": 0,
        "current_node": "planning",
        "awaiting_plan_confirmation": True,
        "awaiting_micro_reply": False,
        "greeting_next": "",
        "quiz_asked": False,
        "active_quiz": "",
        "assessment_result": "",
        "is_adhoc_session": False,
    }


def micro_teach_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    plan = state.get("teaching_plan") or []
    step_index = int(state.get("current_step_index", 0))
    if step_index < 0:
        step_index = 0
    if step_index >= len(plan) and len(plan) > 0:
        step_index = len(plan) - 1
    step_label = plan[step_index] if plan else f"Step {step_index + 1}"
    n_steps = len(plan) if plan else 1
    user_bit = _last_user_text(state.get("messages"))
    context = retrieve_context(f"{topic}\n{step_label}\n{user_bit}")
    base_prompt = _learner_base_prompt(topic, context)
    llm = _llm()
    extra = f"""

NODE: MICRO-TEACH (ONE STEP ONLY)
SYSTEM RULE: You are currently on step {step_index + 1} of {n_steps}. You MUST ONLY teach the specific concept for this SINGLE step: "{step_label}". ABSOLUTELY DO NOT teach or summarize the other steps, the full course, or future steps. One step per message.

- Hard cap: at most 150 words for your entire message including the closing line.
- Use ONLY REFERENCE KNOWLEDGE and include one simple real-world analogy for this step only.
- End with ONE short casual check-in — naturally choose either "Does that make sense?" or "Any questions?" Do NOT mention quizzes, tests, or being quizzed.
"""
    system = base_prompt + extra
    reply = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(
                content=(
                    f"Produce ONLY the lesson for step {step_index + 1} of {n_steps} "
                    f"named \"{step_label}\". Do not teach or preview any other step."
                )
            ),
        ]
    )
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    new_messages = _append_message(state.get("messages"), "assistant", text)
    return {
        "messages": new_messages,
        "current_node": "micro_teach",
        "awaiting_micro_reply": True,
        "awaiting_plan_confirmation": False,
        "quiz_asked": False,
        "assessment_result": "",
        "route_after_assess_correct": "",
    }


def qna_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    plan = state.get("teaching_plan") or []
    step_index = int(state.get("current_step_index", 0))
    step_label = plan[step_index] if plan and step_index < len(plan) else "this step"
    context = retrieve_context(f"{topic}\n{step_label}\n{_last_user_text(state.get('messages'))}")
    base_prompt = _learner_base_prompt(topic, context)
    llm = _llm()
    plan_review = bool(state.get("awaiting_plan_confirmation"))
    if plan_review:
        extra = f"""

NODE: Q&A (PLAN REVIEW — NOT STARTED STEP 1 YET)
- They are reacting to the proposed 3-step plan before any micro-lesson.
- Answer ONLY what they asked, using ONLY REFERENCE KNOWLEDGE.
- Hard cap: at most 100 words for your entire reply including the closing line.
- End with one casual line inviting them to share thoughts or say when they want to start step 1. Do NOT mention quizzes or tests."""
    else:
        extra = f"""

NODE: Q&A (CURRENT STEP: {step_label})
- If they asked a question, answer using ONLY REFERENCE KNOWLEDGE.
- If they only acknowledged (thanks, got it, makes sense), reply briefly and warmly; invite them to go deeper or ask what is next.
- Hard cap: at most 100 words for your entire reply including the closing line.
- End with ONE casual check-in — choose naturally between "Does that make sense?" or "Any questions?" Do NOT mention quizzes, tests, or being quizzed."""
    system = base_prompt + extra
    prior = _messages_from_state(state.get("messages"))
    reply = llm.invoke(
        [
            SystemMessage(content=system),
            HumanMessage(content="Answer the learner's follow-up now."),
        ]
        + prior[-6:]
    )
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    new_messages = _append_message(state.get("messages"), "assistant", text)
    out: TutorState = {
        "messages": new_messages,
        "route_after_assess_correct": "",
    }
    if plan_review:
        out["current_node"] = "planning"
        out["awaiting_plan_confirmation"] = True
        out["awaiting_micro_reply"] = False
    else:
        out["current_node"] = "micro_teach"
        out["awaiting_plan_confirmation"] = False
        out["awaiting_micro_reply"] = True
    return out


def assess_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    plan = state.get("teaching_plan") or []
    step_index = int(state.get("current_step_index", 0))
    step_label = plan[step_index] if plan and step_index < len(plan) else "current step"
    context = retrieve_context(
        f"{topic}\n{step_label}\n{_last_user_text(state.get('messages'))}"
    )
    base_prompt = _base_prompt(topic, context)
    llm = _llm()
    quiz_asked = bool(state.get("quiz_asked"))

    if not quiz_asked:
        learner_base = _learner_base_prompt(topic, context)
        quiz_extra = f"""

NODE: DIAGNOSTIC MCQ (CURRENT MICRO-STEP ONLY)
- The learner is on step: "{step_label}".
- Write ONE compact multiple-choice question with exactly 4 labeled options: A, B, C, D.
- Ground EVERYTHING ONLY in REFERENCE KNOWLEDGE and this step.
- Do NOT reveal the correct answer.
- CRITICAL: If your quiz requires ANY calculation, you MUST explicitly state all necessary numbers, rates, and premises IN the question text itself. Do not assume the user remembers data from earlier messages or the textbook.
- Hard cap: entire MCQ block at most 100 words including options.
- End by asking them to reply with the letter only (A, B, C, or D).
"""
        system = learner_base + quiz_extra
        reply = llm.invoke([SystemMessage(content=system)])
        quiz_text = reply.content if isinstance(reply.content, str) else str(reply.content)
        new_messages = _append_message(state.get("messages"), "assistant", quiz_text)
        return {
            "messages": new_messages,
            "quiz_asked": True,
            "active_quiz": quiz_text,
            "assessment_result": "",
            "current_node": "assess",
        }

    quiz_text = str(state.get("active_quiz") or "")
    answer = _last_user_text(state.get("messages"))
    grade_extra = f"""

NODE: GRADING (INTERNAL)
You are grading the learner's reply to this quiz:
{quiz_text}

Learner reply: {answer}
INTERNAL OVERRIDE: For this grading task only, DO NOT output any ---SUGGESTIONS--- block.

Output exactly one line in this format:
VERDICT: CORRECT
or
VERDICT: WRONG

Use CORRECT only if the chosen letter matches the best answer grounded strictly in REFERENCE KNOWLEDGE.
"""
    system = base_prompt + grade_extra
    reply = llm.invoke([SystemMessage(content=system)])
    raw = reply.content if isinstance(reply.content, str) else str(reply.content)
    upper = raw.upper()
    verdict = "wrong"
    if "VERDICT: CORRECT" in upper:
        verdict = "correct"

    if verdict != "correct":
        return {
            "assessment_result": "wrong",
            "current_node": "remediate",
            "quiz_asked": True,
            "active_quiz": quiz_text,
            "route_after_assess_correct": "",
        }

    plan_len = len(plan) if plan else 0
    new_idx = step_index + 1
    congrats = (
        "Nice work — that fits what we covered in this step."
        + _suggestion_footer(
            "Can we go deeper on what we just covered?",
            "What should I focus on next?",
            "I'm ready to move on—what's next?",
        )
    )

    if plan_len == 0 or new_idx >= plan_len:
        if _is_adhoc_session(state):
            adhoc_done = (
                "Great job! Do you have any other questions about this or something new?"
                + _suggestion_footer(
                    "Can you explain that last idea one more time?",
                    "How does this connect to the rest of the lesson?",
                    "What is a common mistake to avoid here?",
                )
            )
            new_messages = _append_message(state.get("messages"), "assistant", adhoc_done)
            return {
                "messages": new_messages,
                "assessment_result": "correct",
                "quiz_asked": False,
                "active_quiz": "",
                "route_after_assess_correct": "",
                "remediation_attempts": 0,
                "current_node": "micro_teach",
                "awaiting_micro_reply": True,
                "is_adhoc_session": True,
            }
        new_messages = _append_message(state.get("messages"), "assistant", congrats)
        return {
            "messages": new_messages,
            "assessment_result": "correct",
            "quiz_asked": False,
            "active_quiz": "",
            "route_after_assess_correct": "feynman",
            "remediation_attempts": 0,
            "current_node": "assess",
        }

    new_messages = _append_message(state.get("messages"), "assistant", congrats)
    return {
        "messages": new_messages,
        "current_step_index": new_idx,
        "assessment_result": "correct",
        "quiz_asked": False,
        "active_quiz": "",
        "awaiting_micro_reply": False,
        "route_after_assess_correct": "micro_teach",
        "remediation_attempts": 0,
        "current_node": "assess",
    }


def remediate_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    context = retrieve_context(topic + "\n" + _last_user_text(state.get("messages")))
    base_prompt = _learner_base_prompt(topic, context)
    attempts = int(state.get("remediation_attempts", 0)) + 1
    llm = _llm()
    if attempts >= 3:
        breaker = """

CIRCUIT BREAKER (OVERRIDE)
The learner reached the remediation attempt limit. Briefly give a clear step-by-step resolution grounded ONLY in REFERENCE KNOWLEDGE, then encourage them.
You MAY state the final result here because the Socratic no-spoiler rule is lifted for this message only.
Hard cap: at most 100 words total.
"""
        system = base_prompt + breaker
        reply = llm.invoke([SystemMessage(content=system)])
        text = reply.content if isinstance(reply.content, str) else str(reply.content)
        new_messages = _append_message(state.get("messages"), "assistant", text)
        return {
            "messages": new_messages,
            "remediation_attempts": 0,
            "circuit_breaker_triggered": True,
            "quiz_asked": False,
            "active_quiz": "",
            "assessment_result": "",
            "route_after_assess_correct": "",
            "current_node": "micro_teach",
            "awaiting_micro_reply": True,
        }
    socratic = """

NODE: SOCRATIC REMEDIATION
- Respond to the learner's misconception using ONLY REFERENCE KNOWLEDGE.
- Do NOT give the final numerical answer or the final letter answer.
- Offer a hint and ask a guiding question so the learner can retry.
- Hard cap: at most 100 words total. Keep it warm and brief.
"""
    system = base_prompt + socratic
    prior = _messages_from_state(state.get("messages"))
    reply = llm.invoke(
        [SystemMessage(content=system)]
        + prior
        + [HumanMessage(content="The learner answered the quiz incorrectly. Apply Socratic remediation.")]
    )
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    new_messages = _append_message(state.get("messages"), "assistant", text)
    preserved_quiz = str(state.get("active_quiz") or "")
    return {
        "messages": new_messages,
        "remediation_attempts": attempts,
        "quiz_asked": True,
        "active_quiz": preserved_quiz,
        "assessment_result": "",
        "circuit_breaker_triggered": False,
        "route_after_assess_correct": "",
        "current_node": "assess",
    }


def feynman_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    context = retrieve_context(topic + "\n" + _last_user_text(state.get("messages")))
    base_prompt = _learner_base_prompt(topic, context)
    feyn = """

NODE: FEYNMAN CHECKPOINT (FINAL)
- Congratulate the learner for finishing all micro-steps.
- Ask them to summarize the overall idea in plain English in at least two sentences, in their own words.
- If their explanation is sufficiently accurate based ONLY on REFERENCE KNOWLEDGE, reply with a line starting with: PASS: YES
- Otherwise reply with: PASS: NO and give brief, kind feedback grounded in REFERENCE KNOWLEDGE.
- Hard cap: at most 100 words total for your entire message.
"""
    system = base_prompt + feyn
    prior = _messages_from_state(state.get("messages"))
    llm = _llm()
    reply = llm.invoke(
        [SystemMessage(content=system)]
        + prior
        + [HumanMessage(content="Evaluate the learner's latest explanation.")]
    )
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    new_messages = _append_message(state.get("messages"), "assistant", text)
    upper = text.upper()
    mastered = "PASS: YES" in upper
    next_node = "complete" if mastered else "feynman"
    return {
        "messages": new_messages,
        "concept_mastered": mastered,
        "current_node": next_node,
        "route_after_assess_correct": "",
    }


def _route_after_greeting(state: TutorState) -> str:
    next_g = state.get("greeting_next", "")
    if next_g == "planning":
        return "planning"
    if next_g == "adhoc_qna":
        return "qna"
    return END


def _route_entry(state: TutorState) -> str:
    node = state.get("current_node", "greeting")
    if node == "complete":
        return END
    if node == "feynman":
        return "feynman"
    if node == "planning":
        if bool(state.get("awaiting_plan_confirmation")):
            plan = state.get("teaching_plan") or []
            step_index = int(state.get("current_step_index", 0))
            if isinstance(plan, list) and len(plan) > 0 and step_index == 0:
                return _route_plan_vs_micro(state)
        return "micro_teach"
    if node == "micro_teach":
        if not bool(state.get("awaiting_micro_reply")):
            return "micro_teach"
        return _route_micro_vs_assess(state)
    if node == "assess" or (
        bool(state.get("quiz_asked")) and str(state.get("active_quiz") or "").strip() != ""
    ):
        return "assess"
    if node == "greeting":
        return "greeting"
    return "greeting"


def _route_after_assess(state: TutorState) -> str:
    if state.get("assessment_result") == "wrong":
        return "remediate"
    route = state.get("route_after_assess_correct", "")
    if route == "feynman":
        return "feynman"
    if route == "micro_teach":
        return "micro_teach"
    return END


def _route_after_remediate(state: TutorState) -> str:
    if bool(state.get("circuit_breaker_triggered")):
        return END
    return END


def build_graph():
    graph = StateGraph(TutorState)
    graph.add_node("greeting", greeting_node)
    graph.add_node("planning", planning_node)
    graph.add_node("micro_teach", micro_teach_node)
    graph.add_node("qna", qna_node)
    graph.add_node("assess", assess_node)
    graph.add_node("remediate", remediate_node)
    graph.add_node("feynman", feynman_node)

    graph.add_conditional_edges(
        START,
        _route_entry,
        {
            "greeting": "greeting",
            "micro_teach": "micro_teach",
            "qna": "qna",
            "assess": "assess",
            "feynman": "feynman",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "greeting",
        _route_after_greeting,
        {
            "planning": "planning",
            "qna": "qna",
            END: END,
        },
    )
    graph.add_edge("planning", END)
    graph.add_conditional_edges(
        "assess",
        _route_after_assess,
        {
            "remediate": "remediate",
            "feynman": "feynman",
            "micro_teach": "micro_teach",
            END: END,
        },
    )
    graph.add_conditional_edges(
        "remediate",
        _route_after_remediate,
        {
            END: END,
        },
    )
    # After micro_teach / qna content is emitted, the run ends; the next user turn is
    # routed from START via _route_micro_vs_assess (structured) to qna vs assess.
    graph.add_edge("micro_teach", END)
    graph.add_edge("qna", END)
    graph.add_edge("feynman", END)
    return graph.compile()


_COMPILED = build_graph()


def _normalize_incoming(state: dict | None) -> TutorState:
    if state is None:
        state = {}
    messages = state.get("messages")
    if messages is None:
        messages = []
    current_node = state.get("current_node", "greeting")
    if current_node is None:
        current_node = "greeting"
    if str(current_node) == "teach":
        current_node = "greeting"
    topic = state.get("current_topic", "Discounted Dividend Valuation")
    if topic is None or str(topic).strip() == "":
        topic = "Discounted Dividend Valuation"
    remediation = state.get("remediation_attempts", 0)
    if remediation is None:
        remediation = 0
    mastered = bool(state.get("concept_mastered"))
    quiz_asked = bool(state.get("quiz_asked"))
    assessment = state.get("assessment_result", "")
    if assessment is None:
        assessment = ""
    breaker = bool(state.get("circuit_breaker_triggered"))
    active = state.get("active_quiz", "")
    if active is None:
        active = ""
    teaching_plan = state.get("teaching_plan")
    if teaching_plan is None:
        teaching_plan = []
    step_index = state.get("current_step_index", 0)
    if step_index is None:
        step_index = 0
    is_off_topic = bool(state.get("is_off_topic"))
    greeting_shown = bool(state.get("greeting_shown"))
    awaiting_micro = bool(state.get("awaiting_micro_reply"))
    greeting_next = state.get("greeting_next", "")
    if greeting_next is None:
        greeting_next = ""
    route_after = state.get("route_after_assess_correct", "")
    if route_after is None:
        route_after = ""
    awaiting_plan = bool(state.get("awaiting_plan_confirmation"))
    adhoc = bool(state.get("is_adhoc_session"))
    video_url = state.get("current_video_url", "")
    if video_url is None or str(video_url).strip() == "":
        video_url = DEFAULT_CURRENT_VIDEO_URL
    else:
        video_url = str(video_url).strip()
    return {
        "messages": messages,
        "current_node": str(current_node),
        "remediation_attempts": int(remediation),
        "concept_mastered": mastered,
        "current_topic": str(topic),
        "quiz_asked": quiz_asked,
        "assessment_result": str(assessment),
        "circuit_breaker_triggered": breaker,
        "active_quiz": str(active),
        "teaching_plan": list(teaching_plan) if isinstance(teaching_plan, list) else [],
        "current_step_index": int(step_index),
        "is_off_topic": is_off_topic,
        "greeting_shown": greeting_shown,
        "awaiting_micro_reply": awaiting_micro,
        "greeting_next": str(greeting_next),
        "route_after_assess_correct": str(route_after),
        "awaiting_plan_confirmation": awaiting_plan,
        "is_adhoc_session": adhoc,
        "current_video_url": video_url,
    }


def _extract_reply(result: TutorState) -> str:
    messages = result.get("messages")
    if not messages:
        return ""
    idx = len(messages) - 1
    while idx >= 0:
        m = messages[idx]
        if m.get("role") == "assistant":
            return str(m.get("content", ""))
        idx -= 1
    return ""


def _concat_assistant_replies(result: TutorState) -> str:
    """Join recent assistant messages when multiple nodes run in one invoke."""
    messages = result.get("messages")
    if not messages:
        return ""
    parts: list[str] = []
    index = 0
    while index < len(messages):
        m = messages[index]
        if m.get("role") == "assistant":
            parts.append(str(m.get("content", "")))
        index += 1
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0]
    return "\n\n---\n\n".join(parts[-6:])


def run_tutor_flow(user_input: str, state: dict | None) -> dict:
    load_dotenv()
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key == "":
        return {
            "state": state or {},
            "reply": "Server misconfiguration: OPENAI_API_KEY is not set.",
            "error": "missing_api_key",
        }
    normalized = _normalize_incoming(state)
    if normalized.get("current_node") == "complete" or bool(
        normalized.get("concept_mastered")
    ):
        normalized = _reset_session_for_new_learning(normalized)
    text = user_input.strip()
    if text == "":
        if normalized.get("current_node", "greeting") == "greeting" and not normalized.get(
            "greeting_shown"
        ):
            text = "Please begin the lesson."
        else:
            text = "Continue."
    messages = _append_message(normalized.get("messages"), "user", text)
    normalized["messages"] = messages
    result = _COMPILED.invoke(dict(normalized))
    reply = _concat_assistant_replies(result)
    return {"state": dict(result), "reply": reply}
