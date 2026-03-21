import os
from pathlib import Path
from typing import TypedDict

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph

load_dotenv()

BACKEND_DIR = Path(__file__).resolve().parent
CHROMA_DIR = BACKEND_DIR / "chroma_db"
COLLECTION_NAME = "knowledge_base"

IRONCLAD_TEMPLATE = '''You are a strict, content-agnostic Finance AI Tutor.
CURRENT TOPIC: {payload_topic}
REFERENCE KNOWLEDGE: {retrieved_context}

CRITICAL RULES:
1. ZERO HALLUCINATION: You must base your teachings, explanations, and quizzes STRICTLY and ONLY on the {retrieved_context}. Do not use your pre-trained internet knowledge to invent formulas or facts.
2. OUT-OF-DOMAIN REJECTION: If the user asks about ANYTHING unrelated to the CURRENT TOPIC (e.g., weather, coding, unrelated stocks like TSLA, or general advice), you MUST immediately intercept and reply: "That is outside the scope of our current lesson on [CURRENT TOPIC]. Let's refocus on the material."
3. NO SPOILERS: During the Socratic Remediation phase, NEVER give the final mathematical answer directly. Provide a hint based on the {retrieved_context} and ask the user to try again.'''


class TutorState(TypedDict, total=False):
    messages: list
    current_node: str
    remediation_attempts: int
    concept_mastered: bool
    current_topic: str
    quiz_asked: bool
    assessment_result: str
    circuit_breaker_triggered: bool
    active_quiz: str


def _get_collection():
    embedding_fn = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME not in names:
        return None
    return client.get_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )


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


def _llm() -> ChatOpenAI:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key == "":
        raise ValueError("OPENAI_API_KEY is not set")
    return ChatOpenAI(model="gpt-4o", temperature=0.2)


def teach_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    context = retrieve_context(topic)
    base_prompt = IRONCLAD_TEMPLATE.format(
        payload_topic=topic,
        retrieved_context=context,
    )
    teach_extra = """

NODE: TEACH / INTUITION HOOK
- Open with a short real-world scenario that anchors the idea before any formulas.
- Explain the core idea using ONLY the REFERENCE KNOWLEDGE.
- DO NOT ask the user if they want to continue or if they are ready for a quiz. Conclude your explanation by seamlessly handing over to the assessment, e.g., "Let's test your understanding with a quick question:". Do not ask quiz questions in this message; the next step will present the question.
"""
    system = base_prompt + teach_extra
    prior = _messages_from_state(state.get("messages"))
    llm = _llm()
    reply = llm.invoke(
        [SystemMessage(content=system)] + prior + [HumanMessage(content="Teach this topic now following the node rules.")]
    )
    text = reply.content if isinstance(reply.content, str) else str(reply.content)
    new_messages = _append_message(state.get("messages"), "assistant", text)
    return {
        "messages": new_messages,
        "quiz_asked": False,
        "assessment_result": "",
    }


def assess_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    context = retrieve_context(topic + "\n" + _last_user_text(state.get("messages")))
    base_prompt = IRONCLAD_TEMPLATE.format(
        payload_topic=topic,
        retrieved_context=context,
    )
    llm = _llm()
    quiz_asked = bool(state.get("quiz_asked"))
    if not quiz_asked:
        quiz_extra = """

NODE: DIAGNOSTIC ASSESSMENT
- Write ONE multiple-choice question with exactly 4 labeled options: A, B, C, D.
- Design distractors to catch typical misconceptions (misunderstanding definitions, timing, or notation), still grounded ONLY in REFERENCE KNOWLEDGE.
- Do NOT reveal the correct answer.
- End with: "Reply with the letter only (A, B, C, or D)."
"""
        system = base_prompt + quiz_extra
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
    quiz_text = state.get("active_quiz", "")
    answer = _last_user_text(state.get("messages"))
    grade_extra = f"""

NODE: GRADING (INTERNAL)
You are grading the learner's reply to this quiz:
{quiz_text}

Learner reply: {answer}

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
    if verdict == "correct":
        invite = (
            "Great work. Before we move on, explain the core idea in your own words "
            "in plain English (at least two sentences)."
        )
        return {
            "messages": _append_message(state.get("messages"), "assistant", invite),
            "assessment_result": "correct",
            "current_node": "feynman",
            "quiz_asked": False,
            "active_quiz": "",
        }
    return {
        "assessment_result": "wrong",
        "current_node": "remediate",
        "quiz_asked": True,
        "active_quiz": quiz_text,
    }


def remediate_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    context = retrieve_context(topic + "\n" + _last_user_text(state.get("messages")))
    base_prompt = IRONCLAD_TEMPLATE.format(
        payload_topic=topic,
        retrieved_context=context,
    )
    attempts = int(state.get("remediation_attempts", 0)) + 1
    llm = _llm()
    if attempts >= 3:
        breaker = """

CIRCUIT BREAKER (OVERRIDE)
The learner reached the remediation attempt limit. Briefly give a clear step-by-step resolution grounded ONLY in REFERENCE KNOWLEDGE, then encourage them.
You MAY state the final result here because the Socratic no-spoiler rule is lifted for this message only.
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
            "current_node": "feynman",
        }
    socratic = """

NODE: SOCRATIC REMEDIATION
- Respond to the learner's misconception using ONLY REFERENCE KNOWLEDGE.
- Do NOT give the final numerical answer or the final letter answer.
- Offer a hint and ask a guiding question so the learner can retry.
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
        "current_node": "assess",
    }


def feynman_node(state: TutorState) -> TutorState:
    topic = state.get("current_topic", "the lesson")
    context = retrieve_context(topic + "\n" + _last_user_text(state.get("messages")))
    base_prompt = IRONCLAD_TEMPLATE.format(
        payload_topic=topic,
        retrieved_context=context,
    )
    feyn = """

NODE: FEYNMAN CHECKPOINT
- Ask the learner to explain the core idea in plain English without copying phrases blindly.
- If their explanation is sufficiently accurate based ONLY on REFERENCE KNOWLEDGE, reply with a line starting with: PASS: YES
- Otherwise reply with: PASS: NO and give brief feedback grounded in REFERENCE KNOWLEDGE.
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
    }


def _route_entry(state: TutorState) -> str:
    node = state.get("current_node", "teach")
    if node == "complete":
        return END
    if node == "teach":
        return "teach"
    if node == "assess":
        return "assess"
    if node == "remediate":
        return "remediate"
    if node == "feynman":
        return "feynman"
    return "teach"


def _route_after_assess(state: TutorState) -> str:
    if state.get("assessment_result") == "wrong":
        return "remediate"
    return END


def _route_after_remediate(state: TutorState) -> str:
    return END


def build_graph():
    graph = StateGraph(TutorState)
    graph.add_node("teach", teach_node)
    graph.add_node("assess", assess_node)
    graph.add_node("remediate", remediate_node)
    graph.add_node("feynman", feynman_node)
    graph.add_conditional_edges(
        START,
        _route_entry,
        {
            "teach": "teach",
            "assess": "assess",
            "remediate": "remediate",
            "feynman": "feynman",
            END: END,
        },
    )
    graph.add_edge("teach", "assess")
    graph.add_conditional_edges(
        "assess",
        _route_after_assess,
        {
            "remediate": "remediate",
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
    graph.add_edge("feynman", END)
    return graph.compile()


_COMPILED = build_graph()


def _normalize_incoming(state: dict | None) -> TutorState:
    if state is None:
        state = {}
    messages = state.get("messages")
    if messages is None:
        messages = []
    current_node = state.get("current_node", "teach")
    if current_node is None:
        current_node = "teach"
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
    if normalized.get("current_node") == "complete":
        return {
            "state": dict(normalized),
            "reply": "You have completed this module. Send a new session to continue.",
        }
    text = user_input.strip()
    if text == "":
        if normalized.get("current_node", "teach") == "teach":
            text = "Please begin the lesson."
        else:
            text = "Continue."
    messages = _append_message(normalized.get("messages"), "user", text)
    normalized["messages"] = messages
    result = _COMPILED.invoke(dict(normalized))
    reply = _extract_reply(result)
    return {"state": dict(result), "reply": reply}
