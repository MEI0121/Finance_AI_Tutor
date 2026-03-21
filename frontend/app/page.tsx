"use client";

import { BookOpen, MessageCircle, Send } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";

const CHAT_URL = "http://127.0.0.1:8000/chat";

type ChatMessage = {
  role: string;
  content: string;
};

type TutorState = {
  messages?: ChatMessage[];
  current_node?: string;
  current_topic?: string;
  quiz_asked?: boolean;
  active_quiz?: string;
  remediation_attempts?: number;
  concept_mastered?: boolean;
  [key: string]: unknown;
};

const INITIAL_TUTOR_STATE: TutorState = {
  current_topic: "Discounted Dividend Valuation",
  current_node: "teach",
  remediation_attempts: 0,
  concept_mastered: false,
  messages: [],
};

type ChatResponse = {
  state?: TutorState;
  reply?: string;
  error?: string;
};

function lastAssistantIndex(messages: ChatMessage[]): number {
  let idx = messages.length - 1;
  while (idx >= 0) {
    if (messages[idx].role === "assistant") {
      return idx;
    }
    idx -= 1;
  }
  return -1;
}

export default function Home() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [tutorState, setTutorState] = useState<TutorState | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [input, setInput] = useState("");
  const [chatError, setChatError] = useState<string | null>(null);
  const bootstrapped = useRef(false);

  const applyChatResponse = useCallback((data: ChatResponse) => {
    if (data.error === "missing_api_key") {
      setChatError("Server misconfiguration: OPENAI_API_KEY is not set.");
      return;
    }
    const next = data.state;
    if (next) {
      setTutorState(next);
      const nextMessages = next.messages;
      if (Array.isArray(nextMessages)) {
        setMessages(nextMessages as ChatMessage[]);
      }
    }
  }, []);

  const sendToTutor = useCallback(
    (message: string, state: TutorState) => {
      setIsLoading(true);
      setChatError(null);
      fetch(CHAT_URL, {
        method: "POST",
        credentials: "include",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ message, state }),
      })
        .then((response) => {
          if (!response.ok) {
            setIsLoading(false);
            setChatError(`Request failed (${response.status}).`);
            return null;
          }
          return response.json() as Promise<ChatResponse>;
        })
        .then((data) => {
          if (data === null) {
            return;
          }
          applyChatResponse(data);
          setIsLoading(false);
        })
        .catch(() => {
          setIsLoading(false);
          setChatError("Unable to reach the tutor. Is the backend running?");
        });
    },
    [applyChatResponse]
  );

  useEffect(() => {
    if (bootstrapped.current) {
      return;
    }
    bootstrapped.current = true;
    sendToTutor("", INITIAL_TUTOR_STATE);
  }, [sendToTutor]);

  const activeQuiz = tutorState?.active_quiz;
  const hasActiveQuiz =
    typeof activeQuiz === "string" && activeQuiz.trim() !== "";
  const lastIdx = lastAssistantIndex(messages);
  const lastRole = messages.length > 0 ? messages[messages.length - 1].role : "";
  const showQuizButtons =
    hasActiveQuiz &&
    !isLoading &&
    lastRole === "assistant" &&
    lastIdx >= 0;

  const handleSend = () => {
    if (isLoading || tutorState === null) {
      return;
    }
    const trimmed = input.trim();
    if (trimmed === "") {
      return;
    }
    setInput("");
    sendToTutor(trimmed, tutorState);
  };

  const handleQuizPick = (letter: string) => {
    if (isLoading || tutorState === null || !hasActiveQuiz) {
      return;
    }
    sendToTutor(letter, tutorState);
  };

  return (
    <div className="flex h-screen min-h-0 flex-col bg-zinc-100 text-zinc-900">
      <div className="flex min-h-0 flex-1 flex-col md:flex-row">
        {/* Left: course material */}
        <aside className="flex w-full shrink-0 flex-col border-b border-zinc-200 bg-zinc-950 text-zinc-100 md:w-[min(40%,28rem)] md:border-b-0 md:border-r md:border-zinc-800">
          <div className="flex items-center gap-2 border-b border-zinc-800 px-6 py-4">
            <BookOpen className="h-5 w-5 text-emerald-400" aria-hidden />
            <span className="text-sm font-semibold uppercase tracking-wide text-zinc-400">
              Course Material
            </span>
          </div>
          <div className="flex flex-1 flex-col gap-3 px-6 py-8">
            <h1 className="text-xl font-semibold leading-snug text-white">
              Discounted Dividend Valuation
            </h1>
            <p className="text-sm leading-relaxed text-zinc-400">
              Follow the tutor on the right. Your lesson state and quiz are
              synchronized with the server.
            </p>
          </div>
        </aside>

        {/* Right: chat */}
        <section className="flex min-h-0 min-w-0 flex-1 flex-col bg-white">
          <header className="flex items-center gap-2 border-b border-zinc-200 px-4 py-3">
            <MessageCircle className="h-5 w-5 text-emerald-600" aria-hidden />
            <h2 className="text-sm font-semibold text-zinc-800">AI Tutor</h2>
          </header>

          <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
            {chatError ? (
              <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
                {chatError}
              </p>
            ) : null}

            <ul className="flex flex-col gap-4">
              {messages.map((m, i) => {
                const isUser = m.role === "user";
                const isLastAssistant = i === lastIdx;
                return (
                  <li key={`${i}-${m.role}`} className="flex flex-col gap-2">
                    <div
                      className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                    >
                      <div
                        className={`max-w-[min(100%,42rem)] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                          isUser
                            ? "bg-emerald-600 text-white"
                            : "border border-zinc-200 bg-zinc-50 text-zinc-900"
                        }`}
                      >
                        {m.content}
                      </div>
                    </div>
                    {isLastAssistant && showQuizButtons ? (
                      <div className="flex flex-wrap gap-2 pl-0 md:pl-0">
                        {(["A", "B", "C", "D"] as const).map((letter) => (
                          <button
                            key={letter}
                            type="button"
                            disabled={isLoading}
                            onClick={() => handleQuizPick(letter)}
                            className="min-w-[3rem] rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-semibold text-zinc-800 shadow-sm transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50"
                          >
                            {letter}
                          </button>
                        ))}
                      </div>
                    ) : null}
                  </li>
                );
              })}
            </ul>

            {isLoading ? (
              <p className="mt-4 text-sm text-zinc-500">Thinking…</p>
            ) : null}
          </div>

          <div className="border-t border-zinc-200 p-4">
            <div className="flex gap-2">
              <input
                type="text"
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    handleSend();
                  }
                }}
                placeholder={
                  tutorState?.current_node === "complete"
                    ? "Module complete."
                    : "Type a message…"
                }
                disabled={isLoading || tutorState?.current_node === "complete"}
                className="min-w-0 flex-1 rounded-lg border border-zinc-300 px-3 py-2 text-sm outline-none ring-emerald-500 focus:ring-2 disabled:bg-zinc-100"
              />
              <button
                type="button"
                onClick={handleSend}
                disabled={
                  isLoading ||
                  tutorState === null ||
                  tutorState?.current_node === "complete"
                }
                className="inline-flex items-center gap-2 rounded-lg bg-emerald-600 px-4 py-2 text-sm font-medium text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
              >
                <Send className="h-4 w-4" aria-hidden />
                Send
              </button>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
