"use client";

import {
  BookOpen,
  ChevronDown,
  MessageCircle,
  Play,
  Send,
  Sparkles,
  X,
} from "lucide-react";
import dynamic from "next/dynamic";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

import { AiQuiz } from "@/components/AiQuiz";
import { AiSlides } from "@/components/AiSlides";

const TextbookPdf = dynamic(
  () => import("@/components/TextbookPdf").then((m) => m.TextbookPdf),
  {
    ssr: false,
    loading: () => (
      <div className="flex min-h-[320px] items-center justify-center rounded-xl border border-dashed border-zinc-200 bg-zinc-50 text-sm text-zinc-500">
        Loading textbook viewer…
      </div>
    ),
  }
);

const CHAT_URL = "http://127.0.0.1:8000/chat";

type WorkspaceMode = "textbook" | "slides" | "quiz";

type ChatMessage = {
  role: string;
  content: string;
};

type TutorState = {
  messages?: ChatMessage[];
  current_node?: string;
  current_topic?: string;
  current_video_url?: string;
  quiz_asked?: boolean;
  active_quiz?: string;
  remediation_attempts?: number;
  concept_mastered?: boolean;
  [key: string]: unknown;
};

const DEFAULT_VIDEO_EMBED_URL =
  "https://www.youtube.com/embed/-mQJ7a4U9Z8?si=xDlg2zON0SiUOP7-";

const INITIAL_TUTOR_STATE: TutorState = {
  current_topic: "Discounted Dividend Valuation",
  current_node: "greeting",
  current_video_url: DEFAULT_VIDEO_EMBED_URL,
  remediation_attempts: 0,
  concept_mastered: false,
  messages: [],
};

type ChatResponse = {
  state?: TutorState;
  reply?: string;
  error?: string;
};

type SelectionPopoverState = {
  text: string;
  x: number;
  y: number;
  placeAbove: boolean;
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
  const [chatOpen, setChatOpen] = useState(false);
  const [videoExpanded, setVideoExpanded] = useState(false);
  const [workspaceMode, setWorkspaceMode] = useState<WorkspaceMode>("textbook");
  const [selectionPopover, setSelectionPopover] =
    useState<SelectionPopoverState | null>(null);

  const readingRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
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

  const displayTopic =
    (tutorState?.current_topic ?? INITIAL_TUTOR_STATE.current_topic) ||
    "Discounted Dividend Valuation";
  const videoEmbedUrl = (
    tutorState?.current_video_url ??
    INITIAL_TUTOR_STATE.current_video_url ??
    ""
  )
    .toString()
    .trim();
  const showVideoPlayer = videoEmbedUrl.length > 0;

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

  const sendFromSelection = useCallback(
    (message: string) => {
      setSelectionPopover(null);
      setChatOpen(true);
      if (tutorState === null) {
        setChatError("Still connecting to the tutor. Try again in a moment.");
        return;
      }
      sendToTutor(message, tutorState);
    },
    [sendToTutor, tutorState]
  );

  const openChatWithMessage = useCallback(
    (message: string) => {
      setChatOpen(true);
      if (tutorState === null) {
        setChatError("Still connecting to the tutor. Try again in a moment.");
        return;
      }
      sendToTutor(message, tutorState);
    },
    [sendToTutor, tutorState]
  );

  const handleReadingMouseUp = useCallback(() => {
    window.requestAnimationFrame(() => {
      const root = readingRef.current;
      const sel = window.getSelection();
      if (!root || !sel || sel.rangeCount === 0 || sel.isCollapsed) {
        setSelectionPopover(null);
        return;
      }
      const text = sel.toString().replace(/\s+/g, " ").trim();
      if (text.length < 2) {
        setSelectionPopover(null);
        return;
      }
      const range = sel.getRangeAt(0);
      if (!root.contains(range.commonAncestorContainer)) {
        setSelectionPopover(null);
        return;
      }
      const rect = range.getBoundingClientRect();
      const margin = 8;
      const placeAbove = rect.top >= 72;
      const centerX = rect.left + rect.width / 2;
      const x = Math.min(
        window.innerWidth - margin,
        Math.max(margin, centerX)
      );
      const y = placeAbove ? rect.top : rect.bottom;
      setSelectionPopover({
        text,
        x,
        y,
        placeAbove,
      });
    });
  }, []);

  useEffect(() => {
    if (!selectionPopover) {
      return;
    }
    const onDocMouseDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (popoverRef.current?.contains(t)) {
        return;
      }
      setSelectionPopover(null);
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [selectionPopover]);

  useLayoutEffect(() => {
    const onScroll = () => setSelectionPopover(null);
    window.addEventListener("scroll", onScroll, true);
    return () => window.removeEventListener("scroll", onScroll, true);
  }, []);

  const truncatedForMessage = (t: string, max = 1200) =>
    t.length > max ? `${t.slice(0, max)}…` : t;

  return (
    <div className="flex h-screen min-h-0 flex-col bg-zinc-200/80 text-zinc-900 md:flex-row">
      {/* Left: Knowledge sidebar */}
      <aside className="flex w-full shrink-0 flex-col border-b border-zinc-800 bg-gradient-to-b from-zinc-950 to-zinc-900 text-zinc-100 md:w-52 md:border-b-0 md:border-r">
        <div className="flex items-center gap-2 border-b border-zinc-800/90 px-4 py-3">
          <BookOpen className="h-5 w-5 shrink-0 text-emerald-400" aria-hidden />
          <span className="text-[0.65rem] font-semibold uppercase tracking-[0.18em] text-zinc-500">
            Knowledge
          </span>
        </div>
        <div className="flex flex-1 flex-col gap-4 px-4 py-5">
          <div>
            <p className="text-[0.6rem] font-semibold uppercase tracking-wider text-zinc-500">
              Module
            </p>
            <p className="mt-1 text-sm font-medium leading-snug text-white">
              {displayTopic}
            </p>
          </div>
        </div>
      </aside>

      {/* Center: multimodal workspace — tabs, sticky video, tab content */}
      <main className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto bg-white shadow-inner">
        <div className="sticky top-0 z-30 bg-white shadow-[0_8px_30px_-12px_rgba(0,0,0,0.12)]">
          <div
            className="border-b border-zinc-200/90 px-3 py-2.5 sm:px-4"
            role="navigation"
            aria-label="Learning workspace"
          >
            <div className="mx-auto flex max-w-5xl justify-center">
              <div
                className="inline-flex rounded-full bg-zinc-100/95 p-1 ring-1 ring-zinc-200/90"
                role="tablist"
                aria-label="Textbook, slides, or quiz"
              >
                <button
                  type="button"
                  role="tab"
                  aria-selected={workspaceMode === "textbook"}
                  onClick={() => setWorkspaceMode("textbook")}
                  className={`flex items-center gap-1.5 rounded-full px-3 py-2 text-sm font-medium transition sm:px-4 ${
                    workspaceMode === "textbook"
                      ? "bg-white text-zinc-900 shadow-sm ring-1 ring-zinc-200/80"
                      : "text-zinc-600 hover:text-zinc-900"
                  }`}
                >
                  <span aria-hidden>📖</span>
                  <span className="hidden sm:inline">Textbook</span>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={workspaceMode === "slides"}
                  onClick={() => setWorkspaceMode("slides")}
                  className={`flex items-center gap-1.5 rounded-full px-3 py-2 text-sm font-medium transition sm:px-4 ${
                    workspaceMode === "slides"
                      ? "bg-white text-zinc-900 shadow-sm ring-1 ring-zinc-200/80"
                      : "text-zinc-600 hover:text-zinc-900"
                  }`}
                >
                  <span aria-hidden>📽️</span>
                  <span className="hidden sm:inline">AI Slides</span>
                </button>
                <button
                  type="button"
                  role="tab"
                  aria-selected={workspaceMode === "quiz"}
                  onClick={() => setWorkspaceMode("quiz")}
                  className={`flex items-center gap-1.5 rounded-full px-3 py-2 text-sm font-medium transition sm:px-4 ${
                    workspaceMode === "quiz"
                      ? "bg-white text-zinc-900 shadow-sm ring-1 ring-zinc-200/80"
                      : "text-zinc-600 hover:text-zinc-900"
                  }`}
                >
                  <span aria-hidden>📝</span>
                  <span className="hidden sm:inline">Target Quiz</span>
                </button>
              </div>
            </div>
          </div>

          <div className="flex min-h-[3.25rem] items-stretch bg-zinc-950 text-white">
            <div className="flex min-w-0 flex-1 items-center gap-3 px-4 py-3">
              <span
                className="hidden h-9 w-9 shrink-0 items-center justify-center rounded-lg bg-white/10 sm:flex"
                aria-hidden
              >
                <Play className="h-4 w-4 text-emerald-400" />
              </span>
              <span className="min-w-0 text-sm font-semibold tracking-wide sm:text-[0.95rem]">
                WATCH VIDEO: Gordon Growth Model Explained
              </span>
            </div>
            <button
              type="button"
              aria-expanded={videoExpanded}
              aria-label={
                videoExpanded ? "Collapse video panel" : "Expand video panel"
              }
              onClick={(e) => {
                e.currentTarget.blur();
                setVideoExpanded((v) => !v);
              }}
              className="flex w-12 shrink-0 items-center justify-center border-l border-white/10 text-zinc-200 transition hover:bg-zinc-900 hover:text-white"
            >
              <ChevronDown
                className={`h-5 w-5 transition-transform duration-300 ${
                  videoExpanded ? "rotate-180" : ""
                }`}
                aria-hidden
              />
            </button>
          </div>
          {showVideoPlayer ? (
            <div
              className={`grid transition-[grid-template-rows] duration-300 ease-out ${
                videoExpanded ? "grid-rows-[1fr]" : "grid-rows-[0fr]"
              }`}
            >
              <div className="min-h-0 overflow-hidden border-b border-zinc-800 bg-black">
                <div
                  className="relative mx-auto w-full max-w-5xl"
                  style={{ aspectRatio: "16 / 9" }}
                >
                  <iframe
                    title="Gordon Growth Model Explained"
                    src={videoEmbedUrl}
                    className="absolute inset-0 h-full w-full border-0"
                    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                    allowFullScreen
                    loading="lazy"
                    referrerPolicy="strict-origin-when-cross-origin"
                  />
                </div>
              </div>
            </div>
          ) : null}
        </div>

        <div
          ref={readingRef}
          onMouseUp={handleReadingMouseUp}
          className="flex min-h-0 w-full flex-1 flex-col px-3 py-3 sm:px-4 sm:py-4 lg:px-5"
        >
          {/* All panels stay mounted; visibility toggles to avoid refetch on tab switch (V12) */}
          <div
            className={
              workspaceMode === "textbook"
                ? "flex min-h-0 flex-1 flex-col"
                : "hidden"
            }
            aria-hidden={workspaceMode !== "textbook"}
          >
            <TextbookPdf />
          </div>
          <div
            className={
              workspaceMode === "slides"
                ? "flex min-h-0 flex-1 flex-col"
                : "hidden"
            }
            aria-hidden={workspaceMode !== "slides"}
          >
            <AiSlides onOpenChatWithMessage={openChatWithMessage} />
          </div>
          <div
            className={
              workspaceMode === "quiz"
                ? "flex min-h-0 flex-1 flex-col"
                : "hidden"
            }
            aria-hidden={workspaceMode !== "quiz"}
          >
            <AiQuiz onOpenChatWithMessage={openChatWithMessage} />
          </div>
        </div>
      </main>

      {/* Backdrop (chat drawer) */}
      <button
        type="button"
        aria-label="Close chat overlay"
        tabIndex={chatOpen ? 0 : -1}
        aria-hidden={!chatOpen}
        className={`fixed inset-0 z-40 bg-zinc-900/40 backdrop-blur-[2px] transition-opacity duration-300 ${
          chatOpen
            ? "pointer-events-auto opacity-100"
            : "pointer-events-none opacity-0"
        }`}
        onClick={() => setChatOpen(false)}
      />

      {/* Slide-out chat drawer */}
      <aside
        className={`fixed top-0 right-0 z-50 flex h-full w-full max-w-md flex-col border-l border-zinc-200 bg-white shadow-[-12px_0_48px_rgba(0,0,0,0.12)] transition-transform duration-300 ease-out ${
          chatOpen
            ? "translate-x-0 pointer-events-auto"
            : "translate-x-full pointer-events-none"
        }`}
        aria-hidden={!chatOpen}
      >
        <div className="flex shrink-0 items-center justify-between border-b border-zinc-200 px-4 py-3">
          <div className="flex items-center gap-2">
            <MessageCircle className="h-5 w-5 text-emerald-600" aria-hidden />
            <h2 className="text-sm font-semibold text-zinc-800">AI Tutor</h2>
          </div>
          <button
            type="button"
            onClick={() => setChatOpen(false)}
            className="rounded-lg p-2 text-zinc-500 transition hover:bg-zinc-100 hover:text-zinc-900"
            aria-label="Close chat"
          >
            <X className="h-5 w-5" />
          </button>
        </div>

        <div className="min-h-0 flex-1 overflow-y-auto px-4 py-4">
          {chatError ? (
            <p className="rounded-lg border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
              {chatError}
            </p>
          ) : null}

          <ul className="flex flex-col gap-3">
            {messages.map((m, i) => {
              const isUser = m.role === "user";
              const isLastAssistant = i === lastIdx;
              return (
                <li key={`${i}-${m.role}`} className="flex flex-col gap-2">
                  <div
                    className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[min(100%,100%)] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed whitespace-pre-wrap ${
                        isUser
                          ? "bg-emerald-600 text-white"
                          : "border border-zinc-200 bg-zinc-50 text-zinc-900"
                      }`}
                    >
                      {m.content}
                    </div>
                  </div>
                  {isLastAssistant && showQuizButtons ? (
                    <div className="flex flex-wrap gap-2">
                      {(["A", "B", "C", "D"] as const).map((letter) => (
                        <button
                          key={letter}
                          type="button"
                          disabled={isLoading}
                          onClick={() => handleQuizPick(letter)}
                          className="min-w-[2.75rem] rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-sm font-semibold text-zinc-800 shadow-sm transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50"
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
            <p className="mt-4 flex items-center gap-2 text-sm text-zinc-500">
              <Sparkles className="h-4 w-4 animate-pulse text-emerald-500" />
              Thinking…
            </p>
          ) : null}
        </div>

        <div className="shrink-0 border-t border-zinc-200 p-3">
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
              placeholder="Type a message…"
              disabled={isLoading}
              className="min-w-0 flex-1 rounded-lg border border-zinc-300 px-3 py-2 text-sm outline-none ring-emerald-500 focus:ring-2 disabled:bg-zinc-100"
            />
            <button
              type="button"
              onClick={handleSend}
              disabled={isLoading || tutorState === null}
              className="inline-flex shrink-0 items-center gap-1.5 rounded-lg bg-emerald-600 px-3 py-2 text-sm font-medium text-white transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-50"
            >
              <Send className="h-4 w-4" aria-hidden />
              Send
            </button>
          </div>
        </div>
      </aside>

      {!chatOpen ? (
        <button
          type="button"
          onClick={() => setChatOpen(true)}
          className="fixed bottom-6 right-6 z-50 flex h-14 w-14 items-center justify-center rounded-full bg-emerald-600 text-white shadow-lg shadow-emerald-600/35 ring-4 ring-white transition hover:scale-105 hover:bg-emerald-700 focus:outline-none focus-visible:ring-4 focus-visible:ring-emerald-400/50"
          aria-label="Open AI tutor chat"
        >
          <MessageCircle className="h-7 w-7" strokeWidth={1.75} />
        </button>
      ) : null}

      {selectionPopover ? (
        <div
          id="selection-popover"
          ref={popoverRef}
          className="fixed z-[9999] flex flex-col gap-1 rounded-xl border border-zinc-200 bg-white p-1.5 shadow-2xl shadow-zinc-900/20 ring-1 ring-zinc-900/5"
          style={{
            left: selectionPopover.x,
            top: selectionPopover.y,
            transform: selectionPopover.placeAbove
              ? "translate(-50%, calc(-100% - 10px))"
              : "translate(-50%, 10px)",
          }}
          role="toolbar"
          aria-label="Selection actions"
        >
          <button
            type="button"
            className="flex items-center gap-2 rounded-lg px-3 py-2 text-left text-sm font-medium text-zinc-800 transition hover:bg-emerald-50"
            onMouseDown={(e) => e.preventDefault()}
            onClick={() =>
              sendFromSelection(
                `Please explain this concept: '${truncatedForMessage(selectionPopover.text)}'`
              )
            }
          >
            <span aria-hidden>🧠</span> Explain
          </button>
          <button
            type="button"
            className="flex items-center gap-2 rounded-lg px-3 py-2 text-left text-sm font-medium text-zinc-800 transition hover:bg-sky-50"
            onMouseDown={(e) => e.preventDefault()}
            onClick={() =>
              sendFromSelection(
                `Please translate this to Chinese: '${truncatedForMessage(selectionPopover.text)}'`
              )
            }
          >
            <span aria-hidden>🌐</span> Translate
          </button>
        </div>
      ) : null}
    </div>
  );
}
