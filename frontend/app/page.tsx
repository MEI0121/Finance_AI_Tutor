"use client";

import {
  BookOpen,
  ChevronDown,
  MessageCircle,
  NotebookPen,
  Send,
  Sparkles,
  Target,
  UserRound,
  Video,
  Settings,
  X,
} from "lucide-react";
import dynamic from "next/dynamic";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
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

/** Split tutor reply: main markdown vs clickable follow-up pills (must match backend). */
const CHAT_SUGGESTIONS_MARKER = "---SUGGESTIONS---";

function parseChatSuggestions(content: string): {
  mainContent: string;
  suggestions: string[];
} {
  const idx = content.indexOf(CHAT_SUGGESTIONS_MARKER);
  if (idx === -1) {
    return { mainContent: content, suggestions: [] };
  }
  const mainContent = content.slice(0, idx).trimEnd();
  const tail = content.slice(idx + CHAT_SUGGESTIONS_MARKER.length).trim();
  const suggestions: string[] = [];
  for (const rawLine of tail.split(/\r?\n/)) {
    const line = rawLine.trim();
    if (/^(\*|-)\s+/.test(line)) {
      suggestions.push(line.replace(/^(\*|-)\s+/, "").trim());
    }
  }
  return { mainContent, suggestions };
}

/** Bootstrap user message from run_tutor_flow; hidden in UI but kept in state. */
const HIDDEN_BOOTSTRAP_USER_MESSAGE = "Please begin the lesson.";

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

const ACTIVE_COURSE_NAME = "CFA Level II: Equity Valuation";
const ACTIVE_CHAPTER_NAME = "Reading 22: Discounted Dividend Valuation";
const MOCK_CHAPTERS = [
  { id: "r22", label: ACTIVE_CHAPTER_NAME, active: true },
  { id: "r23", label: "Reading 23: Free Cash Flow Valuation", active: false },
  { id: "r24", label: "Reading 24: Market-Based Valuation", active: false },
] as const;

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

/** Selection anchored inside UI chrome (pagination, toolbars) — no Point-and-Read popover. */
/** YouTube embed must include enablejsapi=1 for iframe postMessage commands. */
function ensureYouTubeJsApi(embedUrl: string): string {
  const trimmed = embedUrl.trim();
  if (!trimmed) {
    return trimmed;
  }
  try {
    const u = new URL(trimmed);
    if (!u.searchParams.has("enablejsapi")) {
      u.searchParams.set("enablejsapi", "1");
    }
    return u.toString();
  } catch {
    return trimmed.includes("enablejsapi=")
      ? trimmed
      : `${trimmed}${trimmed.includes("?") ? "&" : "?"}enablejsapi=1`;
  }
}

function selectionTouchesExcludedChrome(container: Node): boolean {
  const el =
    container.nodeType === Node.TEXT_NODE
      ? container.parentElement
      : container instanceof Element
        ? container
        : null;
  if (!el) {
    return false;
  }
  return el.closest("button") != null || el.closest(".no-popover") != null;
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
  const [courseMenuOpen, setCourseMenuOpen] = useState(false);
  const [profileMenuOpen, setProfileMenuOpen] = useState(false);

  const readingRef = useRef<HTMLDivElement>(null);
  const popoverRef = useRef<HTMLDivElement>(null);
  const videoIframeRef = useRef<HTMLIFrameElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const courseMenuRef = useRef<HTMLDivElement>(null);
  const profileMenuRef = useRef<HTMLDivElement>(null);
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
  const videoEmbedUrlWithJsApi = useMemo(
    () => ensureYouTubeJsApi(videoEmbedUrl),
    [videoEmbedUrl]
  );
  const showVideoPlayer = videoEmbedUrl.length > 0;

  useEffect(() => {
    if (videoExpanded) {
      return;
    }
    const win = videoIframeRef.current?.contentWindow;
    if (!win) {
      return;
    }
    try {
      win.postMessage(
        JSON.stringify({
          event: "command",
          func: "pauseVideo",
          args: "",
        }),
        "*"
      );
    } catch {
      /* ignore */
    }
  }, [videoExpanded]);

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

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, isLoading]);

  const submitChatMessage = useCallback(
    (text: string) => {
      if (isLoading || tutorState === null) {
        return;
      }
      const trimmed = text.trim();
      if (trimmed === "") {
        return;
      }
      sendToTutor(trimmed, tutorState);
    },
    [isLoading, sendToTutor, tutorState]
  );

  const handleSend = () => {
    const trimmed = input.trim();
    if (trimmed === "") {
      return;
    }
    setInput("");
    submitChatMessage(trimmed);
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
      if (workspaceMode === "quiz") {
        setSelectionPopover(null);
        return;
      }
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
      if (selectionTouchesExcludedChrome(range.commonAncestorContainer)) {
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
  }, [workspaceMode]);

  useEffect(() => {
    if (workspaceMode === "quiz") {
      setSelectionPopover(null);
    }
  }, [workspaceMode]);

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

  useEffect(() => {
    if (!courseMenuOpen) {
      return;
    }
    const onDocMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (!courseMenuRef.current?.contains(target)) {
        setCourseMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [courseMenuOpen]);

  useEffect(() => {
    if (!profileMenuOpen) {
      return;
    }
    const onDocMouseDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (!profileMenuRef.current?.contains(target)) {
        setProfileMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onDocMouseDown);
    return () => document.removeEventListener("mousedown", onDocMouseDown);
  }, [profileMenuOpen]);

  const truncatedForMessage = (t: string, max = 1200) =>
    t.length > max ? `${t.slice(0, max)}…` : t;

  return (
    <div className="flex h-screen min-h-0 flex-col bg-zinc-200/80 text-zinc-900 md:flex-row">
      {/* Left: Knowledge sidebar */}
      <aside className="flex w-full shrink-0 flex-col border-b border-zinc-800 bg-gradient-to-b from-zinc-950 to-zinc-900 text-zinc-100 md:w-52 md:border-b-0 md:border-r">
        <div className="flex items-center gap-2 border-b border-zinc-800/90 px-4 py-3">
          <BookOpen className="h-5 w-5 shrink-0 text-emerald-400" aria-hidden />
          <span className="text-[0.66rem] font-semibold tracking-[0.08em] text-zinc-300">
            Finance AI Tutor
          </span>
        </div>
        <div className="flex flex-1 flex-col gap-4 px-4 py-4">
          <div
            ref={courseMenuRef}
            className="relative rounded-xl border border-zinc-700/90 bg-zinc-900/70 px-3 py-3 shadow-sm ring-1 ring-zinc-700/50"
          >
            <button
              type="button"
              onClick={() => setCourseMenuOpen((open) => !open)}
              className="flex w-full items-start justify-between gap-2 text-left"
              aria-expanded={courseMenuOpen}
              aria-haspopup="menu"
              aria-label="Open course and chapter selector"
            >
              <div className="min-w-0">
                <p className="text-[0.58rem] font-semibold uppercase tracking-[0.14em] text-zinc-400">
                  {ACTIVE_COURSE_NAME}
                </p>
                <p className="mt-1 text-xs font-semibold leading-snug text-zinc-100">
                  {ACTIVE_CHAPTER_NAME}
                </p>
              </div>
              <ChevronDown
                className={`mt-0.5 h-4 w-4 shrink-0 text-zinc-400 transition-transform ${
                  courseMenuOpen ? "rotate-180" : ""
                }`}
                aria-hidden
              />
            </button>

            {courseMenuOpen ? (
              <div
                role="menu"
                className="absolute left-0 right-0 top-full z-30 mt-2 rounded-xl border border-zinc-700/90 bg-zinc-900 p-2 shadow-2xl ring-1 ring-zinc-700/60"
              >
                {MOCK_CHAPTERS.map((chapter) => (
                  <button
                    key={chapter.id}
                    type="button"
                    role="menuitem"
                    onClick={() => {
                      if (chapter.active) {
                        setCourseMenuOpen(false);
                        return;
                      }
                      setCourseMenuOpen(false);
                      alert("Available in full version");
                    }}
                    className={`mb-1 flex w-full items-center justify-between rounded-lg px-2.5 py-2 text-left text-xs transition last:mb-0 ${
                      chapter.active
                        ? "cursor-default bg-emerald-500/15 text-emerald-200"
                        : "bg-zinc-800/85 text-zinc-400 hover:bg-zinc-800"
                    }`}
                  >
                    <span className="pr-2 leading-snug">{chapter.label}</span>
                    {chapter.active ? (
                      <span className="shrink-0 rounded-full bg-emerald-500/25 px-1.5 py-0.5 text-[0.58rem] font-semibold uppercase tracking-wide text-emerald-200">
                        Active
                      </span>
                    ) : (
                      <span className="shrink-0 rounded-full border border-zinc-600 bg-zinc-900 px-1.5 py-0.5 text-[0.58rem] font-semibold uppercase tracking-wide text-zinc-400">
                        🔒 Pro
                      </span>
                    )}
                  </button>
                ))}
              </div>
            ) : null}
          </div>
          <div ref={profileMenuRef} className="relative mt-auto">
            <button
              type="button"
              onClick={() => setProfileMenuOpen((open) => !open)}
              className="flex w-full items-center gap-2 rounded-xl border border-zinc-700/90 bg-zinc-900/70 px-2.5 py-2 text-left shadow-sm ring-1 ring-zinc-700/40 transition hover:bg-zinc-900"
              aria-haspopup="menu"
              aria-expanded={profileMenuOpen}
            >
              <div className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full bg-emerald-500/20 text-xs font-bold text-emerald-200 ring-1 ring-emerald-500/30">
                JL
              </div>
              <div className="min-w-0 flex-1">
                <p className="truncate text-xs font-semibold text-zinc-100">
                  Learner #42
                </p>
                <p className="text-[0.62rem] text-zinc-400">Workspace</p>
              </div>
              <ChevronDown
                className={`h-4 w-4 shrink-0 text-zinc-400 transition-transform ${
                  profileMenuOpen ? "rotate-180" : ""
                }`}
                aria-hidden
              />
            </button>
            {profileMenuOpen ? (
              <div
                role="menu"
                className="absolute bottom-full left-0 right-0 z-30 mb-2 rounded-xl border border-zinc-700/90 bg-zinc-900 p-1.5 shadow-2xl ring-1 ring-zinc-700/60"
              >
                <button
                  type="button"
                  role="menuitem"
                  className="mb-1 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-xs text-zinc-200 transition hover:bg-zinc-800 last:mb-0"
                  onClick={() => setProfileMenuOpen(false)}
                >
                  <NotebookPen className="h-3.5 w-3.5 text-zinc-300" aria-hidden />
                  My Notebooks
                </button>
                <button
                  type="button"
                  role="menuitem"
                  className="mb-1 flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-xs text-zinc-200 transition hover:bg-zinc-800 last:mb-0"
                  onClick={() => setProfileMenuOpen(false)}
                >
                  <Target className="h-3.5 w-3.5 text-zinc-300" aria-hidden />
                  Practice Bank
                </button>
                <button
                  type="button"
                  role="menuitem"
                  className="flex w-full items-center gap-2 rounded-lg px-2.5 py-2 text-xs text-zinc-200 transition hover:bg-zinc-800"
                  onClick={() => setProfileMenuOpen(false)}
                >
                  <Settings className="h-3.5 w-3.5 text-zinc-300" aria-hidden />
                  Settings
                </button>
              </div>
            ) : null}
          </div>
        </div>
      </aside>

      {/* Center: multimodal workspace — unified header + video, tab content */}
      <main className="relative flex min-h-0 min-w-0 flex-1 flex-col overflow-y-auto bg-white shadow-inner">
        <div className="sticky top-0 z-30 border-b border-zinc-200/80 bg-white/90 shadow-sm backdrop-blur-md">
          <div
            className="mx-auto grid max-w-5xl grid-cols-[1fr_auto_1fr] items-center gap-3 p-4"
            role="navigation"
            aria-label="Learning workspace"
          >
            <div className="min-w-0" aria-hidden />
            <div className="flex justify-center">
              <div
                className="inline-flex rounded-full bg-zinc-100/95 p-1 ring-1 ring-zinc-200/90"
                role="tablist"
                aria-label="Textbook, slides, or AI quiz"
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
                  <span className="hidden sm:inline">AI Quiz</span>
                </button>
              </div>
            </div>
            <div className="flex min-w-0 justify-end">
              {showVideoPlayer ? (
                <button
                  type="button"
                  aria-expanded={videoExpanded}
                  aria-label={
                    videoExpanded
                      ? "Collapse video tutorial"
                      : "Expand video tutorial"
                  }
                  onClick={(e) => {
                    e.currentTarget.blur();
                    setVideoExpanded((v) => !v);
                  }}
                  className="inline-flex shrink-0 items-center gap-2 rounded-full border border-zinc-200/90 bg-white/95 px-3.5 py-2 text-sm font-medium text-zinc-700 shadow-sm ring-1 ring-zinc-900/5 transition hover:border-zinc-300 hover:bg-zinc-50 hover:text-zinc-900"
                >
                  <Video className="h-4 w-4 shrink-0 text-emerald-600" aria-hidden />
                  <span className="hidden sm:inline">Watch video</span>
                  <ChevronDown
                    className={`h-4 w-4 shrink-0 text-zinc-500 transition-transform duration-300 ${
                      videoExpanded ? "rotate-180" : ""
                    }`}
                    aria-hidden
                  />
                </button>
              ) : null}
            </div>
          </div>
          {showVideoPlayer ? (
            <div
              className={`absolute left-0 right-0 top-full z-40 overflow-hidden bg-zinc-950 shadow-2xl transition-[max-height] duration-300 ease-out ${
                videoExpanded
                  ? "pointer-events-auto max-h-[min(85vh,calc(100vw*0.5625))]"
                  : "pointer-events-none max-h-0"
              }`}
            >
              <div className="relative mx-auto w-full max-w-5xl aspect-video">
                <iframe
                  ref={videoIframeRef}
                  title="Video tutorial"
                  src={videoEmbedUrlWithJsApi}
                  className="absolute inset-0 h-full w-full border-0"
                  allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
                  allowFullScreen
                  loading="lazy"
                  referrerPolicy="strict-origin-when-cross-origin"
                />
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
              if (
                m.role === "user" &&
                m.content === HIDDEN_BOOTSTRAP_USER_MESSAGE
              ) {
                return null;
              }
              const isUser = m.role === "user";
              const isLastAssistant = i === lastIdx;
              const assistantParts =
                !isUser && m.content.includes(CHAT_SUGGESTIONS_MARKER)
                  ? parseChatSuggestions(m.content)
                  : { mainContent: m.content, suggestions: [] as string[] };
              return (
                <li key={`${i}-${m.role}`} className="flex flex-col gap-2">
                  <div
                    className={`flex ${isUser ? "justify-end" : "justify-start"}`}
                  >
                    <div
                      className={`max-w-[min(100%,100%)] rounded-2xl px-3.5 py-2.5 text-sm leading-relaxed ${
                        isUser
                          ? "whitespace-pre-wrap bg-emerald-600 text-white"
                          : "border border-zinc-200 bg-zinc-50 text-zinc-900"
                      }`}
                    >
                      {isUser ? (
                        m.content
                      ) : (
                        <div className="prose prose-sm prose-zinc max-w-none [&_p]:my-2 [&_p:first-child]:mt-0 [&_p:last-child]:mb-0 [&_ul]:my-2 [&_ol]:my-2 [&_li]:my-0.5 [&_code]:rounded-md [&_code]:bg-zinc-200/70 [&_code]:px-1.5 [&_code]:py-0.5 [&_code]:text-[0.8125rem] [&_pre]:my-2 [&_pre]:max-w-full [&_pre]:overflow-x-auto [&_pre]:rounded-lg [&_pre]:bg-zinc-900/90 [&_pre]:p-3 [&_pre]:text-zinc-100 [&_blockquote]:my-2 [&_blockquote]:border-l-zinc-300">
                          <ReactMarkdown remarkPlugins={[remarkGfm]}>
                            {assistantParts.mainContent}
                          </ReactMarkdown>
                        </div>
                      )}
                    </div>
                  </div>
                  {!isUser && assistantParts.suggestions.length > 0 ? (
                    <div className="flex flex-row flex-wrap gap-2 pl-0.5">
                      {assistantParts.suggestions.map((label, si) => (
                        <button
                          key={`${i}-sugg-${si}`}
                          type="button"
                          disabled={isLoading || tutorState === null}
                          onClick={() => submitChatMessage(label)}
                          className="cursor-pointer rounded-full border border-zinc-200/90 bg-zinc-100 px-3 py-1.5 text-left text-xs text-zinc-700 transition-colors hover:bg-zinc-200 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                          {label}
                        </button>
                      ))}
                    </div>
                  ) : null}
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
          <div ref={messagesEndRef} />

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
                `Please analyze, explain, or solve the following text/exercise extracted from my textbook: '${truncatedForMessage(selectionPopover.text)}'`
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
          <button
            type="button"
            className="flex items-center gap-2 rounded-lg px-3 py-2 text-left text-sm font-medium text-zinc-800 transition hover:bg-amber-50"
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => {
              setSelectionPopover(null);
              alert("Added to notes (PoC mock)");
            }}
          >
            <span aria-hidden>🔖</span> Add to Notes
          </button>
        </div>
      ) : null}
    </div>
  );
}
