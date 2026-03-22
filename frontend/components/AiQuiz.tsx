"use client";

import { useCallback, useEffect, useState } from "react";

import { API_BASE } from "@/lib/api";

export type ChapterId = "single" | "multi" | "terminal";

const CHAPTERS: { id: ChapterId; label: string }[] = [
  { id: "single", label: "1. Single-Stage DDM" },
  { id: "multi", label: "2. Multi-Stage DDM" },
  { id: "terminal", label: "3. Terminal Value & Growth" },
];

const KEYS = ["A", "B", "C", "D"] as const;

type ApiQuestion = {
  question: string;
  options: string[];
  correct_answer: "A" | "B" | "C" | "D";
  explanation: string;
};

type QuizPayload = {
  questions: ApiQuestion[];
};

type AiQuizProps = {
  onOpenChatWithMessage?: (message: string) => void;
};

export function AiQuiz({ onOpenChatWithMessage }: AiQuizProps) {
  const [chapter, setChapter] = useState<ChapterId>("single");
  const [questions, setQuestions] = useState<ApiQuestion[]>([]);
  const [qIndex, setQIndex] = useState(0);
  const [selected, setSelected] = useState<"A" | "B" | "C" | "D" | null>(
    null
  );
  /** Only lock options after a correct submit; wrong answers allow retry. */
  const [feedback, setFeedback] = useState<"idle" | "wrong" | "correct">("idle");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [quizCache, setQuizCache] = useState<
    Partial<Record<ChapterId, ApiQuestion[]>>
  >({});
  const [fetchNonce, setFetchNonce] = useState(0);

  const runFetch = useCallback(async (sub: ChapterId, signal: AbortSignal) => {
    setLoading(true);
    setError(null);
    try {
      const res = await fetch(`${API_BASE}/api/generate_quiz`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: "Discounted Dividend Valuation",
          sub_chapter: sub,
        }),
        signal,
      });
      const data = (await res.json()) as {
        ok?: boolean;
        error?: string;
        quiz?: QuizPayload;
      };
      if (!res.ok || !data.ok || !data.quiz?.questions?.length) {
        if (!signal.aborted) {
          setError(data.error || `Request failed (${res.status})`);
          setQuestions([]);
        }
        return;
      }
      if (data.quiz.questions.length !== 5) {
        if (!signal.aborted) {
          setError("Quiz data was incomplete. Please try again.");
          setQuestions([]);
        }
        return;
      }
      if (!signal.aborted) {
        const qs = data.quiz.questions;
        setQuizCache((prev) => ({ ...prev, [sub]: qs }));
        setQuestions(qs);
        setQIndex(0);
        setSelected(null);
        setFeedback("idle");
        setError(null);
      }
    } catch (e) {
      if (signal.aborted) {
        return;
      }
      if (e instanceof DOMException && e.name === "AbortError") {
        return;
      }
      setError("Unable to reach the server. Is the backend running?");
      setQuestions([]);
    } finally {
      if (!signal.aborted) {
        setLoading(false);
      }
    }
  }, []);

  useEffect(() => {
    const cached = quizCache[chapter];
    if (cached && cached.length === 5) {
      setQuestions(cached);
      setQIndex(0);
      setSelected(null);
      setFeedback("idle");
      setError(null);
      setLoading(false);
      return;
    }

    const controller = new AbortController();
    runFetch(chapter, controller.signal);
    return () => controller.abort();
    // quizCache is read for hits but omitted from deps so a post-fetch cache update
    // does not reset the current question index.
    // eslint-disable-next-line react-hooks/exhaustive-deps -- cache keyed by `chapter` + `fetchNonce`
  }, [chapter, fetchNonce, runFetch]);

  const q = questions[qIndex];
  const canPrevQ = qIndex > 0;
  const canNextQ = qIndex < questions.length - 1;

  const chapterLabel =
    CHAPTERS.find((c) => c.id === chapter)?.label ?? chapter;

  const handleSubmit = () => {
    if (selected == null || !q) {
      return;
    }
    setFeedback(selected === q.correct_answer ? "correct" : "wrong");
  };

  const handleRegenerate = () => {
    setQuizCache((prev) => {
      const next = { ...prev };
      delete next[chapter];
      return next;
    });
    setFetchNonce((n) => n + 1);
  };

  const handleRetry = () => {
    setQuizCache((prev) => {
      const next = { ...prev };
      delete next[chapter];
      return next;
    });
    setFetchNonce((n) => n + 1);
  };

  const handleSocratic = () => {
    if (!q || selected == null || !onOpenChatWithMessage) {
      return;
    }
    const optIdx = KEYS.indexOf(selected);
    const optText = optIdx >= 0 ? q.options[optIdx] : "";
    const selectedLabel = `${selected}. ${optText}`;
    onOpenChatWithMessage(
      `I was doing a quiz on ${chapterLabel} and incorrectly chose ${selectedLabel} for the question: '${q.question}'. Can you guide me to the correct answer using the Socratic method?`
    );
  };

  if (loading) {
    return (
      <div className="mx-auto w-full max-w-3xl space-y-6 pb-6">
        <div className="h-10 w-full max-w-xs animate-pulse rounded-xl bg-zinc-200/90" />
        <div className="rounded-2xl border border-zinc-200 bg-white p-8 shadow-inner">
          <div className="flex flex-col items-center justify-center py-12">
            <div className="h-10 w-10 animate-spin rounded-full border-2 border-emerald-600 border-t-transparent" />
            <p className="mt-4 max-w-sm text-center text-sm text-zinc-600">
              Generating targeted quiz questions from your curriculum…
            </p>
          </div>
        </div>
      </div>
    );
  }

  if (error || !q) {
    return (
      <div className="mx-auto max-w-lg rounded-2xl border border-amber-200 bg-amber-50 px-6 py-8 text-center">
        <p className="text-sm font-medium text-amber-950">{error}</p>
        <button
          type="button"
          onClick={handleRetry}
          className="mt-4 rounded-full bg-emerald-600 px-5 py-2 text-sm font-semibold text-white shadow-md transition hover:bg-emerald-700"
        >
          Retry
        </button>
      </div>
    );
  }

  const isCorrect = feedback === "correct";
  const isWrong = feedback === "wrong";

  return (
    <div className="mx-auto w-full max-w-3xl space-y-6 pb-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
        <div className="flex min-w-0 flex-1 flex-col gap-2 sm:flex-row sm:items-center sm:gap-4">
          <label
            htmlFor="quiz-chapter"
            className="shrink-0 text-sm font-medium text-zinc-700"
          >
            Sub-chapter
          </label>
          <select
            id="quiz-chapter"
            value={chapter}
            onChange={(e) => {
              setChapter(e.target.value as ChapterId);
            }}
            className="w-full min-w-0 rounded-xl border border-zinc-300 bg-white px-4 py-2.5 text-sm font-medium text-zinc-900 shadow-sm outline-none ring-emerald-500/30 focus:border-emerald-500 focus:ring-2 sm:max-w-md"
          >
            {CHAPTERS.map((c) => (
              <option key={c.id} value={c.id}>
                {c.label}
              </option>
            ))}
          </select>
        </div>
        <button
          type="button"
          onClick={handleRegenerate}
          className="inline-flex shrink-0 items-center justify-center gap-2 self-end rounded-full border border-zinc-300 bg-white px-4 py-2 text-xs font-semibold uppercase tracking-wide text-zinc-700 shadow-sm transition hover:border-emerald-400 hover:bg-emerald-50/80 hover:text-emerald-900 sm:self-auto"
        >
          <span aria-hidden>🔄</span> Re-generate
        </button>
      </div>

      <p className="text-center text-xs font-medium tabular-nums text-zinc-500">
        Question {qIndex + 1} of {questions.length}
      </p>

      <div className="rounded-2xl border border-zinc-200 bg-white p-6 shadow-lg shadow-zinc-900/5 ring-1 ring-zinc-900/5 sm:p-8">
        <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-zinc-500">
          Target quiz
        </p>
        <h2 className="mt-2 text-lg font-semibold leading-snug text-zinc-900 sm:text-xl">
          {q.question}
        </h2>

        <div className="mt-6 space-y-3">
          {KEYS.map((key, i) => (
            <label
              key={key}
              className={`flex cursor-pointer items-start gap-3 rounded-xl border px-4 py-3 text-left transition ${
                selected === key
                  ? "border-emerald-500 bg-emerald-50/80 ring-1 ring-emerald-500/30"
                  : "border-zinc-200 bg-zinc-50/50 hover:border-zinc-300"
              } ${
                feedback !== "idle" && key === q.correct_answer
                  ? "border-emerald-600 bg-emerald-50"
                  : ""
              } ${
                feedback !== "idle" &&
                selected === key &&
                key !== q.correct_answer
                  ? "border-red-300 bg-red-50/50"
                  : ""
              }`}
            >
              <input
                type="radio"
                name={`mcq-${qIndex}`}
                className="mt-1 h-4 w-4 shrink-0 accent-emerald-600"
                checked={selected === key}
                disabled={feedback === "correct"}
                onChange={() => {
                  setSelected(key);
                  if (feedback === "wrong") {
                    setFeedback("idle");
                  }
                }}
              />
              <span className="text-sm leading-relaxed text-zinc-800 sm:text-[0.95rem]">
                <span className="font-semibold text-zinc-600">{key}. </span>
                {q.options[i]}
              </span>
            </label>
          ))}
        </div>

        {feedback !== "idle" ? (
          <div
            className={`mt-6 rounded-xl border px-4 py-3 text-sm font-semibold ${
              isCorrect
                ? "border-emerald-200 bg-emerald-50 text-emerald-900"
                : "border-amber-200 bg-amber-50 text-amber-950"
            }`}
            role="status"
          >
            {isCorrect ? "Correct." : "Incorrect."}
          </div>
        ) : null}

        <div className="mt-6 flex flex-wrap items-center gap-3">
          <button
            type="button"
            disabled={selected == null || feedback === "correct"}
            onClick={handleSubmit}
            className="rounded-xl bg-emerald-600 px-6 py-2.5 text-sm font-semibold text-white shadow-md shadow-emerald-600/25 transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-40"
          >
            Submit answer
          </button>
          <div className="flex gap-2">
            <button
              type="button"
              disabled={!canPrevQ}
              onClick={() => {
                setQIndex((i) => i - 1);
                setSelected(null);
                setFeedback("idle");
              }}
              className="rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 disabled:opacity-40"
            >
              Prev Q
            </button>
            <button
              type="button"
              disabled={!canNextQ}
              onClick={() => {
                setQIndex((i) => i + 1);
                setSelected(null);
                setFeedback("idle");
              }}
              className="rounded-lg border border-zinc-300 bg-white px-3 py-1.5 text-xs font-medium text-zinc-700 disabled:opacity-40"
            >
              Next Q
            </button>
          </div>
        </div>

        {isWrong && onOpenChatWithMessage ? (
          <div className="mt-6">
            <button
              type="button"
              onClick={handleSocratic}
              className="w-full rounded-xl border-2 border-emerald-600 bg-emerald-50/90 px-4 py-3.5 text-center text-sm font-semibold text-emerald-950 shadow-sm transition hover:bg-emerald-100 sm:px-6"
            >
              I don&apos;t understand, guide me Socratically
            </button>
          </div>
        ) : null}
      </div>

      <div className="rounded-2xl border border-dashed border-zinc-300 bg-zinc-50/80 px-5 py-6">
        <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-zinc-500">
          Explanation
        </p>
        <p className="mt-2 text-sm leading-relaxed text-zinc-600">
          After you submit, compare your reasoning with the explanation below.
          If you are stuck, use the Socratic button to work through it in chat.
        </p>
        <div className="mt-4 max-h-72 min-h-[4.5rem] overflow-y-auto rounded-xl border border-zinc-200/80 bg-white/90 px-4 py-3 text-sm text-zinc-700">
          {feedback !== "idle" ? (
            <p className="whitespace-pre-wrap leading-relaxed">{q.explanation}</p>
          ) : (
            <span className="text-zinc-500">
              Submit an answer to see the explanation for this question.
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
