"use client";

import { useCallback, useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import { API_BASE } from "@/lib/api";

export type SlideBlock =
  | {
      type: "concept";
      title?: string | null;
      content_md: string;
    }
  | {
      type: "mini-quiz";
      content_md: string;
      question: string;
      options: { key: "A" | "B" | "C" | "D"; text: string }[];
      correct_answer: "A" | "B" | "C" | "D";
    }
  | {
      type: "feynman";
      title?: string | null;
      content_md: string;
      prompt: string;
    };

type ApiSlide = {
  type: "concept" | "mini-quiz" | "feynman";
  title?: string | null;
  content_md?: string | null;
  question?: string | null;
  options?: string[] | null;
  correct_answer?: "A" | "B" | "C" | "D" | null;
  prompt?: string | null;
};

const KEYS = ["A", "B", "C", "D"] as const;

function mapApiSlideToBlock(s: ApiSlide): SlideBlock | null {
  const md = (s.content_md ?? "").trim();
  if (!md) {
    return null;
  }
  if (s.type === "concept") {
    return { type: "concept", title: s.title, content_md: s.content_md! };
  }
  if (
    s.type === "mini-quiz" &&
    s.question &&
    s.options &&
    s.options.length === 4 &&
    s.correct_answer &&
    (s.correct_answer === "A" ||
      s.correct_answer === "B" ||
      s.correct_answer === "C" ||
      s.correct_answer === "D")
  ) {
    return {
      type: "mini-quiz",
      content_md: s.content_md!,
      question: s.question,
      options: KEYS.map((k, i) => ({ key: k, text: s.options![i] })),
      correct_answer: s.correct_answer,
    };
  }
  if (s.type === "feynman" && s.prompt) {
    return {
      type: "feynman",
      title: s.title,
      content_md: s.content_md!,
      prompt: s.prompt,
    };
  }
  return null;
}

function mapDeck(deck: { slides?: ApiSlide[] }): SlideBlock[] {
  if (!deck.slides?.length) {
    return [];
  }
  const out: SlideBlock[] = [];
  for (const s of deck.slides) {
    const b = mapApiSlideToBlock(s);
    if (b) {
      out.push(b);
    }
  }
  return out;
}

const mdPlugins = [remarkGfm, remarkMath];
const rehypePlugins = [rehypeKatex];

function SlideMarkdown({ children }: { children: string }) {
  return (
    <div className="slide-markdown max-w-none select-text">
      <ReactMarkdown
        remarkPlugins={mdPlugins}
        rehypePlugins={rehypePlugins}
      >
        {children}
      </ReactMarkdown>
    </div>
  );
}

type AiSlidesProps = {
  onOpenChatWithMessage?: (message: string) => void;
};

export function AiSlides({ onOpenChatWithMessage }: AiSlidesProps) {
  const [deck, setDeck] = useState<SlideBlock[]>([]);
  const [index, setIndex] = useState(0);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [mqSelected, setMqSelected] = useState<"A" | "B" | "C" | "D" | null>(
    null
  );
  const [mqSubmitted, setMqSubmitted] = useState(false);
  const [feynmanDraft, setFeynmanDraft] = useState("");

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    setDeck([]);
    setIndex(0);
    setMqSelected(null);
    setMqSubmitted(false);
    setFeynmanDraft("");
    try {
      const res = await fetch(`${API_BASE}/api/generate_slides`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          topic: "Discounted Dividend Valuation",
        }),
      });
      const data = (await res.json()) as {
        ok?: boolean;
        error?: string;
        deck?: { slides?: ApiSlide[] };
      };
      if (!res.ok || !data.ok || !data.deck) {
        setError(data.error || `Request failed (${res.status})`);
        setDeck([]);
        return;
      }
      const blocks = mapDeck(data.deck);
      if (blocks.length === 0) {
        setError(
          "The AI returned slides we could not display. Please try again."
        );
        setDeck([]);
        return;
      }
      setDeck(blocks);
      setIndex(0);
    } catch {
      setError("Unable to reach the server. Is the backend running?");
      setDeck([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    load();
  }, [load]);

  useEffect(() => {
    setMqSelected(null);
    setMqSubmitted(false);
    setFeynmanDraft("");
  }, [index]);

  const slide = deck[index];
  const canPrev = index > 0;
  const canNext = index < deck.length - 1;

  const handleFeynmanDiscuss = () => {
    const t = feynmanDraft.trim();
    if (!t || !onOpenChatWithMessage) {
      return;
    }
    onOpenChatWithMessage(
      `Here is my Feynman summary for this topic: ${t}. Please review it.`
    );
  };

  const handleMqSubmit = () => {
    if (mqSelected == null || slide?.type !== "mini-quiz") {
      return;
    }
    setMqSubmitted(true);
  };

  if (loading) {
    return (
      <div className="mx-auto w-full max-w-5xl space-y-6 pb-4">
        <div
          className="relative flex w-full flex-col items-center justify-center overflow-hidden rounded-2xl border border-zinc-200/90 bg-gradient-to-b from-zinc-50 to-zinc-100/80 shadow-inner ring-1 ring-zinc-900/5"
          style={{ aspectRatio: "16 / 9", minHeight: "220px" }}
        >
          <div className="h-10 w-10 animate-spin rounded-full border-2 border-emerald-600 border-t-transparent" />
          <p className="mt-4 max-w-md px-6 text-center text-sm leading-relaxed text-zinc-600">
            AI is analyzing the textbook and generating your slides…
          </p>
        </div>
        <div className="space-y-2 px-2">
          <div className="h-3 w-full animate-pulse rounded bg-zinc-200/80" />
          <div className="h-3 w-[85%] animate-pulse rounded bg-zinc-200/60" />
        </div>
      </div>
    );
  }

  if (error || !slide) {
    return (
      <div className="mx-auto max-w-lg rounded-2xl border border-amber-200 bg-amber-50 px-6 py-8 text-center">
        <p className="text-sm font-medium text-amber-950">{error}</p>
        <button
          type="button"
          onClick={() => load()}
          className="mt-4 rounded-full bg-emerald-600 px-5 py-2 text-sm font-semibold text-white shadow-md transition hover:bg-emerald-700"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="mx-auto w-full max-w-5xl pb-4">
      <div className="mb-3 flex flex-wrap items-center justify-end gap-2">
        <button
          type="button"
          onClick={() => load()}
          className="inline-flex items-center gap-2 rounded-full border border-zinc-300 bg-white px-4 py-2 text-xs font-semibold uppercase tracking-wide text-zinc-700 shadow-sm transition hover:border-emerald-400 hover:bg-emerald-50/80 hover:text-emerald-900"
        >
          <span aria-hidden>🔄</span> Re-generate
        </button>
      </div>
      <div
        className="relative w-full overflow-hidden rounded-2xl border border-zinc-300/80 bg-[#faf9f7] shadow-[0_20px_50px_-20px_rgba(0,0,0,0.25)] ring-1 ring-black/5"
        style={{ aspectRatio: "16 / 9" }}
      >
        <div className="h-full overflow-y-auto px-6 py-6 sm:px-10 sm:py-8 md:px-14 md:py-10">
          {slide.type === "concept" ? (
            <article>
              {slide.title ? (
                <h2 className="mb-3 font-serif text-2xl font-semibold tracking-tight text-zinc-950 sm:text-3xl md:text-[2rem]">
                  {slide.title}
                </h2>
              ) : null}
              {slide.title ? (
                <div className="mb-6 h-px w-16 bg-zinc-800/80" aria-hidden />
              ) : null}
              <SlideMarkdown>{slide.content_md}</SlideMarkdown>
            </article>
          ) : null}

          {slide.type === "mini-quiz" ? (
            <article>
              <p className="font-serif text-xl font-semibold leading-snug text-zinc-950 sm:text-2xl">
                Quick check
              </p>
              <div className="mt-4">
                <SlideMarkdown>{slide.content_md}</SlideMarkdown>
              </div>
              <p className="mt-6 font-sans text-base font-medium leading-relaxed text-zinc-900 sm:text-lg">
                {slide.question}
              </p>
              <div className="mt-6 space-y-3 font-sans">
                {slide.options.map((opt) => (

                  <label
                    key={opt.key}
                    className={`flex cursor-pointer items-start gap-3 rounded-xl border px-4 py-3 text-left text-sm shadow-sm transition sm:text-base ${
                      mqSelected === opt.key
                        ? "border-emerald-500 bg-emerald-50/80 ring-1 ring-emerald-500/30"
                        : "border-zinc-200/90 bg-white/80 hover:border-zinc-300"
                    } ${
                      mqSubmitted && opt.key === slide.correct_answer
                        ? "border-emerald-600 bg-emerald-50"
                        : ""
                    } ${
                      mqSubmitted &&
                      mqSelected === opt.key &&
                      opt.key !== slide.correct_answer
                        ? "border-red-300 bg-red-50/50"
                        : ""
                    }`}
                  >
                    <input
                      type="radio"
                      name={`slide-mcq-${index}`}
                      className="mt-1 h-4 w-4 shrink-0 accent-emerald-600"
                      checked={mqSelected === opt.key}
                      disabled={mqSubmitted}
                      onChange={() => setMqSelected(opt.key)}
                    />
                    <span className="flex-1 text-zinc-800">
                      <span className="font-semibold tabular-nums text-zinc-500">
                        {opt.key}.{" "}
                      </span>
                      {opt.text}
                    </span>
                  </label>

                ))}
              </div>
              <div className="mt-6 flex flex-wrap items-center gap-3">
                <button
                  type="button"
                  disabled={mqSelected == null || mqSubmitted}
                  onClick={handleMqSubmit}
                  className="rounded-xl bg-emerald-600 px-5 py-2.5 text-sm font-semibold text-white shadow-md shadow-emerald-600/25 transition hover:bg-emerald-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Submit
                </button>
                {mqSubmitted ? (
                  <p
                    className={`text-sm font-semibold ${
                      mqSelected === slide.correct_answer
                        ? "text-emerald-700"
                        : "text-red-700"
                    }`}
                  >
                    {mqSelected === slide.correct_answer
                      ? "Correct."
                      : "Incorrect."}
                  </p>
                ) : null}
              </div>
            </article>
          ) : null}

          {slide.type === "feynman" ? (
            <article>
              {slide.title ? (
                <h2 className="mb-3 font-serif text-2xl font-semibold tracking-tight text-zinc-950 sm:text-3xl">
                  {slide.title}
                </h2>
              ) : null}
              {slide.title ? (
                <div className="mb-6 h-px w-16 bg-emerald-800/70" aria-hidden />
              ) : null}
              <SlideMarkdown>{slide.content_md}</SlideMarkdown>
              <div className="mt-8 rounded-xl border border-dashed border-zinc-400/60 bg-white/70 px-5 py-4">
                <p className="text-[0.65rem] font-semibold uppercase tracking-[0.2em] text-zinc-500">
                  Reflection
                </p>
                <div className="mt-2">
                  <SlideMarkdown>{slide.prompt}</SlideMarkdown>
                </div>
              </div>
              <div className="mt-6 space-y-3">
                <label className="block text-sm font-medium text-zinc-700">
                  Your explanation (Feynman)
                </label>
                <textarea
                  value={feynmanDraft}
                  onChange={(e) => setFeynmanDraft(e.target.value)}
                  rows={6}
                  placeholder="Explain the idea in your own words, as if teaching someone else…"
                  className="w-full resize-y rounded-xl border border-zinc-300 bg-white px-4 py-3 text-sm leading-relaxed text-zinc-900 shadow-inner outline-none ring-emerald-500/20 focus:border-emerald-500 focus:ring-2"
                />
                <button
                  type="button"
                  disabled={
                    feynmanDraft.trim() === "" || !onOpenChatWithMessage
                  }
                  onClick={handleFeynmanDiscuss}
                  className="rounded-xl bg-zinc-900 px-5 py-2.5 text-sm font-semibold text-white shadow-md transition hover:bg-zinc-800 disabled:cursor-not-allowed disabled:opacity-40"
                >
                  Discuss with Tutor
                </button>
              </div>
            </article>
          ) : null}
        </div>
      </div>

      <div className="mt-6 flex flex-wrap items-center justify-center gap-3">
        <button
          type="button"
          disabled={!canPrev}
          onClick={() => setIndex((i) => Math.max(0, i - 1))}
          className="inline-flex min-h-[2.5rem] items-center gap-2 rounded-full border border-zinc-300 bg-white px-5 py-2 text-sm font-medium text-zinc-800 shadow-sm transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          <span aria-hidden>‹</span> Prev
        </button>
        <span className="text-xs font-medium tabular-nums text-zinc-500">
          {index + 1} / {deck.length}
        </span>
        <button
          type="button"
          disabled={!canNext}
          onClick={() => setIndex((i) => Math.min(deck.length - 1, i + 1))}
          className="inline-flex min-h-[2.5rem] items-center gap-2 rounded-full border border-zinc-300 bg-white px-5 py-2 text-sm font-medium text-zinc-800 shadow-sm transition hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Next <span aria-hidden>›</span>
        </button>
      </div>
    </div>
  );
}
