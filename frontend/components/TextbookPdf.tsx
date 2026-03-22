"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { Document, Page, pdfjs } from "react-pdf";

import "react-pdf/dist/Page/AnnotationLayer.css";
import "react-pdf/dist/Page/TextLayer.css";

/** Same-origin worker avoids CDN/CORS issues in production builds. */
const PDF_WORKER_URL = "/pdf.worker.min.mjs";

/** Tailwind `gap-4` = 1rem; used when splitting width across two pages. */
const SPREAD_GAP_PX = 16;

export function TextbookPdf() {
  const [numPages, setNumPages] = useState<number | null>(null);
  const [containerWidth, setContainerWidth] = useState<number | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [leftPage, setLeftPage] = useState(1);
  const [pageInput, setPageInput] = useState("1");

  const measureRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    pdfjs.GlobalWorkerOptions.workerSrc = PDF_WORKER_URL;
  }, []);

  useEffect(() => {
    const el = measureRef.current;
    if (!el) {
      return;
    }
    const update = () => {
      const w = el.getBoundingClientRect().width;
      setContainerWidth(w > 0 ? Math.floor(w) : null);
    };
    update();
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const w = entry.contentRect.width;
        setContainerWidth(w > 0 ? Math.floor(w) : null);
      }
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  /** Keep spread in range when document loads or page count changes. */
  useEffect(() => {
    if (numPages == null || numPages < 1) {
      return;
    }
    const maxLeft = Math.max(1, numPages - 1);
    setLeftPage((prev) => {
      const next = Math.max(1, Math.min(prev, maxLeft));
      return next;
    });
  }, [numPages]);

  useEffect(() => {
    setPageInput(String(leftPage));
  }, [leftPage]);

  const commitPageInput = useCallback(() => {
    if (numPages == null || numPages < 1) {
      return;
    }
    const parsed = parseInt(pageInput, 10);
    if (Number.isNaN(parsed)) {
      setPageInput(String(leftPage));
      return;
    }
    const clamped = Math.max(1, Math.min(parsed, numPages - 1));
    setLeftPage(clamped);
    setPageInput(String(clamped));
  }, [numPages, pageInput, leftPage]);

  const canPrev = numPages != null && leftPage > 1;
  const canNext =
    numPages != null && numPages >= 2 && leftPage < numPages - 1;

  const pageWidthEach =
    containerWidth != null && containerWidth > 0
      ? numPages === 1
        ? Math.max(120, containerWidth)
        : Math.max(
            120,
            Math.floor((containerWidth - SPREAD_GAP_PX) / 2)
          )
      : null;

  const showSpread = numPages != null && numPages >= 2;

  return (
    <div
      ref={measureRef}
      className="flex min-h-0 w-full flex-1 flex-col select-text"
    >
      {loadError ? (
        <div className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-6 text-sm text-amber-900">
          <p className="font-medium">Could not load textbook PDF.</p>
          <p className="mt-1 text-amber-800/90">{loadError}</p>
          <p className="mt-2 text-xs text-amber-800/80">
            Add a file named <code className="font-mono">textbook.pdf</code> to
            the <code className="font-mono">public</code> folder and refresh.
          </p>
        </div>
      ) : null}

      {!loadError && pageWidthEach != null ? (
        <Document
          file="/textbook.pdf"
          loading={
            <div className="rounded-xl border border-zinc-200 bg-zinc-50 py-16 text-center text-sm text-zinc-500">
              Loading textbook…
            </div>
          }
          onLoadSuccess={({ numPages: n }) => {
            setNumPages(n);
            setLoadError(null);
          }}
          onLoadError={(err) => {
            setLoadError(err.message || "Failed to open PDF");
            setNumPages(null);
          }}
          className="flex flex-col"
        >
          {numPages != null && numPages > 0 ? (
            <>
              {/* Immersive spread */}
              <div className="rounded-2xl bg-gradient-to-b from-zinc-100/90 to-zinc-200/50 p-3 shadow-[inset_0_1px_0_rgba(255,255,255,0.6)] ring-1 ring-zinc-900/10">
                {showSpread ? (
                  <div className="flex flex-row justify-center gap-4">
                    <div
                      className="shrink-0 overflow-hidden rounded-l-2xl border-r border-zinc-400/35 bg-white shadow-[4px_0_24px_-8px_rgba(0,0,0,0.18)] ring-1 ring-zinc-900/5"
                      style={{ width: pageWidthEach }}
                    >
                      <Page
                        key={`spread-left-${leftPage}`}
                        pageNumber={leftPage}
                        width={pageWidthEach}
                        renderTextLayer
                        renderAnnotationLayer
                        className="[&_.react-pdf__Page__textContent]:select-text [&_.react-pdf__Page__canvas]:bg-white"
                      />
                    </div>
                    <div
                      className="shrink-0 overflow-hidden rounded-r-2xl border-l border-white/80 bg-white shadow-[-4px_0_24px_-8px_rgba(0,0,0,0.18)] ring-1 ring-zinc-900/5"
                      style={{ width: pageWidthEach }}
                    >
                      <Page
                        key={`spread-right-${leftPage + 1}`}
                        pageNumber={leftPage + 1}
                        width={pageWidthEach}
                        renderTextLayer
                        renderAnnotationLayer
                        className="[&_.react-pdf__Page__textContent]:select-text [&_.react-pdf__Page__canvas]:bg-white"
                      />
                    </div>
                  </div>
                ) : (
                  <div className="flex justify-center overflow-hidden rounded-2xl bg-white shadow-xl ring-1 ring-zinc-900/5">
                    <Page
                      pageNumber={1}
                      width={Math.min(pageWidthEach, 900)}
                      renderTextLayer
                      renderAnnotationLayer
                      className="[&_.react-pdf__Page__textContent]:select-text [&_.react-pdf__Page__canvas]:bg-white"
                    />
                  </div>
                )}
              </div>

              {/* Pagination — sticky within reading flow (no-popover: exclude from Point-and-Read) */}
              <div className="no-popover sticky bottom-4 z-20 mt-8 flex justify-center px-1">
                <div className="flex w-full max-w-2xl flex-wrap items-center justify-between gap-3 rounded-2xl border border-zinc-200/90 bg-white/95 px-4 py-3 shadow-[0_8px_30px_-12px_rgba(0,0,0,0.15)] backdrop-blur-md ring-1 ring-zinc-900/5 supports-[backdrop-filter]:bg-white/85">
                  <button
                    type="button"
                    disabled={!canPrev}
                    onClick={() => {
                      setLeftPage((p) => Math.max(1, p - 1));
                    }}
                    className="inline-flex min-h-[2.5rem] items-center gap-2 rounded-xl border border-zinc-200 bg-zinc-50 px-4 py-2 text-sm font-medium text-zinc-800 shadow-sm transition hover:bg-white hover:shadow disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    <span aria-hidden className="text-zinc-500">
                      ‹
                    </span>
                    Prev
                  </button>

                  <div className="flex flex-wrap items-center justify-center gap-2 sm:gap-3">
                    <span className="text-xs font-medium uppercase tracking-wider text-zinc-500">
                      Left page
                    </span>
                    <input
                      type="number"
                      min={1}
                      max={numPages != null ? Math.max(1, numPages - 1) : 1}
                      value={pageInput}
                      onChange={(e) => setPageInput(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          commitPageInput();
                        }
                      }}
                      onBlur={() => commitPageInput()}
                      className="w-16 rounded-lg border border-zinc-200 bg-white px-2 py-1.5 text-center text-sm font-semibold tabular-nums text-zinc-900 shadow-inner outline-none ring-emerald-500/40 focus:border-emerald-400 focus:ring-2"
                      aria-label="Go to left page of spread"
                    />
                    <span className="hidden text-xs text-zinc-400 sm:inline">
                      of spread ·{" "}
                      <span className="tabular-nums text-zinc-600">
                        {leftPage}–{Math.min(leftPage + 1, numPages ?? 1)}
                      </span>
                    </span>
                  </div>

                  <button
                    type="button"
                    disabled={!canNext}
                    onClick={() => {
                      if (numPages == null) {
                        return;
                      }
                      setLeftPage((p) =>
                        Math.min(Math.max(1, numPages - 1), p + 1)
                      );
                    }}
                    className="inline-flex min-h-[2.5rem] items-center gap-2 rounded-xl border border-zinc-200 bg-zinc-50 px-4 py-2 text-sm font-medium text-zinc-800 shadow-sm transition hover:bg-white hover:shadow disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Next
                    <span aria-hidden className="text-zinc-500">
                      ›
                    </span>
                  </button>
                </div>
              </div>
            </>
          ) : null}
        </Document>
      ) : !loadError ? (
        <div className="min-h-[280px] animate-pulse rounded-xl bg-zinc-100" />
      ) : null}
    </div>
  );
}
