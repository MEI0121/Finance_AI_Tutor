"use client";

import { useEffect, useRef, useState } from "react";
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
  const [pageMenuOpen, setPageMenuOpen] = useState(false);

  const measureRef = useRef<HTMLDivElement>(null);
  const pageMenuRef = useRef<HTMLDivElement>(null);

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
    const maxLeft = Math.max(1, numPages);
    setLeftPage((prev) => {
      const next = Math.max(1, Math.min(prev, maxLeft));
      return next;
    });
  }, [numPages]);

  useEffect(() => {
    if (!pageMenuOpen) {
      return;
    }
    const onPointerDown = (event: MouseEvent) => {
      const target = event.target as Node | null;
      if (target && !pageMenuRef.current?.contains(target)) {
        setPageMenuOpen(false);
      }
    };
    document.addEventListener("mousedown", onPointerDown);
    return () => document.removeEventListener("mousedown", onPointerDown);
  }, [pageMenuOpen]);

  const clampPage = (page: number, total: number | null): number => {
    if (total == null || total < 1) {
      return 1;
    }
    return Math.max(1, Math.min(page, total));
  };

  const safeLeftPage = clampPage(leftPage, numPages);
  const spreadLeftPage =
    numPages != null && numPages >= 2 && safeLeftPage >= numPages
      ? numPages - 1
      : safeLeftPage;
  const rightPage =
    numPages != null && spreadLeftPage + 1 <= numPages ? spreadLeftPage + 1 : null;

  const canPrev = numPages != null && safeLeftPage > 1;
  const canNext = numPages != null && safeLeftPage < numPages;

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
                        key={`spread-left-${spreadLeftPage}`}
                        pageNumber={spreadLeftPage}
                        width={pageWidthEach}
                        renderTextLayer
                        renderAnnotationLayer
                        className="[&_.react-pdf__Page__textContent]:select-text [&_.react-pdf__Page__canvas]:bg-white"
                      />
                    </div>
                    {rightPage != null ? (
                      <div
                        className="shrink-0 overflow-hidden rounded-r-2xl border-l border-white/80 bg-white shadow-[-4px_0_24px_-8px_rgba(0,0,0,0.18)] ring-1 ring-zinc-900/5"
                        style={{ width: pageWidthEach }}
                      >
                        <Page
                          key={`spread-right-${rightPage}`}
                          pageNumber={rightPage}
                          width={pageWidthEach}
                          renderTextLayer
                          renderAnnotationLayer
                          className="[&_.react-pdf__Page__textContent]:select-text [&_.react-pdf__Page__canvas]:bg-white"
                        />
                      </div>
                    ) : (
                      <div
                        aria-hidden
                        className="shrink-0 rounded-r-2xl border border-dashed border-zinc-300/80 bg-zinc-50/70"
                        style={{ width: pageWidthEach }}
                      />
                    )}
                  </div>
                ) : (
                  <div className="flex justify-center overflow-hidden rounded-2xl bg-white shadow-xl ring-1 ring-zinc-900/5">
                    <Page
                      pageNumber={safeLeftPage}
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
                      Page
                    </span>
                    <div ref={pageMenuRef} className="relative">
                      <button
                        type="button"
                        onClick={() => setPageMenuOpen((open) => !open)}
                        className="inline-flex min-w-[5.5rem] items-center justify-between gap-2 rounded-lg border border-zinc-200 bg-white px-2.5 py-1.5 text-sm font-semibold tabular-nums text-zinc-900 shadow-sm outline-none ring-emerald-500/40 transition hover:bg-zinc-50 focus:border-emerald-400 focus:ring-2"
                        aria-label="Choose page"
                        aria-haspopup="listbox"
                        aria-expanded={pageMenuOpen}
                      >
                        <span>{safeLeftPage}</span>
                        <span className="text-xs text-zinc-500" aria-hidden>
                          ▾
                        </span>
                      </button>
                      {pageMenuOpen ? (
                        <ul
                          role="listbox"
                          aria-label="Page options"
                          className="absolute bottom-full left-0 z-30 mb-2 max-h-40 w-full overflow-y-auto rounded-xl border border-zinc-200 bg-white py-1 shadow-xl ring-1 ring-zinc-900/5"
                        >
                          {Array.from({ length: numPages ?? 1 }, (_, idx) => {
                            const page = idx + 1;
                            const selected = page === safeLeftPage;
                            return (
                              <li key={page}>
                                <button
                                  type="button"
                                  onClick={() => {
                                    setLeftPage(clampPage(page, numPages));
                                    setPageMenuOpen(false);
                                  }}
                                  className={`w-full px-3 py-1.5 text-left text-sm tabular-nums transition ${
                                    selected
                                      ? "bg-emerald-50 font-semibold text-emerald-700"
                                      : "text-zinc-700 hover:bg-zinc-100"
                                  }`}
                                >
                                  {page}
                                </button>
                              </li>
                            );
                          })}
                        </ul>
                      ) : null}
                    </div>
                    <span className="text-xs text-zinc-500">
                      of{" "}
                      <span className="font-semibold tabular-nums text-zinc-700">
                        {numPages ?? 1}
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
                      setLeftPage((p) => clampPage(p + 1, numPages));
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
