"""Local extraction service: Mistral OCR (structured annotation) + PyMuPDF (bboxes).

Returns each metadata field together with a list of {page, bbox} for the source
text so the frontend can draw boxes directly without doing fuzzy matching.
"""
import base64
import json
import os
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

load_dotenv(Path(__file__).resolve().parents[1] / ".env")
EXTRACTION_API = os.environ.get(
    "EXTRACTION_API", "http://localhost:9000/api/v1/metadata-extraction-svc"
)
SSE_TIMEOUT_SECONDS = float(os.environ.get("SSE_TIMEOUT_SECONDS", "180"))

app = FastAPI(title="local-extraction-svc")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "extraction_api": EXTRACTION_API}


@app.post("/match")
async def match(pdf: UploadFile = File(...), text: str = Form(...)) -> dict:
    """Match arbitrary text against the PDF and return per-line bboxes.
    Used by manual user selections to get server-quality matching."""
    pdf_bytes = await pdf.read()
    if not pdf_bytes:
        raise HTTPException(400, "Empty file")
    if not text or not text.strip():
        return {"matches": []}
    pages = extract_pages(pdf_bytes)
    return {"matches": find_field_matches(pages, text)}


@app.post("/extract")
async def extract(pdf: UploadFile = File(...)) -> dict:
    pdf_bytes = await pdf.read()
    if not pdf_bytes:
        raise HTTPException(400, "Empty file")

    metadata = call_metadata_svc(pdf_bytes, pdf.filename or "manuscript.pdf")
    pages = extract_pages(pdf_bytes)

    title = (metadata.get("title") or "").strip()
    abstract = (metadata.get("abstract") or "").strip()
    keywords = [k.strip() for k in (metadata.get("keywords") or []) if k and k.strip()]

    return {
        "title": title,
        "title_matches": find_field_matches(pages, title) if title else [],
        "abstract": abstract,
        "abstract_matches": find_field_matches(pages, abstract) if abstract else [],
        "keywords": keywords,
        "keywords_matches": find_keyword_cluster_matches(pages, keywords),
        "pages": [{"page": p["page"], "width": p["width"], "height": p["height"]} for p in pages],
    }


# --- metadata-extraction-svc client (async upload + SSE result) -------------


def call_metadata_svc(pdf_bytes: bytes, filename: str) -> dict[str, Any]:
    upload = requests.post(
        f"{EXTRACTION_API}/extract-metadata-via-llm",
        files={"manuscript": (filename, pdf_bytes, "application/pdf")},
        timeout=60,
    )
    if not upload.ok:
        raise HTTPException(502, f"Upload failed {upload.status_code}: {upload.text[:300]}")
    document_id = upload.json().get("document_id")
    if not document_id:
        raise HTTPException(502, "Upload response missing document_id")

    sse_url = f"{EXTRACTION_API}/extract-metadata-via-llm/{document_id}/result"
    with requests.get(sse_url, stream=True, timeout=SSE_TIMEOUT_SECONDS) as r:
        r.raise_for_status()
        event_name: str | None = None
        for raw in r.iter_lines(decode_unicode=True):
            if raw is None or raw == "":
                continue
            if raw.startswith("event:"):
                event_name = raw.split(":", 1)[1].strip()
            elif raw.startswith("data:"):
                data = raw.split(":", 1)[1].strip()
                if event_name == "result":
                    try:
                        return json.loads(data)
                    except json.JSONDecodeError:
                        raise HTTPException(502, f"Result not JSON: {data[:300]}")
                if event_name == "error":
                    raise HTTPException(502, f"Extract error: {data[:500]}")
                # 'processing' events ignored — stream continues
    raise HTTPException(504, "Stream ended without result")


# --- PyMuPDF span extraction -------------------------------------------------


def extract_pages(pdf_bytes: bytes) -> list[dict]:
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages: list[dict] = []
    for i, page in enumerate(doc):
        spans: list[dict] = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    text = span.get("text", "")
                    if not text or not text.strip():
                        continue
                    spans.append({
                        "text": text,
                        "bbox": list(span["bbox"]),
                        "size": span.get("size", 0.0),
                    })
        pages.append({
            "page": i,
            "width": float(page.rect.width),
            "height": float(page.rect.height),
            "spans": spans,
        })
    doc.close()
    return pages


# --- Matching ----------------------------------------------------------------


def normalize_with_map(s: str) -> tuple[str, list[int]]:
    """Lowercase, collapse whitespace, drop line-end hyphenation, expand ligatures.

    Returns (normalized_string, mapping) where mapping[i] is the original char
    index that contributed normalized_string[i].
    """
    LIGS = {"ﬁ": "fi", "ﬂ": "fl", "ﬀ": "ff", "ﬃ": "ffi", "ﬄ": "ffl"}
    norm: list[str] = []
    mapping: list[int] = []
    prev_ws = True
    i = 0
    while i < len(s):
        c = s[i]
        # Soft hyphenation: "x-\ny" or "x- y" between letters → "xy"
        if c in ("-", "­"):
            j = i + 1
            while j < len(s) and s[j].isspace():
                j += 1
            if j < len(s) and s[j].isalpha() and norm and norm[-1].isalpha():
                i = j
                prev_ws = False
                continue
        if c in LIGS:
            for ch in LIGS[c]:
                norm.append(ch.lower())
                mapping.append(i)
            prev_ws = False
            i += 1
            continue
        if c.isspace():
            if not prev_ws:
                norm.append(" ")
                mapping.append(i)
                prev_ws = True
            i += 1
        else:
            norm.append(c.lower())
            mapping.append(i)
            prev_ws = False
            i += 1
    return "".join(norm), mapping


def build_concat(spans: list[dict]) -> tuple[str, list[int]]:
    """Concat span texts with single-space separators; return raw concat + char→span_idx."""
    parts: list[str] = []
    char_to_span: list[int] = []
    for sidx, span in enumerate(spans):
        text = span["text"]
        parts.append(text)
        char_to_span.extend([sidx] * len(text))
        parts.append(" ")
        char_to_span.append(-1)
    return "".join(parts), char_to_span


def spans_in_range(
    spans: list[dict],
    char_to_span: list[int],
    mapping: list[int],
    start_norm: int,
    end_norm: int,
) -> list[dict]:
    seen: list[dict] = []
    seen_idx: set[int] = set()
    a = mapping[start_norm]
    b = mapping[end_norm]
    for i in range(a, b + 1):
        if i >= len(char_to_span):
            break
        sidx = char_to_span[i]
        if sidx >= 0 and sidx not in seen_idx:
            seen_idx.add(sidx)
            seen.append(spans[sidx])
    return seen


def group_into_lines(spans: list[dict], tol: float = 2.0) -> list[list[dict]]:
    groups: list[list[dict]] = []
    for span in spans:
        y0 = span["bbox"][1]
        for g in groups:
            if abs(g[0]["bbox"][1] - y0) < tol:
                g.append(span)
                break
        else:
            groups.append([span])
    return groups


def line_groups_to_bboxes(line_groups: list[list[dict]], page: dict) -> list[dict]:
    out: list[dict] = []
    for grp in line_groups:
        if not grp:
            continue
        # Sort spans within a line by x for natural reading order
        ordered = sorted(grp, key=lambda s: s["bbox"][0])
        x0 = min(s["bbox"][0] for s in ordered)
        y0 = min(s["bbox"][1] for s in ordered)
        x1 = max(s["bbox"][2] for s in ordered)
        y1 = max(s["bbox"][3] for s in ordered)
        text = " ".join(s["text"] for s in ordered).strip()
        out.append({"page": page["page"], "bbox": [x0, y0, x1, y1], "text": text})
    return out


def find_field_matches(pages: list[dict], target: str) -> list[dict]:
    """Find target text in the document; return bboxes for the matched spans
    on the first page where it's found (one bbox per visual line)."""
    target_norm, _ = normalize_with_map(target)
    if len(target_norm.strip()) < 4:
        return []
    for page in pages:
        if not page["spans"]:
            continue
        concat, char_to_span = build_concat(page["spans"])
        concat_norm, mapping = normalize_with_map(concat)
        idx = concat_norm.find(target_norm)
        if idx < 0:
            # Fuzzy: try matching on prefix/suffix anchors
            idx, length = fuzzy_locate(concat_norm, target_norm)
            if idx < 0:
                continue
            end_idx = idx + length - 1
        else:
            end_idx = idx + len(target_norm) - 1
        matched = spans_in_range(page["spans"], char_to_span, mapping, idx, end_idx)
        line_groups = group_into_lines(matched)
        return line_groups_to_bboxes(line_groups, page)
    return []


def fuzzy_locate(hay: str, needle: str) -> tuple[int, int]:
    """Return (start, length) of best match of needle in hay using start+end anchors."""
    ANCHOR = 20
    if len(needle) < ANCHOR:
        return -1, 0
    start_anchor = needle[:ANCHOR]
    start = hay.find(start_anchor)
    if start < 0:
        return -1, 0
    if len(needle) < ANCHOR * 2 + 10:
        return start, min(len(hay) - start, len(needle))
    end_anchor = needle[-ANCHOR:]
    expected = start + len(needle)
    tol = min(400, max(60, int(len(needle) * 0.3)))
    lo = max(start + ANCHOR, expected - tol)
    hi = min(len(hay), expected + tol)
    end_idx = hay.rfind(end_anchor, lo, hi)
    if end_idx < 0:
        return start, min(len(hay) - start, len(needle))
    return start, end_idx + ANCHOR - start


def find_keyword_cluster_matches(pages: list[dict], keywords: list[str]) -> list[dict]:
    """Pick the page+window where the most distinct keywords cluster tightly,
    return bboxes for each keyword occurrence within that window."""
    if not keywords:
        return []
    best: tuple[int, int, dict, list, list, list, int, int, list] | None = None
    for page in pages:
        if not page["spans"]:
            continue
        concat, char_to_span = build_concat(page["spans"])
        concat_norm, mapping = normalize_with_map(concat)
        occs: list[tuple[str, int, int]] = []
        for kw in keywords:
            kw_norm, _ = normalize_with_map(kw)
            if len(kw_norm) < 3:
                continue
            start = 0
            while True:
                pos = concat_norm.find(kw_norm, start)
                if pos < 0:
                    break
                occs.append((kw_norm, pos, pos + len(kw_norm)))
                start = pos + 1
        if not occs:
            continue
        occs.sort(key=lambda o: o[1])

        WINDOW = 600
        page_best = (0, 10**9, 0, 0)  # (count, size, lo, hi)
        for i in range(len(occs)):
            seen: set[str] = set()
            lo_i = occs[i][1]
            for j in range(i, len(occs)):
                hi_j = occs[j][2]
                size = hi_j - lo_i
                if size > WINDOW:
                    break
                seen.add(occs[j][0])
                if len(seen) > page_best[0] or (len(seen) == page_best[0] and size < page_best[1]):
                    page_best = (len(seen), size, lo_i, hi_j)
        count, size, lo, hi = page_best
        if count < 2:
            continue
        if best is None or count > best[0] or (count == best[0] and size < best[1]):
            best = (count, size, page, char_to_span, mapping, page["spans"], lo, hi, occs)
    if best is None:
        return []
    _, _, page, char_to_span, mapping, spans, lo, hi, occs = best
    seen_idx: set[int] = set()
    selected_spans: list[dict] = []
    for kw, s, e in occs:
        if s < lo or e > hi:
            continue
        a = mapping[s]
        b = mapping[e - 1]
        for i in range(a, b + 1):
            if i >= len(char_to_span):
                break
            sidx = char_to_span[i]
            if sidx >= 0 and sidx not in seen_idx:
                seen_idx.add(sidx)
                selected_spans.append(spans[sidx])
    line_groups = group_into_lines(selected_spans)
    return line_groups_to_bboxes(line_groups, page)
