import base64
import json
import os
import re
import sys
import requests
import markdown as md_lib
from dotenv import load_dotenv

load_dotenv()

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
       max-width: 900px; margin: 2rem auto; padding: 0 1rem; line-height: 1.6;
       color: #222; }}
img {{ max-width: 100%; height: auto; display: block; margin: 1rem auto; }}
table {{ border-collapse: collapse; margin: 1rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 0.4rem 0.7rem; }}
th {{ background: #f4f4f4; }}
hr {{ border: none; border-top: 2px dashed #ccc; margin: 2rem 0; }}
pre, code {{ background: #f6f8fa; padding: 0.1rem 0.3rem; border-radius: 4px; }}
h1, h2, h3 {{ border-bottom: 1px solid #eee; padding-bottom: 0.2rem; }}
</style>
<script>
window.MathJax = {{
  tex: {{ inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
          displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']] }}
}};
</script>
<script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" async></script>
</head>
<body>
{body}
</body>
</html>
"""

AZURE_API_KEY = os.environ.get("AZURE_API_KEY", "")
ENDPOINT = "https://validations-poc-resource.services.ai.azure.com/providers/mistral/azure/ocr"
MODEL = "mistral-document-ai-2512"

MIME_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
}


def file_to_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def get_mime_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext not in MIME_TYPES:
        raise ValueError(
            f"Unsupported file type: {ext}. Supported: {', '.join(MIME_TYPES)}"
        )
    return MIME_TYPES[ext]


BBOX_ANNOTATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "properties": {
                "document_type": {"type": "string", "description": "The type of the image."},
                "short_description": {"type": "string", "description": "A description in English describing the image."},
                "summary": {"type": "string", "description": "Summarize the image."},
            },
            "required": ["document_type", "short_description", "summary"],
            "title": "BBOXAnnotation",
            "type": "object",
            "additionalProperties": False,
        },
        "name": "bbox_annotation",
        "strict": True,
    },
}

DOCUMENT_ANNOTATION_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "schema": {
            "properties": {
                "funding": {
                    "type": "string",
                    "description": "The funding statement of the paper, including grant numbers and funding bodies. Return 'Not specified' if not present.",
                },
                "conflict_of_interest": {
                    "type": "string",
                    "description": "The conflict of interest / competing interests declaration. Return 'Not specified' if not present.",
                },
                "ethics_approval": {
                    "type": "string",
                    "description": "Ethics approval, IRB approval, or informed consent statement. Return 'Not specified' if not present.",
                },
                "data_availability": {
                    "type": "string",
                    "description": "Data availability statement, including any repository links or accession codes. Return 'Not specified' if not present.",
                },
                "acknowledgments": {
                    "type": "string",
                    "description": "The acknowledgments section. Return 'Not specified' if not present.",
                },
            },
            "required": ["funding", "conflict_of_interest", "ethics_approval", "data_availability", "acknowledgments"],
            "title": "DocumentAnnotation",
            "type": "object",
            "additionalProperties": False,
        },
        "name": "document_annotation",
        "strict": True,
    },
}


def run_ocr(file_path: str, annotate_images: bool = False) -> dict:
    if not AZURE_API_KEY:
        raise ValueError("AZURE_API_KEY environment variable is not set")

    mime = get_mime_type(file_path)
    b64 = file_to_base64(file_path)

    payload = {
        "model": MODEL,
        "document": {
            "type": "document_url",
            "document_url": f"data:{mime};base64,{b64}",
        },
        "document_annotation_format": DOCUMENT_ANNOTATION_SCHEMA,
        "include_image_base64": True,
    }
    if annotate_images:
        payload["bbox_annotation_format"] = BBOX_ANNOTATION_SCHEMA

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AZURE_API_KEY}",
    }

    suffix = " (with image annotations)" if annotate_images else ""
    print(f"Sending {file_path} to Mistral OCR{suffix}...")
    response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=240)
    response.raise_for_status()
    return response.json()


def _save_image(img: dict, images_dir: str) -> None:
    data = img.get("image_base64", "")
    if "," in data:
        data = data.split(",", 1)[1]
    os.makedirs(images_dir, exist_ok=True)
    with open(os.path.join(images_dir, img["id"]), "wb") as f:
        f.write(base64.b64decode(data))


def _annotation_caption(annotation) -> str:
    if not annotation:
        return ""
    try:
        parsed = json.loads(annotation) if isinstance(annotation, str) else annotation
        lines = [f"**{k.replace('_', ' ').title()}:** {v}" for k, v in parsed.items()]
        return "\n\n> " + "\n> ".join(lines) + "\n"
    except (json.JSONDecodeError, AttributeError):
        return ""


def _process_page(page: dict, images_dir: str, images_dirname: str, with_captions: bool) -> str:
    md = page.get("markdown", "")
    for img in page.get("images", []):
        _save_image(img, images_dir)
        img_id = img["id"]
        new_ref = f"]({images_dirname}/{img_id})"
        md = md.replace(f"]({img_id})", new_ref)
        if with_captions:
            caption = _annotation_caption(img.get("image_annotation"))
            if caption:
                md = md.replace(new_ref, new_ref + caption, 1)
    md += _hyperlinks_block(page)
    return md


def _hyperlinks_block(page: dict) -> str:
    links = page.get("hyperlinks") or []
    seen = []
    for url in links:
        if isinstance(url, str) and url not in seen and url not in page.get("markdown", ""):
            seen.append(url)
    if not seen:
        return ""
    items = "\n".join(f"- <{url}>" for url in seen)
    return f"\n\n**Links on this page:**\n\n{items}\n"


def _highlights_block(document_annotation) -> str:
    if not document_annotation:
        return ""
    try:
        parsed = json.loads(document_annotation) if isinstance(document_annotation, str) else document_annotation
    except json.JSONDecodeError:
        return ""

    lines = ["# Highlights\n"]
    for key, value in parsed.items():
        title = key.replace("_", " ").title()
        lines.append(f"## {title}\n\n{value}\n")
    lines.append("\n---\n")
    return "\n".join(lines)


_CITATION_CONTENT_RE = re.compile(r'^[\d,\s\-–—]+$')
_BARE_SUPSUB_RE = re.compile(r'(?<!\$)([\^_])\{([^}]+)\}(?!\$)')
_WRAPPED_SUPSUB_RE = re.compile(r'\$([\^_])\{([^}]+)\}\$')


def _supsub_replace(match: "re.Match") -> str:
    sym, content = match.group(1), match.group(2)
    tag = "sup" if sym == "^" else "sub"
    if _CITATION_CONTENT_RE.match(content):
        return f"<{tag}>{content.strip()}</{tag}>"
    return f"${sym}{{{content}}}$"


_DASH_RANGE_RE = re.compile(r'(\d)--\s+(\d)')


def _normalize_supsub(text: str) -> str:
    text = _DASH_RANGE_RE.sub(r'\1--\2', text)
    text = _WRAPPED_SUPSUB_RE.sub(_supsub_replace, text)
    text = _BARE_SUPSUB_RE.sub(_supsub_replace, text)
    return text


def _write_outputs(pages: list, output_dir: str, base: str, images_dirname: str,
                   with_captions: bool, document_annotation=None):
    images_dir = os.path.join(output_dir, images_dirname)
    page_markdowns = [_process_page(p, images_dir, images_dirname, with_captions) for p in pages]
    full_markdown = _highlights_block(document_annotation) + "\n\n---\n\n".join(page_markdowns)

    markdown_path = os.path.join(output_dir, f"{base}.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    html_path = os.path.join(output_dir, f"{base}.html")
    body = md_lib.markdown(
        _normalize_supsub(full_markdown),
        extensions=["tables", "fenced_code", "smarty", "pymdownx.arithmatex", "pymdownx.magiclink"],
        extension_configs={"pymdownx.arithmatex": {"generic": True}},
    )
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(HTML_TEMPLATE.format(title=base, body=body))

    return markdown_path, html_path


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}

    if not args:
        print("Usage: python ocr.py <path-to-document> [--annotate]")
        print("Supported formats: .pdf, .docx, .pptx")
        print("Example: python ocr.py sample.pdf --annotate")
        sys.exit(1)

    file_path = args[0]
    if not os.path.exists(file_path):
        print(f"File not found: {file_path}")
        sys.exit(1)

    annotate = "--annotate" in flags
    result = run_ocr(file_path, annotate_images=annotate)

    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(file_path))[0]
    images_dirname = f"{base}_images"
    pages = result.get("pages", [])
    doc_annotation = result.get("document_annotation")

    output_path = os.path.join(output_dir, f"{base}_ocr_result.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    md_clean, html_clean = _write_outputs(pages, output_dir, base, images_dirname,
                                          with_captions=False, document_annotation=doc_annotation)
    print(f"\nResult saved to: {output_path}")
    print(f"Markdown (clean) saved to: {md_clean}")
    print(f"HTML (clean) saved to: {html_clean}")

    if annotate:
        md_ann, html_ann = _write_outputs(pages, output_dir, f"{base}_annotated", images_dirname,
                                          with_captions=True, document_annotation=doc_annotation)
        print(f"Markdown (annotated) saved to: {md_ann}")
        print(f"HTML (annotated) saved to: {html_ann}")
    print("\n--- Preview (first 2000 chars) ---")
    print(json.dumps(result, indent=2)[:2000])


if __name__ == "__main__":
    main()
