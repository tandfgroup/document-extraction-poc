import subprocess
import sys
from pathlib import Path

import markdown as md_lib

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
</head>
<body>
{body}
</body>
</html>
"""

IMAGE_TAG = "docling-poc:latest"
DOCKERFILE = "Dockerfile.docling"


def ensure_image():
    inspect = subprocess.run(
        ["docker", "image", "inspect", IMAGE_TAG],
        capture_output=True,
    )
    if inspect.returncode == 0:
        return
    print(f"Building {IMAGE_TAG} (first run only, downloads ~1GB)...")
    subprocess.run(
        ["docker", "build", "-t", IMAGE_TAG, "-f", DOCKERFILE, "."],
        check=True,
    )


def main():
    if len(sys.argv) < 2:
        print("Usage: python docling_ocr.py <path-to-document>")
        sys.exit(1)

    src = Path(sys.argv[1]).resolve()
    if not src.exists():
        print(f"File not found: {src}")
        sys.exit(1)

    out_dir = Path("outputs").resolve()
    out_dir.mkdir(exist_ok=True)
    base = src.stem

    ensure_image()

    print(f"Running docling on {src.name}...")
    subprocess.run([
        "docker", "run", "--rm",
        "-v", f"{src.parent}:/in:ro",
        "-v", f"{out_dir}:/out",
        "-v", "docling-rapidocr-models:/usr/local/lib/python3.11/site-packages/rapidocr/models",
        "-v", "docling-hf-cache:/root/.cache/huggingface",
        IMAGE_TAG,
        f"/in/{src.name}",
        "--to", "md",
        "--image-export-mode", "referenced",
        "--output", "/out",
    ], check=True)

    raw_md = out_dir / f"{base}.md"
    if not raw_md.exists():
        print(f"Expected output not found: {raw_md}")
        sys.exit(1)

    docling_md = out_dir / f"{base}_docling.md"
    raw_md.rename(docling_md)

    body = md_lib.markdown(
        docling_md.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "smarty", "pymdownx.magiclink"],
    )
    html_path = out_dir / f"{base}_docling.html"
    html_path.write_text(HTML_TEMPLATE.format(title=base, body=body), encoding="utf-8")

    print(f"Markdown saved to: {docling_md}")
    print(f"HTML saved to:     {html_path}")


if __name__ == "__main__":
    main()
