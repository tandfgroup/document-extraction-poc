# document-extraction-poc

CLI that sends a document to the Azure-hosted Mistral Document AI OCR endpoint
(`mistral-document-ai-2512`) and saves the result as JSON, Markdown, and HTML.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
AZURE_API_KEY=your-key-here
```

## Usage

```bash
python3 ocr.py <path-to-document>
```

Supported formats: `.pdf`, `.docx`, `.pptx`.

Add `--annotate` to also generate per-image captions (document type, short
description, summary):

```bash
python3 ocr.py /path/to/paper.pdf --annotate
```

## Output

Files land in `outputs/` (created automatically):

| File                          | Purpose                                                      |
| ----------------------------- | ------------------------------------------------------------ |
| `<name>_ocr_result.json`      | Raw API response                                             |
| `<name>.md` / `<name>.html`   | Clean rendering — page text, images, and a Highlights block  |
| `<name>_annotated.md` / `.html` | Same plus AI captions under each image (only with `--annotate`) |
| `<name>_images/`              | Extracted images referenced by the markdown/HTML             |

The **Highlights** block at the top of every render contains structured
extracts of: funding, conflict of interest, ethics approval, data
availability, and acknowledgments. To change those sections, edit
`DOCUMENT_ANNOTATION_SCHEMA` in [ocr.py](ocr.py).

Open the HTML in a browser:

```bash
open outputs/<name>.html
```

Or preview the markdown in VS Code with `Cmd+Shift+V`.
