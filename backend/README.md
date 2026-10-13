# IntelliAssist AI — Document Processing Module

## 1. Overview & Purpose
The `document_processor` module provides core document ingestion for **IntelliAssist AI**. It extracts text and structural metadata from user-supplied files, returning a normalized, predictable dictionary format suitable for downstream search and AI workflows.

---

## 2. Supported Formats
- **PDF (`.pdf`)**: Extracts text on a page-by-page basis, preserving original physical page order and 1-based page numbering.
- **TXT (`.txt`)**: Reads UTF-8 (including UTF-8 with Byte Order Mark `BOM`) plain text files, stripped of extraneous leading/trailing whitespace, represented as a single logical page (`page_number: 1`).
- **DOCX (`.docx`)**: Extracts non-empty paragraphs and tabular data in natural document reading order, represented as a single logical page (`page_number: 1`).

All format extensions are checked case-insensitively (`.PDF`, `.Txt`, `.DOCX` are supported).

---

## 3. System Requirements
- **Python**: Python 3.10 or higher (tested on Python 3.12.4).
- **Operating System**: Cross-platform (Windows, macOS, Linux).

---

## 4. Setup & Installation (Windows)

### Step 1: Navigate to the Backend Directory
```powershell
cd "d:\IntelliAssist AI\backend"
```

### Step 2: Create a Virtual Environment
```powershell
python -m venv .venv
```

### Step 3: Activate the Virtual Environment
On Windows PowerShell:
```powershell
.\.venv\Scripts\Activate.ps1
```
*(If script execution is disabled in PowerShell, you can run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` or invoke Python directly via `.\.venv\Scripts\python.exe`)*

On Windows Command Prompt (`cmd.exe`):
```cmd
.\.venv\Scripts\activate.bat
```

### Step 4: Install Dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 5. Running Automated Tests

Run the test suite using `pytest` from the `backend/` directory:

```powershell
python -m pytest -v
```

Or with detailed test report and summary:
```powershell
python -m pytest tests/test_document_processor.py -v
```

---

## 6. Usage Example

```python
from pathlib import Path
from app.document_processor import process_document

# Example: Process a PDF file
pdf_path = Path("sample_contract.pdf")
try:
    result = process_document(pdf_path)
    print(f"Processed: {result['filename']}")
    print(f"Format: {result['file_type']}")
    print(f"Total Pages: {result['total_pages']}")
    print(f"Total Characters: {result['total_characters']}")

    for page in result["pages"]:
        print(f"\n--- Page {page['page_number']} ---")
        print(page["text"][:200])  # preview first 200 chars
except FileNotFoundError:
    print("Error: Document file not found.")
except ValueError as e:
    print(f"Document processing error: {e}")
```

---

## 7. Return Schema

`process_document(file_path)` returns a structured dictionary:

```json
{
  "filename": "sample_contract.pdf",
  "file_type": "pdf",
  "pages": [
    {
      "page_number": 1,
      "text": "IntelliAssist AI Contract Agreement..."
    },
    {
      "page_number": 2,
      "text": "Terms and conditions continued..."
    }
  ],
  "total_pages": 2,
  "total_characters": 4500
}
```

### Field Descriptions:
- **`filename`** (`str`): Base name of the file including extension.
- **`file_type`** (`str`): Lowercase file extension without dot (`"pdf"`, `"txt"`, or `"docx"`).
- **`pages`** (`list[dict]`): Ordered list of page objects, each containing:
  - **`page_number`** (`int`): 1-based index of the page.
  - **`text`** (`str`): Extracted text content for that page.
- **`total_pages`** (`int`): Total count of pages processed (`len(pages)`).
- **`total_characters`** (`int`): Total character count across all extracted pages.

---

## 8. Error Handling & Validation
- **Missing File**: Raises `FileNotFoundError`.
- **Directory / Non-Regular File**: Raises `ValueError`.
- **Unsupported File Extension**: Raises `ValueError` detailing allowed formats (`.pdf`, `.txt`, `.docx`).
- **Empty / Whitespace-Only Text Document**: Raises `ValueError`.
- **Corrupted or Malformed PDF/DOCX**: Raises `ValueError` with parser diagnostic details.
- **Scanned / Image-Only PDF**: Detects the absence of extractable text and raises `ValueError` notifying that OCR is required.

---

## 9. Known Limitations
1. **OCR is Not Implemented**: Image-only scanned PDFs containing no text layers produce no extractable text and are safely rejected with an explicit OCR message.
2. **Logical Pagination for TXT & DOCX**: DOCX and TXT files do not contain fixed physical pagination in standard file structures; they are faithfully represented as a single logical page (`page_number: 1`).
3. **Scope Boundary**: This module extracts readable document text and structural metadata only. It does not perform text chunking, embeddings generation, vector database indexing, LLM retrieval, or question-answering.
