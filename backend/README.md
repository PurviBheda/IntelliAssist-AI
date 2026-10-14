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
3. **Scope Boundary**: Optical Character Recognition (OCR) for image-only scans is not yet implemented. Text chunking and local embeddings are implemented (see Section 10 below); vector indexing with FAISS, retrieval, and LLM chatbot endpoints belong to subsequent milestone days.

---

## 10. Text Chunking and Embeddings

### 10.1 Why Documents are Divided into Chunks
Large language models and embedding models operate on bounded input contexts (for example, 256 or 512 wordpieces in `all-MiniLM-L6-v2`). When multi-page contracts, manuals, or research papers are embedded as a single monolithic block of text, specific factual details, figures, and nuanced statements are averaged out into a fuzzy vector. 

By dividing documents into smaller, coherent chunks (approx. 800 characters by default), each vector captures a focused topic or concept. This allows downstream search to retrieve precisely the relevant paragraphs rather than entire documents.

### 10.2 What Chunk Overlap Means
When text is split strictly by character count, critical sentences or ideas can be cut in half right across the boundary. Chunk overlap repeats a small portion (~150 characters by default) from the tail of the preceding chunk at the beginning of the succeeding chunk.

This sliding window overlap ensures:
- Key phrases and contextual relationships spanning chunk boundaries remain intact in at least one chunk.
- No concepts or words are lost between chunks.
- Natural boundary awareness (paragraphs `\n\n`, lines `\n`, sentence punctuation `. `, `! `, `? `, and spaces) is prioritized before falling back to hard cuts.

### 10.3 Why Page Numbers and Filenames are Preserved
Every chunk produced by `chunk_document()` carries metadata:
- **`filename`**: Source file name (e.g., `contract.pdf`).
- **`page_number`**: The original physical page index (1-based).
- **`chunk_index`**: The sequence index of the chunk within that page (1-based).
- **`chunk_id`**: A deterministic, unique identifier formatted as `{filename}_page_{page_number}_chunk_{chunk_index}`.

Preserving page numbers and filenames enables **verifiable source attribution and exact citations**. When an AI assistant answers user questions, it can link directly back to the exact page of the original document. Furthermore, chunks never cross page boundaries, keeping citation integrity pristine.

### 10.4 What Embeddings Are
An **embedding** is a dense numerical vector (an array of 384 floating-point numbers) that represents the semantic meaning of a text string in high-dimensional vector space.

Unlike raw text strings or keyword bags, embedding vectors place texts with similar conceptual meanings near one another. For instance, the sentences *"The engine requires synthetic oil"* and *"Automobile motors need high-grade lubricant"* will have vectors with high cosine similarity (close to 1.0), even though they share almost no identical vocabulary.

### 10.5 Why all-MiniLM-L6-v2 is Used
`sentence-transformers/all-MiniLM-L6-v2` was selected because:
1. **Lightweight & Fast**: Produces compact 384-dimensional dense vectors with low CPU latency and minimal RAM usage.
2. **100% Local Execution**: Runs entirely on the local machine without requiring paid API keys, internet connectivity during inference, or third-party cloud services.
3. **High Quality**: Consistently ranks as one of the best speed-to-performance embedding models for retrieval-augmented generation (RAG) benchmarks.

### 10.6 Why Embeddings Help Semantic Search
Traditional keyword search (e.g. lexical matching or grep) only matches literal words. If a user searches for *"vacation policy"*, keyword search will miss a document section titled *"Annual Paid Leave Guidelines"*. 

Semantic search with embeddings solves this: queries and document chunks are mapped into the same semantic space, allowing the system to retrieve passages by **conceptual intent** rather than strict keyword identity.

### 10.7 Why FAISS is the Next Step (Not Implemented Today)
While embeddings convert chunks into numerical vectors, calculating similarity scores across thousands of vectors using linear scans becomes slow as documents accumulate.

**FAISS (Facebook AI Similarity Search)** is a specialized library optimized for lightning-fast similarity search and clustering of dense vectors. FAISS is scheduled for implementation in Day 6. Today's scope focuses strictly on reliable chunking, lazy model initialization, vector dimension consistency, and associating chunk metadata with vectors.

### 10.8 Local Operation & Zero API Key Requirements
The embedding service is built strictly on local open-source components (`sentence-transformers` and `PyTorch` CPU). It does **not** make calls to OpenAI, Cohere, Google, or any paid API endpoint. No API keys or external billing are required.

> [!NOTE]
> **First-Time Model Download**: The first time `EmbeddingService.embed_text()` or `smoke_test_embedding.py` is invoked, the underlying library downloads the model weights (~90MB) from the Hugging Face Hub into your local cache (`~/.cache/huggingface/hub`). Subsequent runs load instantly and offline from this local cache.

### 10.9 Dependency Notes
In `backend/requirements.txt`:
- `sentence-transformers>=3.0.0`: Modern sentence embedding pipeline with optimized CPU encoding.
- `numpy>=1.26.0`: Fast vector manipulation and array conversions for downstream FAISS indexing.

### 10.10 Python Usage Example

```python
from pathlib import Path
from app.text_chunker import chunk_document
from app.embedding_service import EmbeddingService, embed_chunks

# 1. Sample processed document (from process_document)
processed_doc = {
    "filename": "security_whitepaper.pdf",
    "file_type": "pdf",
    "pages": [
        {
            "page_number": 1,
            "text": (
                "IntelliAssist AI utilizes local embedding architectures to "
                "ensure enterprise confidentiality. All documents are processed "
                "on-premises without routing data to third-party endpoints."
            ),
        },
        {
            "page_number": 2,
            "text": (
                "Role-based access controls and deterministic chunk tracking "
                "ensure reproducible citations for every generated answer."
            ),
        },
    ],
    "total_pages": 2,
    "total_characters": 315,
}

# 2. Divide document into overlapping chunks (character-based)
chunks = chunk_document(processed_doc, chunk_size=800, chunk_overlap=150)
print(f"Generated {len(chunks)} chunks.")
print(f"First Chunk ID: {chunks[0]['chunk_id']}")
print(f"First Chunk Metadata: {chunks[0]['metadata']}")

# 3. Generate embeddings for all chunks (lazy-loads model on first call)
service = EmbeddingService()
embedded_chunks = service.embed_chunks(chunks)

for item in embedded_chunks:
    print(f"\nChunk: {item['chunk_id']}")
    print(f"  Vector Shape: {item['embedding'].shape}")  # (384,)
    print(f"  Dtype: {item['embedding'].dtype}")         # float32
    print(f"  Source Page: {item['metadata']['page_number']}")
```

---

## 11. Running Automated Tests

Run the full automated test suite (including document processor, text chunker, and embedding service):

```powershell
python -m pytest tests/ -v
```

Run test suites individually:
```powershell
# Document processor extraction tests (19 tests)
python -m pytest tests/test_document_processor.py -v

# Text chunking and boundary split tests (16 tests)
python -m pytest tests/test_text_chunker.py -v

# Text embedding service tests with mock models (12 tests)
python -m pytest tests/test_embedding_service.py -v
```

> [!TIP]
> The automated unit tests use deterministic mock models. They run in under 3 seconds and require **no internet connection** and **no model downloads**.

---

## 12. Running the Real Embedding Smoke Test (Manual)

To verify the real `sentence-transformers/all-MiniLM-L6-v2` model end-to-end on your local system:

```powershell
python smoke_test_embedding.py
```

This manual test will:
1. Divide a multi-page test document into overlapping chunks.
2. Lazy-load the real `all-MiniLM-L6-v2` model weights.
3. Generate real 384-dimensional dense vectors.
4. Verify L2 unit normalization (~1.0).
5. Compute and print the cosine similarity between chunks.

