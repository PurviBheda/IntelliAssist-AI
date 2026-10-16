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

### 10.7 Vector Indexing with FAISS
While embeddings convert chunks into numerical vectors, calculating similarity scores across thousands of vectors using linear scans becomes slow as documents accumulate.

**FAISS (Facebook AI Similarity Search)** provides an optimized C++ implementation with Python bindings for lightning-fast similarity search and clustering of dense vectors. IntelliAssist AI integrates an in-memory `FAISSVectorStore` based on `faiss.IndexFlatIP` (see Section 13 below).

### 10.8 Local Operation & Zero API Key Requirements
The embedding service is built strictly on local open-source components (`sentence-transformers` and `PyTorch` CPU). It does **not** make calls to OpenAI, Cohere, Google, or any paid API endpoint. No API keys or external billing are required.

> [!NOTE]
> **First-Time Model Download**: The first time `EmbeddingService.embed_text()` or `smoke_test_embedding.py` is invoked, the underlying library downloads the model weights (~90MB) from the Hugging Face Hub into your local cache (`~/.cache/huggingface/hub`). Subsequent runs load instantly and offline from this local cache.

### 10.9 Dependency Notes
In `backend/requirements.txt`:
- `sentence-transformers>=3.0.0`: Modern sentence embedding pipeline with optimized CPU encoding.
- `numpy>=1.26.0`: Fast vector manipulation and array conversions for downstream FAISS indexing.
- `faiss-cpu>=1.8.0`: High-performance similarity search library compiled for local CPU execution.

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

Run the full automated test suite (89 tests covering document processing, chunking, embeddings, vector indexing, and retrieval evaluation):

```powershell
python -m pytest tests/ -v
```

Run test suites individually:
```powershell
# Document processor extraction tests (19 tests)
python -m pytest tests/test_document_processor.py -v

# Text chunking and boundary split tests (16 tests)
python -m pytest tests/test_text_chunker.py -v

# Text embedding service tests with mock models (13 tests)
python -m pytest tests/test_embedding_service.py -v

# FAISS vector store indexing and retrieval tests (26 tests)
python -m pytest tests/test_vector_store.py -v

# Retrieval evaluation metrics and harness tests (15 tests)
python -m pytest tests/test_retrieval_evaluator.py -v
```

> [!TIP]
> The automated unit tests use deterministic mock models and synthetic vectors. They run in under 3 seconds (aside from the initial PyTorch import) and require **no internet connection** and **no model downloads**.

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

---

## 13. FAISS Vector Store

### 13.1 What a Vector Database / Index Does
Traditional relational databases and search engines index text using keyword lookups or inverted indexes. In contrast, a **vector index** stores high-dimensional numerical vectors (embeddings) produced by neural networks and indexes them geometrically. 

When a user submits a query, the query is converted into an embedding in the same vector space. The vector store computes geometric distances between the query vector and stored document vectors, retrieving the passages that are conceptually closest in meaning, even if they share zero keywords.

### 13.2 Why IntelliAssist AI Uses FAISS
**FAISS (Facebook AI Similarity Search)** is the industry-standard library for efficient similarity search and clustering of dense vectors.
- **Extreme Speed**: Written in highly optimized C++ with AVX/SSE vector instructions and multi-threaded CPU execution.
- **100% Local & Self-Contained**: Operates entirely in process memory on the host machine without requiring external cloud databases (e.g. Pinecone, Qdrant Cloud), Docker containers, or network latency.
- **Low Memory Overhead**: Direct flat arrays with negligible indexing overhead for thousands of chunks.

### 13.3 What IndexFlatIP Means
`faiss.IndexFlatIP` stands for **Index Flat Inner Product**:
- **Flat**: Exact, brute-force search without lossy quantization or compression. Every query is compared against all vectors in the index, guaranteeing **100% recall** (exact nearest neighbors) without approximations.
- **IP (Inner Product)**: The distance metric computed between query vector $q$ and stored vector $v$ is the dot product:
  $$\text{IP}(q, v) = \sum_{i=1}^{d} q_i \cdot v_i$$

### 13.4 Why Normalized Embeddings Equal Cosine Similarity
Cosine similarity measures the angle between two non-zero vectors in multi-dimensional space:
$$\text{Cosine Similarity}(q, v) = \frac{q \cdot v}{\|q\|_2 \cdot \|v\|_2}$$

When both vectors are **L2-normalized** such that their Euclidean lengths are exactly 1 ($\|q\|_2 = 1$ and $\|v\|_2 = 1$):
$$\text{Cosine Similarity}(q, v) = \frac{q \cdot v}{1 \cdot 1} = q \cdot v = \text{Inner Product}(q, v)$$

By enforcing unit L2 normalization on all vectors, `IndexFlatIP` directly computes **exact cosine similarity** scores in the range $[-1.0, 1.0]$. The store validates vector norms and normalizes them automatically by default (`normalize_vectors=True`), or verifies unit length under strict mode (`normalize_vectors=False`).

> [!NOTE]
> Similarity scores represent cosine similarity, **not probabilities**. A score of `1.0` denotes identical semantic orientation, `0.0` denotes orthogonality (unrelated concepts), and negative values denote opposing semantic directions.

### 13.5 What top_k Means
The `top_k` parameter specifies the maximum number of most-similar document chunks to return. For example, `top_k=5` retrieves the 5 vectors with the highest similarity scores.
- If the index contains fewer vectors than `top_k`, the store safely returns only the available records without padding or errors.
- Any FAISS internal `-1` sentinel markers (indicating unfilled neighbor slots) are safely filtered out before returning results.
- `top_k` must be a positive integer ($k \ge 1$).

### 13.6 How Chunk IDs and Page Metadata are Preserved
FAISS internally assigns sequential integer row indices (`0, 1, 2, ...`) to indexed vectors. To prevent loss of document context, `FAISSVectorStore` maintains an explicit parallel record mapping:
1. Position `i` in FAISS directly corresponds to index `i` in `self._records`.
2. Each record stores:
   - **`chunk_id`**: Deterministic unique identifier (e.g. `"handbook.pdf_page_1_chunk_1"`).
   - **`text`**: Complete text excerpt for that chunk.
   - **`metadata`**: Source attribution including `filename`, `file_type`, `page_number`, and `chunk_index`.
3. In addition, `self._chunk_ids` maintains an active set of stored chunk IDs to reject duplicates in $O(1)$ time and guarantee that every chunk ID is globally unique within the index.

### 13.7 How Multiple Documents Share an Index
A single `FAISSVectorStore` instance can hold chunks from dozens of different uploaded documents:
- Chunk IDs incorporate the filename and page index (`f"{filename}_page_{page_number}_chunk_{chunk_index}"`), preventing namespace collisions.
- Search queries retrieve chunks ranked strictly by semantic relevance, regardless of which document they originated from.
- Returned metadata preserves the source document name and exact page number, enabling precise citations and multi-document synthesis.

### 13.8 Installing Dependencies
Ensure the virtual environment is activated, then install `faiss-cpu`:

```powershell
python -m pip install faiss-cpu>=1.8.0
```

### 13.9 Running the Vector Store Tests
Run the 26 automated unit tests for `FAISSVectorStore`:

```powershell
python -m pytest tests/test_vector_store.py -v
```

These tests verify empty index safety, single and multi-chunk additions, vector counts, nearest-neighbor accuracy, descending ranking, multi-document retrieval, top_k boundary cases, invalid input rejection, duplicate ID rejection, NaN/Inf rejection, index clearing, and atomic batch insertions.

### 13.10 Running the Vector Store Smoke Test
Run the standalone deterministic smoke test:

```powershell
python smoke_test_vector_store.py
```

This smoke test:
1. Initializes `FAISSVectorStore(dimension=384)`.
2. Indexes sample chunks from `handbook.pdf` and `security_policy.docx`.
3. Queries with a synthetic vector matching the security policy.
4. Prints the ranked chunk IDs, similarity scores, source filenames, and page numbers.
5. Verifies that the best-matching chunk is ranked #1.
6. Tests index clearing and safe empty-index searching.

### 13.11 Current Limitations
1. **In-Memory Storage Only**: The FAISS index and metadata mappings live purely in process memory (`RAM`). All indexed vectors and metadata reset when the Python process terminates. Disk serialization (e.g. `faiss.write_index`) is scheduled for future milestones.
2. **Batch Re-indexing**: Because this uses flat in-memory indexing, removing a single chunk without rebuilding the index is not supported; use `store.clear()` to flush the entire store.
3. **Scope Boundary**: Full RAG retrieval pipelines, LLM answer synthesis, and chatbot API endpoints belong to subsequent milestone days.

### 13.12 Complete End-to-End Python Example

```python
from pathlib import Path
from app.document_processor import process_document
from app.text_chunker import chunk_document
from app.embedding_service import EmbeddingService
from app.vector_store import FAISSVectorStore

# 1. Process documents (PDF, TXT, DOCX)
doc_a = process_document(Path("sample.pdf"))

# 2. Split into overlapping chunks
chunks_a = chunk_document(doc_a, chunk_size=800, chunk_overlap=150)
print(f"Generated {len(chunks_a)} chunks from {doc_a['filename']}.")

# 3. Generate dense 384-dimensional embeddings
embedder = EmbeddingService()
embedded_chunks = embedder.embed_chunks(chunks_a)

# 4. Initialize FAISS vector store and insert chunks
store = FAISSVectorStore(dimension=384, normalize_vectors=True)
added = store.add_chunks(embedded_chunks)
print(f"Added {added} chunks to FAISS index. Total vectors: {store.total_vectors}")

# 5. Embed a user search query
query_text = "What are the security guidelines and policies?"
query_vec = embedder.embed_text(query_text)

# 6. Retrieve top-k most relevant chunks
results = store.search(query_vec, top_k=3)

print("\n--- Search Results ---")
for rank, res in enumerate(results, start=1):
    print(f"\nRank #{rank}:")
    print(f"  Chunk ID:         {res['chunk_id']}")
    print(f"  Cosine Sim Score: {res['similarity_score']:.4f}")
    print(f"  Source File:      {res['metadata']['filename']}")
    print(f"  Page Number:      {res['metadata']['page_number']}")
    print(f"  Excerpt:          {res['text'][:150]}...")
```

---

## 14. Retrieval Testing and Quality Evaluation

### 14.1 Purpose of Retrieval Evaluation
In a Retrieval-Augmented Generation (RAG) system, downstream answer accuracy is strictly bounded by retrieval quality: if the vector store fails to retrieve the correct passage containing the answer fact, an LLM cannot synthesize an accurate response.

Retrieval evaluation benchmarks whether the existing local embedding service (`sentence-transformers/all-MiniLM-L6-v2`) and FAISS vector store (`IndexFlatIP`) consistently rank the ground-truth document chunks within the top retrieval results for natural-language questions.

### 14.2 Evaluation Corpus Structure
The evaluation system uses a realistic, self-contained multi-document corpus located in `backend/evaluation_data/`:

| Document Filename | Format | Key Subject Matter | Character Count | Chunks |
| :--- | :--- | :--- | :--- | :--- |
| `security_policy.txt` | TXT | Password requirements, MFA policies, account lockout, access control / RBAC | ~1,732 chars | 3 chunks |
| `employee_handbook.txt` | TXT | Paid time off (PTO), sick leave, working hours, core hours, remote work, attendance | ~2,194 chars | 4 chunks |
| `expense_policy.txt` | TXT | Reimbursable expenses, meal per diem, itemized receipt limits, approval tiers, deadlines | ~2,151 chars | 4 chunks |

Total corpus size: 3 documents, 11 indexed chunks.

A standardized dataset of 12 natural-language questions is maintained in `backend/evaluation_data/evaluation_queries.json`. The dataset balances:
- **Direct factual questions** (e.g., specific dollar limits or character requirements)
- **Paraphrased questions** (e.g., asking about "two-factor authentication" rather than "MFA")
- **Questions using synonyms** (e.g., "vacation days" for "PTO", "stipend" for "equipment reimbursement")
- **Questions with differing grammatical structures** from the source documents
- **Balanced document coverage** (4 queries per document)

### 14.3 Definition of Ground-Truth Relevance
A retrieved chunk is deemed **relevant** to an evaluation query if and only if:
1. **Document Match**: The retrieved chunk's metadata filename exactly matches the query's `expected_doc`.
2. **Passage Match**: Either:
   - The chunk's `chunk_id` is within the query's list of `expected_chunk_ids`, OR
   - The chunk's text contains the query's `expected_passage_text` (case-insensitive substring match).

Irrelevant chunks or chunks from other documents are strictly non-relevant, ensuring evaluation integrity without false positives.

### 14.4 Retrieval Quality Metrics and Mathematical Formulas

Evaluation calculates standard Information Retrieval (IR) metrics evaluated at cutoffs $K \in \{1, 3, 5\}$:

#### A. Hit Rate@K (Success Rate)
The proportion of evaluation queries for which at least one relevant chunk appears within the top $K$ retrieved results:

$$\text{Hit}_i@K = \begin{cases} 1 & \text{if } \left| \text{Retrieved}_i[:K] \cap \text{Relevant}_i \right| \ge 1 \\ 0 & \text{otherwise} \end{cases}$$

$$\text{Hit Rate}@K = \frac{1}{N} \sum_{i=1}^{N} \text{Hit}_i@K$$

#### B. Recall@K
For each query, the fraction of all known relevant chunks that are retrieved in the top $K$ results:

$$\text{Recall}_i@K = \frac{\left| \text{Retrieved}_i[:K] \cap \text{Relevant}_i \right|}{\left| \text{Relevant}_i \right|}$$

$$\text{Recall}@K = \frac{1}{N} \sum_{i=1}^{N} \text{Recall}_i@K$$

> [!NOTE]
> **Single-Relevant Chunk Equivalence**: When an evaluation query has exactly one known relevant chunk ($|\text{Relevant}_i| = 1$), $\text{Recall}_i@K$ and $\text{Hit}_i@K$ are mathematically identical (both yield $1.0$ if the chunk is retrieved in top $K$, and $0.0$ otherwise).

#### C. Mean Reciprocal Rank (MRR)
Measures the position of the first relevant chunk in the ranked search results:

$$\text{RR}_i = \begin{cases} \frac{1}{\text{rank}_i} & \text{if a relevant chunk appears in retrieved results at 1-based rank } \text{rank}_i \\ 0 & \text{if no relevant chunk is retrieved} \end{cases}$$

$$\text{MRR} = \frac{1}{N} \sum_{i=1}^{N} \text{RR}_i$$

- If the first relevant result is at Rank 1, $\text{RR} = 1.0$.
- If at Rank 2, $\text{RR} = 0.5$.
- If at Rank 3, $\text{RR} = 0.3333$.
- If not found in the evaluated candidates, $\text{RR} = 0.0$.

### 14.5 Exact PowerShell Commands to Run

Activate your virtual environment and run the commands from the `backend/` directory:

```powershell
cd "d:\IntelliAssist AI\backend"
.\.venv\Scripts\Activate.ps1
```

#### Run All 89 Automated Unit Tests:
```powershell
python -m pytest -v
```

#### Run the Retrieval Evaluator Unit Tests (15 tests, fast mock mode):
```powershell
python -m pytest tests/test_retrieval_evaluator.py -v
```

#### Run the Real End-to-End Semantic Retrieval Evaluation:
```powershell
python evaluate_retrieval.py
```

### 14.6 How to Interpret the Output
Running `python evaluate_retrieval.py` outputs:
1. **Corpus Ingestion Summary**: Document names, character counts, and total chunks indexed.
2. **Per-Query Log**: Shows the question, expected document, ground-truth fact, first relevant rank, Hit@K, Recall@K, and top 3 retrieved chunks with cosine similarity scores and relevance markers (`[RELEVANT]` / `[NON-RELEVANT]`).
3. **Aggregate Metric Table**: Hit Rate@1, Hit Rate@3, Hit Rate@5, Recall@1, Recall@3, Recall@5, and MRR.
4. **Exported Reports**:
   - `backend/evaluation_results/retrieval_evaluation_report.json`: Full machine-readable evaluation report with per-query retrieved chunks and configuration metadata.
   - `backend/evaluation_results/retrieval_evaluation_report.csv`: Tabular spreadsheet format for query-level analysis.

### 14.7 Automated Unit Tests vs. Actual Semantic Evaluation
It is vital to distinguish between unit tests and the evaluation script:
- **Automated Unit Tests (`tests/test_retrieval_evaluator.py`)**: Use deterministic mock embeddings and synthetic vector responses. They test code correctness, mathematical formula accuracy, ranking logic, boundary cases (empty lists, invalid K), and error handling in under 1 second without internet access or GPU/CPU model loading.
- **Semantic Evaluation Script (`evaluate_retrieval.py`)**: Uses the **actual** `sentence-transformers/all-MiniLM-L6-v2` neural network and real **FAISS IndexFlatIP** index to measure real-world semantic retrieval quality on genuine text passages.

### 14.8 Known Limitations of Small Synthetic Evaluation Corpora
1. **Corpus Scale**: This synthetic evaluation corpus contains 3 documents and 11 chunks. High Hit Rates on an 11-chunk index do not guarantee identical accuracy on a corpus of 100,000 documents where distractor paragraphs and lexical collisions are far more prevalent.
2. **Semantic Distractors**: In question `HB-03` ("Does the company provide financial assistance or a stipend to purchase home office equipment?"), generic terms like "financial assistance" and "reimbursement" caused expense policy chunks to rank slightly higher than the employee handbook stipend chunk (ranking the true chunk at Rank 3 rather than Rank 1).
3. **Synthetic Queries**: Real end-user queries often contain typos, ambiguous intent, multi-hop reasoning requirements, or cross-document dependencies not fully captured by synthetic test queries.
4. **Scope Boundary**: This evaluation measures **retrieval only**. Full RAG generation, answer faithfulness, and LLM hallucination benchmarking are scheduled for subsequent milestone days.

