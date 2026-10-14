"""IntelliAssist AI - Text Chunking Module.

This module splits extracted document pages into smaller, overlapping text chunks
optimized for semantic search and vector embeddings. It preserves document structure,
page numbers, source filenames, and natural text boundaries (paragraphs and sentences).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 150


def split_text_into_chunks(
    text: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[str]:
    """Split a string of text into chunks of approximately chunk_size characters with overlap.

    Splits prefer natural boundaries in the following order of priority:
      1. Paragraph breaks (\\n\\n)
      2. Line breaks (\\n)
      3. Sentence endings (. , ! , ? , .\\n, !\\n, ?\\n)
      4. Word boundaries (spaces)
      5. Hard split at chunk_size if no boundary is found

    Args:
        text: Raw text string to split.
        chunk_size: Maximum character count per chunk. Must be a positive integer.
        chunk_overlap: Number of characters to overlap between consecutive chunks.
                       Must be a non-negative integer strictly less than chunk_size.

    Returns:
        List of non-empty text chunk strings.

    Raises:
        ValueError: If chunk_size or chunk_overlap configuration is invalid.
    """
    _validate_chunk_config(chunk_size, chunk_overlap)

    cleaned = text.strip()
    if not cleaned:
        return []

    total_len = len(cleaned)
    if total_len <= chunk_size:
        return [cleaned]

    chunks: List[str] = []
    start_idx = 0

    while start_idx < total_len:
        # If the remaining portion fits comfortably in one chunk, consume it
        remaining = total_len - start_idx
        if remaining <= chunk_size:
            final_chunk = cleaned[start_idx:].strip()
            if final_chunk:
                chunks.append(final_chunk)
            break

        # Proposed cut boundary for this chunk
        target_end = start_idx + chunk_size

        # Search window for boundary delimiters: look within the latter portion
        # of the chunk to keep chunk sizes reasonable.
        # Ensure search_start is strictly ahead of start_idx to guarantee progress.
        search_start = max(start_idx + 1, target_end - max(chunk_overlap, chunk_size // 3))

        split_pos = _find_split_boundary(cleaned, start_idx, search_start, target_end)

        chunk_text = cleaned[start_idx:split_pos].strip()
        if chunk_text:
            chunks.append(chunk_text)

        # Calculate start index for next chunk respecting chunk_overlap
        if chunk_overlap > 0:
            raw_next_start = max(start_idx + 1, split_pos - chunk_overlap)
            # Snap forward to next word boundary within [raw_next_start, split_pos]
            space_pos = cleaned.find(" ", raw_next_start, split_pos)
            if space_pos != -1 and (space_pos + 1) < split_pos:
                next_start = space_pos + 1
            else:
                next_start = raw_next_start
        else:
            next_start = split_pos

        # Ensure next_start strictly advances start_idx to prevent infinite loops
        if next_start <= start_idx:
            next_start = start_idx + 1

        # Advance past any leading whitespace for the subsequent chunk
        while next_start < total_len and cleaned[next_start].isspace():
            next_start += 1

        start_idx = next_start

    return chunks


def _find_split_boundary(
    text: str, start_idx: int, search_start: int, target_end: int
) -> int:
    """Find the best natural split index in text between search_start and target_end.

    Checks for:
      1. Paragraph breaks (\\n\\n)
      2. Line breaks (\\n)
      3. Sentence endings (. , ! , ? , etc.)
      4. Word boundaries (spaces)
    Falls back to target_end if no boundary is found.
    """
    # 1. Paragraph boundary: \n\n
    p_idx = text.rfind("\n\n", search_start, target_end)
    if p_idx != -1:
        return p_idx + 2

    # 2. Line boundary: \n
    nl_idx = text.rfind("\n", search_start, target_end)
    if nl_idx != -1:
        return nl_idx + 1

    # 3. Sentence boundary: punctuation followed by space or newline
    best_sent_idx = -1
    for punct in (". ", "! ", "? ", ".\n", "!\n", "?\n"):
        idx = text.rfind(punct, search_start, target_end)
        if idx != -1 and idx > best_sent_idx:
            best_sent_idx = idx

    if best_sent_idx != -1:
        # Include punctuation mark in current chunk, split immediately after
        return best_sent_idx + 1

    # 4. Word boundary: space in preferred search window
    sp_idx = text.rfind(" ", search_start, target_end)
    if sp_idx != -1:
        return sp_idx + 1

    # Fallback word boundary: space anywhere between start_idx + 1 and target_end
    earlier_sp = text.rfind(" ", start_idx + 1, target_end)
    if earlier_sp != -1:
        return earlier_sp + 1

    # Hard split if no boundary found (e.g. unbroken token or URL)
    return target_end


def _validate_chunk_config(chunk_size: int, chunk_overlap: int) -> None:
    """Validate chunk_size and chunk_overlap parameters.

    Raises:
        ValueError: If chunk_size <= 0, chunk_overlap < 0, or chunk_overlap >= chunk_size.
    """
    # Check boolean explicitly because in Python isinstance(True, int) is True
    if not isinstance(chunk_size, int) or isinstance(chunk_size, bool) or chunk_size <= 0:
        raise ValueError(
            f"chunk_size must be a positive integer, got: {chunk_size!r}"
        )

    if (
        not isinstance(chunk_overlap, int)
        or isinstance(chunk_overlap, bool)
        or chunk_overlap < 0
    ):
        raise ValueError(
            f"chunk_overlap must be a non-negative integer, got: {chunk_overlap!r}"
        )

    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"chunk_overlap ({chunk_overlap}) must be strictly smaller than chunk_size ({chunk_size})"
        )


def _validate_processed_document(processed_document: Any) -> None:
    """Validate structure and types of the processed_document dictionary.

    Raises:
        ValueError: If processed_document is None, not a dict, or missing required fields.
    """
    if processed_document is None:
        raise ValueError("processed_document cannot be None")

    if not isinstance(processed_document, dict):
        raise ValueError(
            f"processed_document must be a dictionary, got: {type(processed_document).__name__}"
        )

    required_keys = ("filename", "file_type", "pages")
    missing_keys = [k for k in required_keys if k not in processed_document]
    if missing_keys:
        raise ValueError(
            f"processed_document missing required key(s): {', '.join(missing_keys)}"
        )

    filename = processed_document["filename"]
    if not isinstance(filename, str) or not filename.strip():
        raise ValueError("processed_document['filename'] must be a non-empty string")

    file_type = processed_document["file_type"]
    if not isinstance(file_type, str):
        raise ValueError("processed_document['file_type'] must be a string")

    pages = processed_document["pages"]
    if not isinstance(pages, list):
        raise ValueError("processed_document['pages'] must be a list")

    for idx, page in enumerate(pages):
        if not isinstance(page, dict):
            raise ValueError(
                f"Page at index {idx} must be a dictionary, got: {type(page).__name__}"
            )
        if "page_number" not in page:
            raise ValueError(f"Page at index {idx} missing 'page_number' field")
        if "text" not in page:
            raise ValueError(f"Page at index {idx} missing 'text' field")

        page_num = page["page_number"]
        if not isinstance(page_num, int) or isinstance(page_num, bool):
            raise ValueError(
                f"Page at index {idx} has invalid 'page_number': {page_num!r} (must be an integer)"
            )

        page_text = page["text"]
        if not isinstance(page_text, str):
            raise ValueError(
                f"Page at index {idx} has invalid 'text': {type(page_text).__name__} (must be a string)"
            )


def chunk_document(
    processed_document: Dict[str, Any],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Dict[str, Any]]:
    """Chunk a processed document into structured, overlapping chunks for embedding.

    Each page is chunked individually so that citation and page attribution remain
    precise. Empty or whitespace-only pages are skipped.

    Args:
        processed_document: Dictionary returned by process_document() containing:
            - 'filename' (str)
            - 'file_type' (str)
            - 'pages' (list of dicts with 'page_number' and 'text')
        chunk_size: Target maximum character count per chunk (default 800).
        chunk_overlap: Target character overlap between consecutive chunks (default 150).

    Returns:
        List of chunk dictionaries with the following schema:
        [
            {
                "chunk_id": "sample.pdf_page_1_chunk_1",
                "text": "Extracted chunk text...",
                "metadata": {
                    "filename": "sample.pdf",
                    "file_type": "pdf",
                    "page_number": 1,
                    "chunk_index": 1
                }
            }
        ]

    Raises:
        ValueError: If input document is malformed or chunk parameters are invalid.
    """
    _validate_processed_document(processed_document)
    _validate_chunk_config(chunk_size, chunk_overlap)

    filename: str = processed_document["filename"]
    file_type: str = processed_document["file_type"]
    pages: List[Dict[str, Any]] = processed_document["pages"]

    document_chunks: List[Dict[str, Any]] = []

    for page in pages:
        page_number: int = page["page_number"]
        page_text: str = page["text"]

        text_chunks = split_text_into_chunks(
            text=page_text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        for chunk_idx, chunk_text in enumerate(text_chunks, start=1):
            chunk_id = f"{filename}_page_{page_number}_chunk_{chunk_idx}"
            document_chunks.append({
                "chunk_id": chunk_id,
                "text": chunk_text,
                "metadata": {
                    "filename": filename,
                    "file_type": file_type,
                    "page_number": page_number,
                    "chunk_index": chunk_idx,
                },
            })

    return document_chunks
