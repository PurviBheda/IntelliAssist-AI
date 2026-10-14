"""Automated unit tests for IntelliAssist AI Text Chunker.

Tests coverage:
  1. Short text produces one chunk.
  2. Long text produces multiple chunks.
  3. Overlap is applied correctly.
  4. The original page number is preserved.
  5. The source filename is preserved.
  6. Chunks remain in document order.
  7. Each page is processed separately.
  8. Empty pages are skipped.
  9. Invalid chunk_size is rejected.
  10. Invalid chunk_overlap is rejected.
  11. chunk_overlap >= chunk_size is rejected.
  12. Whitespace-only pages produce no chunks.
  13. Malformed document input is rejected clearly.
  14. Chunk IDs are deterministic and unique.
  15. The chunking algorithm terminates for valid configurations.
"""

from __future__ import annotations

import pytest

from app.text_chunker import (
    chunk_document,
    split_text_into_chunks,
)


def test_short_text_produces_one_chunk() -> None:
    """1. Short text produces one chunk."""
    doc = {
        "filename": "short.txt",
        "file_type": "txt",
        "pages": [{"page_number": 1, "text": "This is a concise single page note."}],
        "total_pages": 1,
        "total_characters": 35,
    }
    chunks = chunk_document(doc, chunk_size=800, chunk_overlap=150)
    assert len(chunks) == 1
    assert chunks[0]["text"] == "This is a concise single page note."
    assert chunks[0]["chunk_id"] == "short.txt_page_1_chunk_1"
    assert chunks[0]["metadata"]["chunk_index"] == 1


def test_long_text_produces_multiple_chunks() -> None:
    """2. Long text produces multiple chunks."""
    long_text = "IntelliAssist AI provides advanced document retrieval and cognitive search. " * 30
    doc = {
        "filename": "long_manual.pdf",
        "file_type": "pdf",
        "pages": [{"page_number": 1, "text": long_text}],
        "total_pages": 1,
        "total_characters": len(long_text),
    }
    chunks = chunk_document(doc, chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1
    for chunk in chunks:
        assert len(chunk["text"]) <= 350  # roughly bounded near chunk_size
        assert chunk["metadata"]["filename"] == "long_manual.pdf"
        assert chunk["metadata"]["page_number"] == 1


def test_overlap_is_applied_correctly() -> None:
    """3. Overlap is applied correctly between consecutive chunks."""
    text = (
        "Alpha section introduces the core principles. "
        "Beta section describes the technical architecture and pipeline. "
        "Gamma section covers operational security and enterprise deployment. "
        "Delta section concludes with empirical performance metrics."
    )
    doc = {
        "filename": "guide.txt",
        "file_type": "txt",
        "pages": [{"page_number": 1, "text": text}],
        "total_pages": 1,
        "total_characters": len(text),
    }
    chunks = chunk_document(doc, chunk_size=120, chunk_overlap=40)
    assert len(chunks) >= 2

    # Verify that the beginning of chunk 2 contains text from chunk 1
    chunk_1_text = chunks[0]["text"]
    chunk_2_text = chunks[1]["text"]

    # There should be shared common substring between the end of chunk 1 and start of chunk 2
    words_chunk_1 = set(chunk_1_text.split()[-5:])
    words_chunk_2 = set(chunk_2_text.split()[:5])
    assert bool(words_chunk_1.intersection(words_chunk_2)), (
        f"Expected overlap between Chunk 1: '{chunk_1_text}' and Chunk 2: '{chunk_2_text}'"
    )


def test_original_page_number_is_preserved() -> None:
    """4. The original page number is preserved across all chunks."""
    doc = {
        "filename": "multi_page.pdf",
        "file_type": "pdf",
        "pages": [
            {"page_number": 5, "text": "Page 5 content text."},
            {"page_number": 12, "text": "Page 12 content text."},
        ],
        "total_pages": 2,
        "total_characters": 42,
    }
    chunks = chunk_document(doc)
    assert len(chunks) == 2
    assert chunks[0]["metadata"]["page_number"] == 5
    assert chunks[0]["chunk_id"] == "multi_page.pdf_page_5_chunk_1"
    assert chunks[1]["metadata"]["page_number"] == 12
    assert chunks[1]["chunk_id"] == "multi_page.pdf_page_12_chunk_1"


def test_source_filename_is_preserved() -> None:
    """5. The source filename is preserved in all chunk metadata."""
    doc = {
        "filename": "quarterly_earnings_2026.docx",
        "file_type": "docx",
        "pages": [{"page_number": 1, "text": "Earnings report summary data."}],
        "total_pages": 1,
        "total_characters": 29,
    }
    chunks = chunk_document(doc)
    assert chunks[0]["metadata"]["filename"] == "quarterly_earnings_2026.docx"
    assert chunks[0]["metadata"]["file_type"] == "docx"
    assert "quarterly_earnings_2026.docx" in chunks[0]["chunk_id"]


def test_chunks_remain_in_document_order() -> None:
    """6. Chunks remain in document order."""
    doc = {
        "filename": "ordered_book.pdf",
        "file_type": "pdf",
        "pages": [
            {"page_number": 1, "text": "Chapter 1: The Beginning of the journey."},
            {"page_number": 2, "text": "Chapter 2: The Path forward unfolding."},
            {"page_number": 3, "text": "Chapter 3: The Final destination reached."},
        ],
        "total_pages": 3,
        "total_characters": 115,
    }
    chunks = chunk_document(doc)
    assert len(chunks) == 3
    assert chunks[0]["metadata"]["page_number"] == 1
    assert "Chapter 1" in chunks[0]["text"]
    assert chunks[1]["metadata"]["page_number"] == 2
    assert "Chapter 2" in chunks[1]["text"]
    assert chunks[2]["metadata"]["page_number"] == 3
    assert "Chapter 3" in chunks[2]["text"]


def test_each_page_is_processed_separately() -> None:
    """7. Each page is processed separately without mixing text across pages."""
    doc = {
        "filename": "distinct_pages.pdf",
        "file_type": "pdf",
        "pages": [
            {"page_number": 1, "text": "Page one confidential data strictly isolated."},
            {"page_number": 2, "text": "Page two public disclosure statements."},
        ],
        "total_pages": 2,
        "total_characters": 83,
    }
    chunks = chunk_document(doc, chunk_size=800, chunk_overlap=150)
    assert len(chunks) == 2

    # Verify no page 1 text appears in page 2 chunk, and vice versa
    assert "Page one" in chunks[0]["text"]
    assert "Page two" not in chunks[0]["text"]
    assert chunks[0]["metadata"]["page_number"] == 1

    assert "Page two" in chunks[1]["text"]
    assert "Page one" not in chunks[1]["text"]
    assert chunks[1]["metadata"]["page_number"] == 2


def test_empty_pages_are_skipped() -> None:
    """8. Empty pages are skipped without creating meaningless chunks."""
    doc = {
        "filename": "gaps.pdf",
        "file_type": "pdf",
        "pages": [
            {"page_number": 1, "text": ""},
            {"page_number": 2, "text": "Only valid content on page two."},
            {"page_number": 3, "text": ""},
        ],
        "total_pages": 3,
        "total_characters": 31,
    }
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0]["metadata"]["page_number"] == 2
    assert chunks[0]["metadata"]["chunk_index"] == 1
    assert chunks[0]["text"] == "Only valid content on page two."


def test_invalid_chunk_size_is_rejected() -> None:
    """9. Invalid chunk_size is rejected with clear ValueError messages."""
    doc = {
        "filename": "test.txt",
        "file_type": "txt",
        "pages": [{"page_number": 1, "text": "Sample text"}],
        "total_pages": 1,
        "total_characters": 11,
    }
    # Zero chunk_size
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        chunk_document(doc, chunk_size=0)

    # Negative chunk_size
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        chunk_document(doc, chunk_size=-100)

    # Non-integer (boolean)
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        chunk_document(doc, chunk_size=True)  # type: ignore

    # Non-integer (float)
    with pytest.raises(ValueError, match="chunk_size must be a positive integer"):
        chunk_document(doc, chunk_size=500.5)  # type: ignore


def test_invalid_chunk_overlap_is_rejected() -> None:
    """10. Invalid chunk_overlap is rejected with clear ValueError messages."""
    doc = {
        "filename": "test.txt",
        "file_type": "txt",
        "pages": [{"page_number": 1, "text": "Sample text"}],
        "total_pages": 1,
        "total_characters": 11,
    }
    # Negative overlap
    with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
        chunk_document(doc, chunk_size=800, chunk_overlap=-1)

    # Boolean overlap
    with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
        chunk_document(doc, chunk_size=800, chunk_overlap=False)  # type: ignore

    # String overlap
    with pytest.raises(ValueError, match="chunk_overlap must be a non-negative integer"):
        chunk_document(doc, chunk_size=800, chunk_overlap="50")  # type: ignore


def test_chunk_overlap_greater_equal_chunk_size_is_rejected() -> None:
    """11. chunk_overlap >= chunk_size is rejected."""
    doc = {
        "filename": "test.txt",
        "file_type": "txt",
        "pages": [{"page_number": 1, "text": "Sample text"}],
        "total_pages": 1,
        "total_characters": 11,
    }
    # Equal
    with pytest.raises(ValueError, match="strictly smaller than chunk_size"):
        chunk_document(doc, chunk_size=500, chunk_overlap=500)

    # Greater
    with pytest.raises(ValueError, match="strictly smaller than chunk_size"):
        chunk_document(doc, chunk_size=500, chunk_overlap=600)


def test_whitespace_only_pages_produce_no_chunks() -> None:
    """12. Whitespace-only pages produce no chunks."""
    doc = {
        "filename": "blanks.pdf",
        "file_type": "pdf",
        "pages": [
            {"page_number": 1, "text": "   \n\t   \n  "},
            {"page_number": 2, "text": "\n\n   \n"},
        ],
        "total_pages": 2,
        "total_characters": 0,
    }
    chunks = chunk_document(doc)
    assert chunks == []


def test_malformed_document_input_is_rejected() -> None:
    """13. Malformed document input is rejected clearly."""
    # None document
    with pytest.raises(ValueError, match="cannot be None"):
        chunk_document(None)  # type: ignore

    # Non-dictionary
    with pytest.raises(ValueError, match="must be a dictionary"):
        chunk_document("not a dict")  # type: ignore

    # Missing filename
    with pytest.raises(ValueError, match="missing required key"):
        chunk_document({"file_type": "pdf", "pages": []})

    # Non-string filename
    with pytest.raises(ValueError, match="filename"):
        chunk_document({"filename": 123, "file_type": "pdf", "pages": []})

    # Non-list pages
    with pytest.raises(ValueError, match="pages"):
        chunk_document({"filename": "a.pdf", "file_type": "pdf", "pages": "not a list"})

    # Malformed page entry
    with pytest.raises(ValueError, match="missing 'text'"):
        chunk_document({
            "filename": "a.pdf",
            "file_type": "pdf",
            "pages": [{"page_number": 1}],
        })


def test_chunk_ids_are_deterministic_and_unique() -> None:
    """14. Chunk IDs are deterministic and unique."""
    doc = {
        "filename": "contract.pdf",
        "file_type": "pdf",
        "pages": [
            {"page_number": 1, "text": "Paragraph A " * 40},
            {"page_number": 2, "text": "Paragraph B " * 40},
        ],
        "total_pages": 2,
        "total_characters": 960,
    }
    chunks_run_1 = chunk_document(doc, chunk_size=200, chunk_overlap=40)
    chunks_run_2 = chunk_document(doc, chunk_size=200, chunk_overlap=40)

    # Determinism across runs
    assert len(chunks_run_1) == len(chunks_run_2)
    ids_1 = [c["chunk_id"] for c in chunks_run_1]
    ids_2 = [c["chunk_id"] for c in chunks_run_2]
    assert ids_1 == ids_2

    # Uniqueness within document
    assert len(ids_1) == len(set(ids_1))

    # Pattern check
    for c in chunks_run_1:
        expected_prefix = f"contract.pdf_page_{c['metadata']['page_number']}_chunk_{c['metadata']['chunk_index']}"
        assert c["chunk_id"] == expected_prefix


def test_chunking_terminates_for_extreme_valid_configurations() -> None:
    """15. The chunking algorithm terminates for valid edge-case configurations."""
    # Tiny chunk size with near-maximum overlap
    text = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    chunks = split_text_into_chunks(text, chunk_size=5, chunk_overlap=4)
    assert len(chunks) > 0
    # Overlap of zero
    chunks_zero_overlap = split_text_into_chunks(text, chunk_size=10, chunk_overlap=0)
    assert len(chunks_zero_overlap) > 0

    # Ensure no infinite loop occurs with repetitive text
    repeated = "word " * 100
    chunks_rep = split_text_into_chunks(repeated, chunk_size=50, chunk_overlap=25)
    assert len(chunks_rep) > 0


def test_no_text_silently_lost() -> None:
    """Verifies that every character/word in the source is preserved across chunks."""
    words = [f"token_{i}" for i in range(100)]
    full_text = " ".join(words)
    doc = {
        "filename": "tokens.txt",
        "file_type": "txt",
        "pages": [{"page_number": 1, "text": full_text}],
        "total_pages": 1,
        "total_characters": len(full_text),
    }
    chunks = chunk_document(doc, chunk_size=150, chunk_overlap=40)

    # Collect all words present in chunks
    chunk_words = set()
    for c in chunks:
        chunk_words.update(c["text"].split())

    for word in words:
        assert word in chunk_words, f"Word '{word}' was lost during chunking!"
