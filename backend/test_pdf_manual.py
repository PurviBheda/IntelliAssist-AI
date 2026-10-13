from app.document_processor import process_document


def main():
    result = process_document("sample.pdf")

    print("Filename:", result["filename"])
    print("File type:", result["file_type"])
    print("Total pages:", result["total_pages"])
    print("Total characters:", result["total_characters"])

    for page in result["pages"]:
        print(f"\n--- Page {page['page_number']} ---")
        print(page["text"])


if __name__ == "__main__":
    main()