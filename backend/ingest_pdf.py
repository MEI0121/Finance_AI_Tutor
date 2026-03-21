from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

BACKEND_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BACKEND_DIR.parent
DATA_DIR = PROJECT_ROOT / "data"
PDF_FILENAME = "Discounted Dividend Valuation.pdf"
CHROMA_DIR = BACKEND_DIR / "chroma_db"
COLLECTION_NAME = "knowledge_base"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def extract_pdf_text(pdf_path: Path) -> str | None:
    if not pdf_path.is_file():
        print(f"PDF not found: {pdf_path}")
        return None

    reader = PdfReader(str(pdf_path))
    page_count = len(reader.pages)
    if page_count == 0:
        print("PDF has no pages.")
        return None

    parts: list[str] = []
    page_index = 0
    while page_index < page_count:
        page = reader.pages[page_index]
        text = page.extract_text()
        if text:
            parts.append(text)
        page_index += 1

    combined = "\n\n".join(parts).strip()
    if not combined:
        print("No text extracted from PDF.")
        return None

    return combined


def main() -> None:
    pdf_path = DATA_DIR / PDF_FILENAME
    full_text = extract_pdf_text(pdf_path)
    if full_text is None:
        return

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_text(full_text)
    if not chunks:
        print("No chunks produced.")
        return

    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    embedding_fn = DefaultEmbeddingFunction()
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    existing_names = [c.name for c in client.list_collections()]
    if COLLECTION_NAME in existing_names:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedding_fn,
    )

    ids = [f"chunk_{i}" for i in range(len(chunks))]
    metadatas = [{"source": PDF_FILENAME} for _ in chunks]
    collection.add(documents=chunks, ids=ids, metadatas=metadatas)

    print(f"Ingested {len(chunks)} chunks into {CHROMA_DIR}")


if __name__ == "__main__":
    main()
