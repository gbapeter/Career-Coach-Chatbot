import os
import io
import requests
import tempfile
from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
# from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from tqdm import tqdm
from sources import PDF_URLS, URL_SOURCES

# Use pure python to resolve compatibility issues with C++
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

CHROMA_DIR = "./chroma_db"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/134.0.0.0 Safari/537.36"
    )
}


def load_pdf_from_url(url: str) -> list[Document]:
    """Download PDF and extract as LangChain Documents."""
    print(f"  Downloading PDF: {url}")
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
    except Exception as e:
        print(f"  [ERROR] Could not load PDF from {url}: {e}")
        return []

    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        tmp.write(response.content)
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = url
            page.metadata["type"] = "pdf"
        print(f"  [OK] {len(pages)} pages extracted from PDF")
        return pages
    finally:
        os.unlink(tmp_path)


def load_all_pdfs() -> list[Document]:
    docs = []
    for url in PDF_URLS:
        docs.extend(load_pdf_from_url(url))
    return docs


def scrape_url(url: str) -> str:
    """Clean scrape of webpage."""
    try:
        print(f"  Scraping: {url}")
        response = requests.get(url, headers=HEADERS, timeout=20)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        for tag in soup(["script", "style", "nav", "footer", "header", "aside", "button", "iframe"]):
            tag.decompose()

        main = soup.find("article") or soup.find("main") or soup.body
        raw = main.get_text(separator="\n") if main else soup.get_text(separator="\n")

        lines = [line.strip() for line in raw.splitlines() if line.strip()]
        return "\n".join(lines)
    except Exception as e:
        print(f"  [ERROR] {url}: {e}")
        return ""


def load_all_urls() -> list[Document]:
    docs = []
    for url in URL_SOURCES:
        text = scrape_url(url)
        if len(text) < 300:
            print(f"  [SKIP] Too little content at: {url}")
            continue
        doc = Document(
            page_content=text,
            metadata={"source": url, "type": "url"}
        )
        docs.append(doc)
        print(f"  [OK] {len(text):,} chars scraped")
    return docs


def ingest():
    print("\n" + "="*60)
    print("STEP 1: Loading PDFs")
    print("="*60)
    pdf_docs = load_all_pdfs()
    print(f"Total PDF pages loaded: {len(pdf_docs)}")

    print("\n" + "="*60)
    print("STEP 2: Scraping web pages")
    print("="*60)
    url_docs = load_all_urls()
    print(f"Total web pages loaded: {len(url_docs)}")

    all_docs = pdf_docs + url_docs
    print(f"\nGrand total documents: {len(all_docs)}")

    if not all_docs:
        print("No documents loaded. Exiting.")
        return

    print("\n" + "="*60)
    print("STEP 3: Chunking documents")
    print("="*60)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150,     
        separators=["\n\n", "\n", ". ", "!", "?", " "],
        length_function=len,
    )
    chunks = splitter.split_documents(all_docs)
    print(f"Total chunks created: {len(chunks)}")

    print("\n" + "="*60)
    print("STEP 4: Embedding & storing in ChromaDB")
    print("="*60)

    # Clear old database for clean re-ingestion
    if os.path.exists(CHROMA_DIR):
        import shutil
        shutil.rmtree(CHROMA_DIR)
        print("Cleared old ChromaDB.")

    embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )


    vectorstore = Chroma(
    persist_directory=CHROMA_DIR,
    embedding_function=embeddings,
    collection_name="career_advisor_2026"
)

    batch_size = 100

    for i in tqdm(range(0, len(chunks), batch_size)):
        batch = chunks[i:i + batch_size]
        vectorstore.add_documents(batch)

    vectorstore.persist()

    print(f"\n Done! {len(chunks)} chunks embedded and stored in '{CHROMA_DIR}'")


if __name__ == "__main__":
    ingest()