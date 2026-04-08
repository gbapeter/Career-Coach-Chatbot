import os
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

# Use pure python to resolve compatibility issues with C++
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

CHROMA_DIR = "./chroma_db"


def get_vectorstore():
    # Load the Chroma vector store
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name="career_advisor_2026"
    )


def retrieve_with_sources(query: str, k: int = 6):
    """
    Retrieve relevant documents and return clean context + deduplicated sources.
    """
    try:
        vectorstore = get_vectorstore()
        
        # Perform similarity search
        docs = vectorstore.similarity_search(query, k=k)

        # Build context by joining document content
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])

        # Deduplicate sources while preserving first occurrence order
        seen = set()
        sources = []
        for doc in docs:
            src_url = doc.metadata.get("source", "Unknown source")
            src_type = doc.metadata.get("type", "url")
            
            if src_url not in seen:
                seen.add(src_url)
                sources.append({
                    "url": src_url,
                    "type": src_type
                })

        return context, sources

    except Exception as e:
        print(f"Retrieval error: {e}")
        # Fallback in case of error
        return "Sorry, I couldn't retrieve relevant information from the knowledge base at the moment.", []