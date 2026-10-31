import uuid

from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


QDRANT_DIR = "qdrant_db"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"}
    )


def build_vector_store(transcript: str) -> QdrantVectorStore:
    print("Building Qdrant Vector Store...", flush=True)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_text(transcript)

    docs = [
        Document(
            page_content=chunk,
            metadata={"chunk_index": i}
        )
        for i, chunk in enumerate(chunks)
    ]

    print(f"Total chunks: {len(docs)}", flush=True)

    embeddings = get_embeddings()

    print("Embeddings loaded. Adding documents to Qdrant...", flush=True)

    vector_store = QdrantVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        path=QDRANT_DIR,
        collection_name=f"video_{uuid.uuid4().hex[:12]}"
    )

    print("Documents successfully added to Qdrant!", flush=True)

    return vector_store


def get_retriever(vector_store: QdrantVectorStore, k: int = 4):
    return vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )