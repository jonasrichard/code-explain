from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import Language, RecursiveCharacterTextSplitter


def create_vector_store(documents: list[Document]) -> FAISS:
    print(f"Embed {len(documents)} documents...")

    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    vector_store = FAISS.from_documents(documents, embeddings)

    return vector_store


def embed_document(vector_store: FAISS, documents: list[Document]):
    vector_store.add_documents(documents)


def read_source_file(path: Path) -> list[Document]:
    """Read Golang source file as a document for embedding"""

    tf = path.open()
    content = tf.read()

    go_splitter = RecursiveCharacterTextSplitter.from_language(
        language=Language.GO, chunk_size=500, chunk_overlap=0
    )

    documents = go_splitter.create_documents([content])

    return documents
