from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings

import embed
import query


def read_project(root: Path) -> list[Document]:
    documents = []

    for entry in root.iterdir():
        if entry.is_file() and entry.suffix == ".go":
            file_path = root.joinpath(entry.name)

            print(f"Reading {file_path}")
            docs = embed.read_source_file(file_path)

            print(docs)

            documents.extend(docs)

            if len(documents) > 50:
                return documents

        if entry.is_dir() and entry.name != "vendor":
            documents.extend(read_project(root.joinpath(entry.name)))

    return documents


def load_vector_store() -> FAISS:
    try:
        return FAISS.load_local(
            "faiss_index",
            OllamaEmbeddings(model="mxbai-embed-large"),
            allow_dangerous_deserialization=True,
        )
    except Exception:
        project_root = Path("<your project root>")
        docs = read_project(project_root)

        print(f"Embedding {len(docs)} documents...")

        vector_store = embed.create_vector_store(docs)
        vector_store.save_local("faiss_index")

        return vector_store


# TODO get parameters from a yaml file, like models, source root, etc
def main():
    vector_store = load_vector_store()

    query.query(vector_store)


if __name__ == "__main__":
    main()
