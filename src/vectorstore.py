from langchain_community.vectorstores import Chroma


def create_chroma_index(persist_dir: str, embedding_fn, collection_name: str):
    # Creates a Chroma index; documents will be added in the RAG pipeline later.
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )


def load_chroma_index(persist_dir: str, embedding_fn, collection_name: str):
    # Loads an existing Chroma index from disk.
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )
