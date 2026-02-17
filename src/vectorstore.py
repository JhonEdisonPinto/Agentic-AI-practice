from langchain_chroma import Chroma


def create_chroma_index(persist_dir: str, embedding_fn, collection_name: str):
    """
    Crea o conecta a un índice de Chroma existente.
    Si ya existe, agrega documentos a la colección existente.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )


def load_chroma_index(persist_dir: str, embedding_fn, collection_name: str):
    """
    Carga un índice existente de Chroma desde disco.
    """
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )

