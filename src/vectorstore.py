"""
    Wrapper mínimo sobre langchain_chroma.Chroma que centraliza la configuración
 de la base vectorial. La separación en dos funciones es semántica (indexación vs. consulta);
 la implementación interna es idéntica.
    CRÍTICO: embedding_fn debe ser el mismo modelo usado durante la indexación;
 un modelo distinto produce recuperaciones silenciosamente incorrectas.
"""
from langchain_chroma import Chroma


def create_chroma_index(persist_dir: str, embedding_fn, collection_name: str):
    """
    Crea o conecta a un índice de Chroma existente.
    Si ya existe, agrega documentos a la colección existente.
    """
    # Si persist_dir ya contiene una colección con collection_name, Chroma conecta
    # a la existente en lugar de crear una nueva, permitiendo indexación incremental.
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )


def load_chroma_index(persist_dir: str, embedding_fn, collection_name: str):
    """
    Carga un índice existente de Chroma desde disco.
    """
    # Carga la colección desde disco sin validar que persist_dir exista o contenga datos.
    # Si la colección está vacía o ausente, retorna una instancia válida pero sin documentos,
    # lo que produce resultados vacíos en similarity_search sin lanzar excepción.
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_dir,
    )

