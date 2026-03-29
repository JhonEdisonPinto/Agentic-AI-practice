"""
    Wrapper mínimo sobre langchain_chroma.Chroma que centraliza la configuración
 de la base vectorial. La separación en dos funciones es semántica (indexación vs. consulta);
 la implementación interna es idéntica.
    CRÍTICO: embedding_fn debe ser el mismo modelo usado durante la indexación;
 un modelo distinto produce recuperaciones silenciosamente incorrectas.
"""
import os

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


def retrieval_search_with_optional_scores(
    vectorstore,
    query: str,
    k: int,
    filter_dict: dict | None = None,
    strategy: str | None = None,
    mmr_fetch_k: int | None = None,
    mmr_lambda_mult: float | None = None,
):
    """Busca documentos usando Similarity o MMR según configuración.

    Returns:
        Lista de tuplas (Document, score). En modo MMR se devuelve score=0.0
        porque max_marginal_relevance_search no expone score de similitud.
    """
    resolved_strategy = (strategy or os.getenv("RETRIEVAL_STRATEGY", "similarity")).strip().lower()
    fetch_k = int(mmr_fetch_k or os.getenv("MMR_FETCH_K", max(k * 4, 20)))
    lambda_mult = float(mmr_lambda_mult or os.getenv("MMR_LAMBDA_MULT", 0.5))

    if resolved_strategy == "mmr":
        if filter_dict:
            docs = vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
                filter=filter_dict,
            )
        else:
            docs = vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=fetch_k,
                lambda_mult=lambda_mult,
            )
        return [(doc, 0.0) for doc in docs]

    if filter_dict:
        return vectorstore.similarity_search_with_score(query, k=k, filter=filter_dict)
    return vectorstore.similarity_search_with_score(query, k=k)

