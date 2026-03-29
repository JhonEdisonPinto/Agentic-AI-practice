"""Utilidades para evaluar recuperación en sistemas RAG.

Este módulo implementa Recall@k y Precision@k para una o múltiples
consultas, con soporte para evaluar distintos valores de k en una sola
ejecución.
"""

from __future__ import annotations # Para compatibilidad con Python 3.7+ y anotaciones de tipo más limpias.

from collections.abc import Sequence


def _validate_k(k: int) -> None:
    """Valida que k sea un entero positivo.

    Args:
        k: Número de documentos del top-k a evaluar.

    Raises:
        ValueError: Si k no es mayor que cero.
    """
    # La métrica @k no tiene sentido para k <= 0 porque no existiría top-k.
    if k <= 0:
        raise ValueError("k debe ser mayor que 0.")


def recall_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float:
    """Calcula Recall@k para una sola consulta.

    Args:
        retrieved_doc_ids: IDs recuperados por el sistema de búsqueda.
        relevant_doc_ids: IDs relevantes reales (ground truth).
        k: Profundidad de corte para top-k.

    Returns:
        Valor de Recall@k en el rango [0, 1].
    """
    # Validamos k para evitar resultados ambiguos o divisiones incorrectas.
    _validate_k(k)

    # Truncamos la lista al top-k porque la definición de la métrica exige
    # evaluar únicamente los primeros k resultados recuperados.
    top_k_retrieved = list(retrieved_doc_ids[:k])

    # Convertimos a set para medir intersección eficiente y evitar que IDs
    # repetidos inflen artificialmente el conteo de aciertos.
    top_k_set = set(top_k_retrieved)
    relevant_set = set(relevant_doc_ids)

    # Si no hay ground truth, por convención devolvemos 0.0 para no dividir
    # por cero y porque no se puede recuperar lo que no está definido.
    if not relevant_set:
        return 0.0

    # Relevantes recuperados correctamente dentro del top-k.
    true_positives = len(top_k_set.intersection(relevant_set))

    # Recall@k = relevantes recuperados en top-k / total relevantes reales.
    return true_positives / len(relevant_set)


def precision_at_k(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    k: int,
) -> float:
    """Calcula Precision@k para una sola consulta.

    Args:
        retrieved_doc_ids: IDs recuperados por el sistema de búsqueda.
        relevant_doc_ids: IDs relevantes reales (ground truth).
        k: Profundidad de corte para top-k.

    Returns:
        Valor de Precision@k en el rango [0, 1].
    """
    # Validamos k para garantizar que el denominador de la métrica sea válido.
    _validate_k(k)

    # Aplicamos corte top-k para alinear el cálculo con la definición @k.
    top_k_retrieved = list(retrieved_doc_ids[:k])

    # Conjuntos para intersección y robustez ante IDs duplicados.
    top_k_set = set(top_k_retrieved)
    relevant_set = set(relevant_doc_ids)

    # Aciertos relevantes dentro del top-k.
    true_positives = len(top_k_set.intersection(relevant_set))

    # Usamos k como denominador (definición clásica de Precision@k),
    # penalizando listas cortas cuando se recuperan menos de k resultados.
    return true_positives / k


def evaluate_query_at_ks(
    retrieved_doc_ids: Sequence[str],
    relevant_doc_ids: Sequence[str],
    ks: Sequence[int],
) -> dict[int, dict[str, float]]:
    """Evalúa una consulta para múltiples valores de k.

    Args:
        retrieved_doc_ids: IDs recuperados para una consulta.
        relevant_doc_ids: IDs relevantes reales para la consulta.
        ks: Lista o tupla de valores k a evaluar.

    Returns:
        Diccionario por k con `recall` y `precision`.
    """
    # Contenedor de resultados por cada k.
    metrics_by_k: dict[int, dict[str, float]] = {}

    # Recorremos ks para reutilizar la misma consulta y obtener una curva
    # rápida de calidad según profundidad de recuperación.
    for k in ks:
        # Se imprime trazabilidad para saber exactamente qué k se está midiendo.
        print(f"[Métricas] Evaluando consulta individual con k={k}...")
        metrics_by_k[k] = {
            "recall": recall_at_k(retrieved_doc_ids, relevant_doc_ids, k),
            "precision": precision_at_k(retrieved_doc_ids, relevant_doc_ids, k),
        }

    return metrics_by_k


def evaluate_retrieval_dataset(
    retrieved_ids_per_query: Sequence[Sequence[str]],
    relevant_ids_per_query: Sequence[Sequence[str]],
    ks: Sequence[int] = (3, 5, 10),
) -> dict[int, dict[str, float | int]]:
    """Evalúa Recall@k y Precision@k promedio para múltiples consultas.

    Args:
        retrieved_ids_per_query: Lista donde cada elemento contiene los IDs
            recuperados para una consulta.
        relevant_ids_per_query: Lista donde cada elemento contiene los IDs
            relevantes reales para la consulta correspondiente.
        ks: Valores de k a evaluar (por defecto 3, 5 y 10).

    Returns:
        Diccionario con promedios por cada k. Formato:
            {
                k: {
                    "avg_recall": float,
                    "avg_precision": float,
                    "num_queries": int,
                }
            }

    Raises:
        ValueError: Si las listas de consultas no tienen la misma longitud.
    """
    # Las listas deben estar alineadas por índice para comparar consulta a consulta.
    if len(retrieved_ids_per_query) != len(relevant_ids_per_query):
        raise ValueError(
            "retrieved_ids_per_query y relevant_ids_per_query deben tener la misma longitud."
        )

    # Si no hay consultas, devolvemos estructura vacía para evitar promedios inválidos.
    if not retrieved_ids_per_query:
        return {}

    # Inicializamos acumuladores para sumar métricas antes de promediar.
    totals: dict[int, dict[str, float]] = {
        k: {"recall": 0.0, "precision": 0.0} for k in ks
    }

    # Número total de consultas evaluadas para normalizar al final.
    num_queries = len(retrieved_ids_per_query)

    # Iteramos en paralelo para mantener correspondencia exacta por consulta.
    for idx, (retrieved_ids, relevant_ids) in enumerate(
        zip(retrieved_ids_per_query, relevant_ids_per_query), start=1
    ):
        # Mensaje de progreso para trazabilidad operativa en consola.
        print(f"[Métricas] Procesando consulta {idx}/{num_queries}...")

        # Calculamos métricas por consulta y por cada k.
        per_query_metrics = evaluate_query_at_ks(
            retrieved_doc_ids=retrieved_ids,
            relevant_doc_ids=relevant_ids,
            ks=ks,
        )

        # Acumulamos para luego obtener promedios globales por k.
        for k in ks:
            totals[k]["recall"] += per_query_metrics[k]["recall"]
            totals[k]["precision"] += per_query_metrics[k]["precision"]

    # Construimos resultado final con promedios por valor de k.
    averaged_metrics: dict[int, dict[str, float | int]] = {}
    for k in ks:
        averaged_metrics[k] = {
            "avg_recall": totals[k]["recall"] / num_queries,
            "avg_precision": totals[k]["precision"] / num_queries,
            "num_queries": num_queries,
        }

    return averaged_metrics
