"""Pruebas para métricas de recuperación (Recall@k y Precision@k)."""

import sys
from pathlib import Path

# Asegura que `src` sea importable al ejecutar `pytest` desde la raíz.
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.retrieval_metrics import (
    evaluate_query_at_ks,
    evaluate_retrieval_dataset,
    precision_at_k,
    recall_at_k,
)


def test_recall_and_precision_single_query() -> None:
    """Valida métricas @k para una consulta sencilla."""
    retrieved = ["D1", "D2", "D3", "D4", "D5"]
    relevant = ["D2", "D4", "D6"]

    # top-3 = [D1, D2, D3] -> 1 relevante (D2)
    assert recall_at_k(retrieved, relevant, 3) == 1 / 3
    assert precision_at_k(retrieved, relevant, 3) == 1 / 3

    # top-5 = [D1, D2, D3, D4, D5] -> 2 relevantes (D2, D4)
    assert recall_at_k(retrieved, relevant, 5) == 2 / 3
    assert precision_at_k(retrieved, relevant, 5) == 2 / 5


def test_evaluate_query_at_multiple_ks() -> None:
    """Valida salida estructurada para múltiples valores de k."""
    retrieved = ["A", "B", "C", "D"]
    relevant = ["B", "C", "X"]

    results = evaluate_query_at_ks(retrieved, relevant, ks=(1, 2, 3))

    assert set(results.keys()) == {1, 2, 3}
    assert results[1]["recall"] == 0.0
    assert results[1]["precision"] == 0.0
    assert results[2]["recall"] == 1 / 3
    assert results[2]["precision"] == 1 / 2
    assert results[3]["recall"] == 2 / 3
    assert results[3]["precision"] == 2 / 3


def test_dataset_average_metrics() -> None:
    """Valida promedios por k para múltiples consultas."""
    retrieved_per_query = [
        ["D1", "D2", "D3", "D4"],
        ["A1", "A2", "A3", "A4"],
    ]
    relevant_per_query = [
        ["D2", "D4"],
        ["A3", "A9"],
    ]

    results = evaluate_retrieval_dataset(
        retrieved_ids_per_query=retrieved_per_query,
        relevant_ids_per_query=relevant_per_query,
        ks=(2, 4),
    )

    # Consulta 1: k=2 -> recall=1/2, precision=1/2 ; k=4 -> recall=1, precision=2/4
    # Consulta 2: k=2 -> recall=0, precision=0 ; k=4 -> recall=1/2, precision=1/4
    # Promedios:  k=2 -> recall=1/4, precision=1/4
    #             k=4 -> recall=3/4, precision=3/8
    assert results[2]["avg_recall"] == 0.25
    assert results[2]["avg_precision"] == 0.25
    assert results[4]["avg_recall"] == 0.75
    assert results[4]["avg_precision"] == 0.375
