"""Pruebas unitarias para metricas de generacion (LLM-as-a-judge)."""

import sys
from pathlib import Path

# Asegura que src sea importable al ejecutar pytest desde la raiz.
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.generation_metrics import (  # noqa: E402
    evaluate_answer_relevance,
    evaluate_faithfulness,
    evaluate_generation_metrics,
)


class _FakeLLM:
    """Doble de prueba para simular respuestas del juez LLM."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = responses
        self._idx = 0

    def invoke(self, prompt: str) -> str:
        _ = prompt
        if self._idx >= len(self._responses):
            return '{"score": 0.0, "rationale": "sin respuesta mock"}'
        response = self._responses[self._idx]
        self._idx += 1
        return response


def test_evaluate_faithfulness_parses_json() -> None:
    """Valida parseo de score y rationale para metrica de fidelidad."""
    llm = _FakeLLM(['{"score": 0.9, "rationale": "Todo soportado."}'])

    result = evaluate_faithfulness(
        query="Que exige la norma sobre vacaciones?",
        contexts=["La norma indica 15 dias habiles de vacaciones al anio."],
        answer="La norma exige 15 dias habiles de vacaciones al anio.",
        llm=llm,
    )

    assert result["metric"] == "faithfulness"
    assert result["score"] == 0.9
    assert "soportado" in result["rationale"].lower()


def test_evaluate_answer_relevance_parses_embedded_json() -> None:
    """Valida parseo cuando el juez agrega texto alrededor del JSON."""
    llm = _FakeLLM([
        'Analisis:\n{"score": 0.7, "rationale": "Responde parcialmente."}\nFin'
    ])

    result = evaluate_answer_relevance(
        query="Como calcular cesantias?",
        contexts=["Contexto legal sobre liquidacion."],
        answer="Se explican algunos pasos de calculo.",
        llm=llm,
    )

    assert result["metric"] == "answer_relevance"
    assert result["score"] == 0.7
    assert "parcial" in result["rationale"].lower()


def test_evaluate_generation_metrics_returns_average() -> None:
    """Valida evaluacion conjunta y score promedio final."""
    llm = _FakeLLM([
        '{"score": 0.8, "rationale": "Sin alucinaciones."}',
        '{"score": 0.6, "rationale": "Responde la idea principal."}',
    ])

    result = evaluate_generation_metrics(
        query="Que obligaciones tiene el empleador?",
        contexts=["Contexto con deberes del empleador."],
        answer="Describe obligaciones basicas del empleador.",
        llm=llm,
    )

    assert result["faithfulness"]["score"] == 0.8
    assert result["answer_relevance"]["score"] == 0.6
    assert result["average_score"] == 0.7
