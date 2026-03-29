"""Metricas de generacion para evaluar respuestas en sistemas RAG.

Este modulo implementa dos metricas LLM-as-a-judge:
1) Faithfulness (fidelidad): detecta si la respuesta esta soportada por el contexto.
2) Answer Relevance (relevancia): mide si la respuesta atiende la intencion de la consulta.
"""

import json
import re
from collections.abc import Sequence
from typing import Any, Protocol

from langsmith import traceable

from src.config import init_verification_llm


class LLMJudge(Protocol):
    """Protocolo minimo para cualquier LLM usado como juez.

    El modulo solo necesita que el objeto exponga un metodo invoke(prompt)
    y retorne un objeto con atributo content o un string directo.
    """

    def invoke(self, prompt: str) -> Any:
        """Invoca el modelo con un prompt y retorna su salida."""


def _format_contexts(contexts: Sequence[str]) -> str:
    """Convierte la lista de contextos en un bloque legible para el juez.

    Args:
        contexts: Fragmentos recuperados por el sistema RAG.

    Returns:
        Texto enumerado para facilitar verificabilidad durante el juicio.
    """
    # Enumerar contexto ayuda al juez a citar y verificar afirmaciones por bloque.
    if not contexts:
        return "[Sin contexto recuperado]"

    return "\n\n".join(f"Contexto {i}:\n{ctx}" for i, ctx in enumerate(contexts, start=1))


def _extract_text_from_llm_response(response: Any) -> str:
    """Normaliza la respuesta del LLM a texto plano.

    Args:
        response: Objeto retornado por invoke().

    Returns:
        Contenido textual de la salida del modelo.
    """
    # Algunos modelos retornan string directo; otros retornan objetos AIMessage.
    if isinstance(response, str):
        return response

    # getattr permite compatibilidad sin acoplarse a una clase especifica.
    content = getattr(response, "content", "")
    if isinstance(content, str):
        return content

    # Fallback seguro para cualquier tipo no esperado.
    return str(response)


def _extract_json_block(raw_text: str) -> dict[str, Any]:
    """Extrae el primer objeto JSON valido presente en un texto.

    Args:
        raw_text: Salida completa del LLM.

    Returns:
        Diccionario parseado con score y rationale.

    Raises:
        ValueError: Si no se puede parsear JSON valido.
    """
    # Primero intentamos parseo directo para el caso ideal.
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Si el modelo agrega texto extra, buscamos el primer bloque {...}.
    match = re.search(r"\{[\s\S]*\}", raw_text)
    if not match:
        raise ValueError("No se encontro un bloque JSON en la respuesta del juez.")

    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError("No se pudo parsear el JSON del juez.") from exc

    if not isinstance(parsed, dict):
        raise ValueError("El JSON del juez no es un objeto valido.")

    return parsed


def _normalize_score(score_value: Any) -> float:
    """Normaliza el puntaje del juez al rango [0.0, 1.0].

    Args:
        score_value: Valor devuelto por el LLM en JSON.

    Returns:
        Puntaje float entre 0.0 y 1.0.
    """
    # Convertimos a float para soportar string numerico o entero.
    score = float(score_value)

    # Acotamos el rango para evitar valores invalidos por deriva del modelo.
    if score < 0.0:
        return 0.0
    if score > 1.0:
        return 1.0
    return score


def _build_faithfulness_prompt(query: str, contexts: Sequence[str], answer: str) -> str:
    """Construye prompt interno para evaluar Faithfulness.

    Args:
        query: Pregunta original del usuario.
        contexts: Contextos recuperados por el retriever.
        answer: Respuesta final generada por el agente.

    Returns:
        Prompt completo para el LLM evaluador.
    """
    # Faithfulness y Relevance se separan porque responden preguntas distintas:
    # - Faithfulness: "esta soportado por evidencia?"
    # - Relevance: "responde la intencion de la pregunta?"
    # Esta separacion permite detectar alucinacion incluso si la respuesta parece util.
    context_block = _format_contexts(contexts)

    return (
        "Eres un evaluador estricto de fidelidad para sistemas RAG.\n"
        "Tu tarea es verificar si TODAS las afirmaciones factuales de la respuesta "
        "estan respaldadas por los contextos dados.\n"
        "No evalues estilo ni redaccion. Solo evidencia vs afirmaciones.\n\n"
        "Devuelve EXCLUSIVAMENTE JSON con este formato:\n"
        "{\n"
        '  "score": 0.0-1.0,\n'
        '  "rationale": "justificacion breve en espanol"\n'
        "}\n\n"
        "Criterio de score:\n"
        "- 1.0: todas las afirmaciones estan soportadas por el contexto.\n"
        "- 0.5: parcialmente soportada; hay afirmaciones dudosas o incompletas.\n"
        "- 0.0: hay alucinaciones claras o contradicciones con el contexto.\n\n"
        f"Pregunta:\n{query}\n\n"
        f"Contextos:\n{context_block}\n\n"
        f"Respuesta:\n{answer}\n"
    )


def _build_relevance_prompt(query: str, contexts: Sequence[str], answer: str) -> str:
    """Construye prompt interno para evaluar Answer Relevance.

    Args:
        query: Pregunta original del usuario.
        contexts: Contextos recuperados por el retriever.
        answer: Respuesta final generada por el agente.

    Returns:
        Prompt completo para el LLM evaluador.
    """
    # Incluir contextos aqui permite al juez detectar respuestas genericas que
    # ignoran informacion relevante recuperada para la intencion del usuario.
    context_block = _format_contexts(contexts)

    return (
        "Eres un evaluador estricto de relevancia de respuesta para sistemas RAG.\n"
        "Tu tarea es medir si la respuesta atiende la intencion principal de la pregunta.\n"
        "No evalues veracidad factual; eso se mide en otra metrica.\n\n"
        "Devuelve EXCLUSIVAMENTE JSON con este formato:\n"
        "{\n"
        '  "score": 0.0-1.0,\n'
        '  "rationale": "justificacion breve en espanol"\n'
        "}\n\n"
        "Criterio de score:\n"
        "- 1.0: responde directamente lo solicitado y cubre la intencion principal.\n"
        "- 0.5: parcialmente relevante; responde solo una parte o con rodeos.\n"
        "- 0.0: no responde la pregunta o se desvía del objetivo.\n\n"
        f"Pregunta:\n{query}\n\n"
        f"Contextos:\n{context_block}\n\n"
        f"Respuesta:\n{answer}\n"
    )


def _judge_metric(prompt: str, llm: LLMJudge) -> dict[str, Any]:
    """Ejecuta el juicio del LLM y parsea score y justificacion.

    Args:
        prompt: Prompt construido para la metrica.
        llm: Modelo evaluador con metodo invoke.

    Returns:
        Diccionario con keys: score (float) y rationale (str).
    """
    # Invocamos el modelo evaluador para obtener un juicio estructurado.
    raw_response = llm.invoke(prompt)

    # Normalizamos a texto para parsear independientemente del proveedor.
    raw_text = _extract_text_from_llm_response(raw_response)

    # Extraemos JSON para mantener formato consistente en downstream.
    parsed = _extract_json_block(raw_text)

    # Estandarizamos salida para evitar sorpresas en consumidores.
    score = _normalize_score(parsed.get("score", 0.0))
    rationale = str(parsed.get("rationale", "Sin justificacion."))

    return {"score": score, "rationale": rationale}


@traceable(name="evaluate_faithfulness")
def evaluate_faithfulness(
    query: str,
    contexts: Sequence[str],
    answer: str,
    llm: LLMJudge | None = None,
) -> dict[str, Any]:
    """Evalua Faithfulness de una respuesta RAG.

    Args:
        query: Pregunta original del usuario.
        contexts: Fragmentos recuperados usados como evidencia.
        answer: Respuesta final del agente.
        llm: LLM evaluador opcional. Si es None, usa init_verification_llm().

    Returns:
        Diccionario con score, rationale y metadatos de la metrica.
    """
    # Mensaje explicito para trazabilidad local durante pruebas o batch runs.
    print("[Generacion] Procesando metrica: Faithfulness")

    # Si no inyectan juez, usamos el modelo de verificacion del proyecto.
    evaluator = llm or init_verification_llm(temperature=0)

    # Construimos prompt especializado para fidelidad y alucinaciones.
    prompt = _build_faithfulness_prompt(query=query, contexts=contexts, answer=answer)

    # Ejecutamos juez y parseamos score + rationale.
    result = _judge_metric(prompt=prompt, llm=evaluator)

    # Reporte de puntaje en consola para seguimiento rapido.
    print(f"[Generacion] Faithfulness score: {result['score']:.3f}")

    # Devolvemos formato enriquecido y estable para pipeline y reportes.
    return {
        "metric": "faithfulness",
        "score": result["score"],
        "rationale": result["rationale"],
    }


@traceable(name="evaluate_answer_relevance")
def evaluate_answer_relevance(
    query: str,
    contexts: Sequence[str],
    answer: str,
    llm: LLMJudge | None = None,
) -> dict[str, Any]:
    """Evalua Answer Relevance de una respuesta RAG.

    Args:
        query: Pregunta original del usuario.
        contexts: Fragmentos recuperados (contexto de referencia).
        answer: Respuesta final del agente.
        llm: LLM evaluador opcional. Si es None, usa init_verification_llm().

    Returns:
        Diccionario con score, rationale y metadatos de la metrica.
    """
    # Mensaje explicito para trazabilidad local durante pruebas o batch runs.
    print("[Generacion] Procesando metrica: Answer Relevance")

    # Reutilizamos la inicializacion central para consistencia de entorno.
    evaluator = llm or init_verification_llm(temperature=0)

    # Construimos prompt especializado para cobertura de intencion de consulta.
    prompt = _build_relevance_prompt(query=query, contexts=contexts, answer=answer)

    # Ejecutamos juicio del LLM y normalizamos la salida.
    result = _judge_metric(prompt=prompt, llm=evaluator)

    # Reporte de puntaje en consola para inspeccion operacional.
    print(f"[Generacion] Answer Relevance score: {result['score']:.3f}")

    # Devolvemos el resultado en un formato facil de serializar y comparar.
    return {
        "metric": "answer_relevance",
        "score": result["score"],
        "rationale": result["rationale"],
    }


@traceable(name="evaluate_generation_metrics")
def evaluate_generation_metrics(
    query: str,
    contexts: Sequence[str],
    answer: str,
    llm: LLMJudge | None = None,
) -> dict[str, Any]:
    """Evalua en conjunto Faithfulness y Answer Relevance.

    Args:
        query: Pregunta original del usuario.
        contexts: Contextos recuperados por la fase de retrieval.
        answer: Respuesta final entregada por el agente.
        llm: LLM evaluador opcional, compartido entre metricas.

    Returns:
        Diccionario con resultados de ambas metricas y score promedio.
    """
    # Trazabilidad de inicio para lotes de evaluacion.
    print("[Generacion] Iniciando evaluacion conjunta de metricas")

    # Reusar la misma instancia del juez evita variabilidad innecesaria.
    evaluator = llm or init_verification_llm(temperature=0)

    # Evaluamos fidelidad para detectar alucinacion respecto al contexto.
    faithfulness = evaluate_faithfulness(
        query=query,
        contexts=contexts,
        answer=answer,
        llm=evaluator,
    )

    # Evaluamos relevancia para medir alineacion con la intencion del usuario.
    relevance = evaluate_answer_relevance(
        query=query,
        contexts=contexts,
        answer=answer,
        llm=evaluator,
    )

    # Promedio simple util para dashboard rapido; se preservan metricas separadas.
    average_score = (faithfulness["score"] + relevance["score"]) / 2

    # Log final de resultados para corrida local o CI.
    print(f"[Generacion] Score final promedio: {average_score:.3f}")

    return {
        "faithfulness": faithfulness,
        "answer_relevance": relevance,
        "average_score": average_score,
    }
