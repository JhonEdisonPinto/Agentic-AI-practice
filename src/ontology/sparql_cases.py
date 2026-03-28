"""Casos SPARQL requeridos para GraphDB usando RDFLib."""

from __future__ import annotations

from typing import Any, Dict, List

from .graphdb_client import GraphDBClient

PREFIXES = """
PREFIX ex: <http://example.org/ontologia-laboral#>
PREFIX owl: <http://www.w3.org/2002/07/owl#>
PREFIX rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX xsd: <http://www.w3.org/2001/XMLSchema#>
"""


def case_select() -> str:
    return PREFIXES + """
SELECT ?trabajador ?nombre
WHERE {
  ?trabajador a ex:Trabajador ;
              ex:tieneNombre ?nombre .
}
"""


def case_filter() -> str:
    return PREFIXES + """
SELECT ?contrato ?salario
WHERE {
  ?contrato a ex:ContratoLaboral ;
            ex:salarioMensual ?salario .
  FILTER (?salario >= 2500000)
}
"""


def case_order_by() -> str:
    return PREFIXES + """
SELECT ?contrato ?salario
WHERE {
  ?contrato a ex:ContratoLaboral ;
            ex:salarioMensual ?salario .
}
ORDER BY DESC(?salario)
"""


def case_limit() -> str:
    return PREFIXES + """
SELECT ?norma ?nombre ?anio
WHERE {
  ?norma a ex:NormaJuridica ;
         ex:tieneNombre ?nombre ;
         ex:tieneAnioPublicacion ?anio .
}
ORDER BY DESC(?anio)
LIMIT 3
"""


def case_update_insert_data() -> str:
    return PREFIXES + """
INSERT DATA {
  ex:obligacion_capacitacion a ex:ObligacionLaboral ;
    ex:tieneNombre "Capacitar al personal en SST" .
}
"""


def case_update_delete_insert_where() -> str:
    return PREFIXES + """
DELETE {
  ex:trabajador_mario ex:tieneNombre ?oldName .
}
INSERT {
  ex:trabajador_mario ex:tieneNombre "Mario Perez Actualizado" .
}
WHERE {
  ex:trabajador_mario ex:tieneNombre ?oldName .
}
"""


def case_seed_inference_data() -> str:
    """Inserta datos minimos para demostrar inferencia en GraphDB."""
    return PREFIXES + """
INSERT DATA {
  ex:kg_demo_contrato ex:tieneTrabajador ex:kg_demo_trabajador .
  ex:kg_demo_ley ex:otorgaDerecho ex:kg_demo_derecho .
}
"""


def inference_case_inverse_property() -> str:
    return PREFIXES + """
SELECT ?contrato
WHERE {
  ex:kg_demo_trabajador ex:participaEnContrato ?contrato .
}
"""


def inference_case_range_worker() -> str:
    return PREFIXES + """
SELECT ?type
WHERE {
  ex:kg_demo_trabajador a ?type .
  FILTER (?type = ex:Trabajador)
}
"""


def inference_case_subproperty() -> str:
    return PREFIXES + """
SELECT ?actor
WHERE {
  ex:kg_demo_contrato ex:tieneRelacionConActor ?actor .
}
"""


def inference_case_domain_norma() -> str:
    return PREFIXES + """
SELECT ?type
WHERE {
  ex:kg_demo_ley a ?type .
  FILTER (?type = ex:NormaJuridica)
}
"""


def inference_case_range_derecho() -> str:
    return PREFIXES + """
SELECT ?type
WHERE {
  ex:kg_demo_derecho a ?type .
  FILTER (?type = ex:DerechoLaboral)
}
"""


def run_required_query_cases(client: GraphDBClient) -> Dict[str, List[Dict[str, Any]]]:
    """Ejecuta casos obligatorios SELECT, FILTER, ORDER BY y LIMIT."""
    return {
        "select": client.select(case_select()),
        "filter": client.select(case_filter()),
        "order_by": client.select(case_order_by()),
        "limit": client.select(case_limit()),
    }


def run_required_update_cases(client: GraphDBClient) -> None:
    """Ejecuta 2 operandos UPDATE: INSERT DATA y DELETE/INSERT WHERE."""
    client.update(case_update_insert_data())
    client.update(case_update_delete_insert_where())


def run_inference_cases(client: GraphDBClient) -> Dict[str, List[Dict[str, Any]]]:
    """Inserta datos demo y consulta 5 inferencias."""
    client.update(case_seed_inference_data())
    return {
        "inverse_property": client.select(inference_case_inverse_property()),
        "range_worker": client.select(inference_case_range_worker()),
        "subproperty": client.select(inference_case_subproperty()),
        "domain_norma": client.select(inference_case_domain_norma()),
        "range_derecho": client.select(inference_case_range_derecho()),
    }
