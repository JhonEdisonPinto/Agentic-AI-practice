"""Agente ligero de Knowledge Graph para retrieval en GraphDB."""

from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.documents import Document

from .graphdb_client import GraphDBClient
from .sparql_cases import PREFIXES


class KGAgent:
    """Agente que convierte consulta de usuario en retrieval SPARQL pragmatica."""

    def __init__(self, client: GraphDBClient | None = None) -> None:
        self.client = client or GraphDBClient()

    def _select_query(self, user_query: str) -> str:
        q = user_query.lower()

        if "derecho" in q:
            return PREFIXES + """
SELECT ?derecho ?nombre ?norma
WHERE {
  ?derecho a ex:DerechoLaboral ;
           ex:tieneNombre ?nombre .
  OPTIONAL { ?derecho ex:esOtorgadoPor ?norma . }
}
LIMIT 8
"""

        if "oblig" in q:
            return PREFIXES + """
SELECT ?obligacion ?nombre
WHERE {
  ?obligacion a ex:ObligacionLaboral ;
              ex:tieneNombre ?nombre .
}
LIMIT 8
"""

        if "contrato" in q or "salario" in q:
            return PREFIXES + """
SELECT ?contrato ?trabajador ?salario
WHERE {
  ?contrato a ex:ContratoLaboral ;
            ex:tieneTrabajador ?trabajador ;
            ex:salarioMensual ?salario .
}
ORDER BY DESC(?salario)
LIMIT 8
"""

        if "ley" in q or "norma" in q:
            return PREFIXES + """
SELECT ?norma ?nombre ?anio
WHERE {
  ?norma a ex:NormaJuridica ;
         ex:tieneNombre ?nombre ;
         ex:tieneAnioPublicacion ?anio .
}
ORDER BY DESC(?anio)
LIMIT 8
"""

        return PREFIXES + """
SELECT ?recurso ?nombre
WHERE {
  ?recurso ex:tieneNombre ?nombre .
}
LIMIT 8
"""

    def retrieve(self, user_query: str) -> Dict[str, Any]:
        if not self.client.enabled:
            return {
                "enabled": False,
                "rows": [],
                "sparql": None,
                "error": "GraphDB deshabilitado por GRAPHDB_ENABLED=false",
            }

        try:
            sparql = self._select_query(user_query)
            rows = self.client.select(sparql)
            return {
                "enabled": True,
                "rows": rows,
                "sparql": sparql,
                "error": None,
            }
        except Exception as exc:
            return {
                "enabled": True,
                "rows": [],
                "sparql": None,
                "error": str(exc),
            }

    def as_documents(self, user_query: str) -> List[Document]:
        result = self.retrieve(user_query)
        rows = result.get("rows", [])
        docs: List[Document] = []

        for i, row in enumerate(rows, 1):
            text_parts = [f"{k}: {v}" for k, v in row.items() if v is not None]
            docs.append(
                Document(
                    page_content="\n".join(text_parts),
                    metadata={
                        "source": "graphdb",
                        "id_documento": f"GRAPHDB_ROW_{i}",
                        "tipo_documento": "TRIPLE_STORE",
                        "sparql": result.get("sparql"),
                    },
                )
            )

        return docs
