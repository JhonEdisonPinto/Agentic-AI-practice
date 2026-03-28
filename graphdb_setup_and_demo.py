"""Script de setup para GraphDB + demostracion de consultas e inferencia.

Uso:
1) Levantar GraphDB y crear repositorio.
2) Configurar variables en .env:
   GRAPHDB_ENABLED=true
   GRAPHDB_BASE_URL=http://localhost:7200
   GRAPHDB_REPOSITORY=ontologia-laboral
3) Ejecutar:
   python graphdb_setup_and_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

from src.config import load_settings
from src.ontology.graphdb_client import GraphDBClient
from src.ontology.sparql_cases import (
    run_inference_cases,
    run_required_query_cases,
    run_required_update_cases,
)


def main() -> None:
    load_settings()
    client = GraphDBClient()

    print("=" * 72)
    print("SETUP GRAPHDB + RDFLIB")
    print("=" * 72)
    print(f"Auth mode: {client.auth_mode}")
    print(f"Repositorio: {client.repository_id}")
    print(f"Endpoint: {client.query_endpoint}")

    if not client.enabled:
        raise RuntimeError("GRAPHDB_ENABLED=false. Activalo en .env para ejecutar.")

    if client.auth_mode == "token":
        print("Obteniendo token GDB via /rest/login...")
        client.obtain_token()
        print("OK token obtenido")

    if not client.ping():
        raise RuntimeError(
            "No hay conexion con GraphDB. Verifica endpoint/repositorio/autenticacion. "
            "Si usas sandbox, define GRAPHDB_BASE_URL con el host sandbox y GRAPHDB_LOGIN_URL=/rest/login."
        )

    ttl_path = Path("ontologia_practica2.ttl")
    print(f"Subiendo ontologia: {ttl_path}")
    client.upload_ttl_file(ttl_path)
    print("OK ontologia cargada")

    print("\nEjecutando casos SPARQL requeridos...")
    query_results = run_required_query_cases(client)
    print(json.dumps(query_results, indent=2, ensure_ascii=False)[:3000])

    print("\nEjecutando casos UPDATE requeridos...")
    run_required_update_cases(client)
    print("OK updates ejecutados")

    print("\nEjecutando 5 casos de inferencia...")
    inference_results = run_inference_cases(client)
    print(json.dumps(inference_results, indent=2, ensure_ascii=False)[:3000])

    print("\nProceso completado")


if __name__ == "__main__":
    main()
