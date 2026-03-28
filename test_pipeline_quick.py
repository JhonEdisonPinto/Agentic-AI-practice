import sys
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

from src.config import load_settings
from src.graph import build_graph

print("\n" + "="*70)
print("  EJECUTANDO PRUEBA DEL PIPELINE COMPLETO")
print("="*70 + "\n")

# Cargar config y construir grafo
print("[1/3] Cargando configuracion...")
load_settings()
print("    LISTO\n")

print("[2/3] Construyendo workflow...")
graph = build_graph()
print("    LISTO\n")

# Prueba simple: una query
print("[3/3] Ejecutando query de prueba...")
query = "¿Cuales son los derechos de los trabajadores?"

print(f"\nQuery: {query}\n")
print("-" * 70)

result = graph.invoke({
    "query": query,
    "classification": None,
    "query_type": None,
    "transformed_queries": None,
    "documents": [],
    "tool_results": None,
    "kg_results": None,
    "answer": "",
    "verification": {},
    "metadata": {}
})

print("\nRESULTADOS:")
print(f"  Clasificacion: {result.get('classification')}")
print(f"  Documentos recuperados: {len(result.get('documents', []))}")
print(f"  GraphDB habilitado: {result.get('metadata', {}).get('kg_enabled')}")
print(f"  Filas de GraphDB: {result.get('metadata', {}).get('kg_rows', 0)}")
print(f"  Respuesta:\n{result.get('answer', 'N/A')[:500]}")
print(f"\n  Verificacion score: {result.get('verification', {}).get('score', 'N/A')}")
print("\n" + "="*70)
print("  TEST COMPLETADO EXITOSAMENTE")
print("="*70 + "\n")
