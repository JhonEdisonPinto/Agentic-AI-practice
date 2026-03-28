# Casos de inferencia con GraphDB

Este documento describe 5 casos de inferencia sobre la ontologia laboral.

## Requisito previo

En GraphDB, crea/edita el repositorio para que tenga razonador activo.
Sugerido: ruleset rdfsplus-optimized u owl-horst-optimized.

Configura en .env:

GRAPHDB_ENABLED=true
GRAPHDB_BASE_URL=http://localhost:7200
GRAPHDB_REPOSITORY=ontologia-laboral

## Carga de ontologia y ejecucion de casos

Ejecutar:

python graphdb_setup_and_demo.py

Este script:
1. Sube grafo_extenso.ttl al repositorio.
2. Ejecuta SELECT, FILTER, ORDER BY, LIMIT.
3. Ejecuta UPDATE con 2 operandos (INSERT DATA y DELETE/INSERT WHERE).
4. Inserta datos demo para inferencia.
5. Ejecuta 5 consultas de inferencia.

## Caso 1. Propiedad inversa

Axioma:
ex:tieneTrabajador owl:inverseOf ex:participaEnContrato

Dato base insertado:
ex:kg_demo_contrato ex:tieneTrabajador ex:kg_demo_trabajador .

Inferencia esperada:
ex:kg_demo_trabajador ex:participaEnContrato ex:kg_demo_contrato .

Consulta:
SELECT ?contrato WHERE {
  ex:kg_demo_trabajador ex:participaEnContrato ?contrato .
}

## Caso 2. Inferencia por rango (Trabajador)

Axioma de rango:
ex:tieneTrabajador rdfs:range ex:Trabajador

Dato base:
ex:kg_demo_contrato ex:tieneTrabajador ex:kg_demo_trabajador .

Inferencia esperada:
ex:kg_demo_trabajador a ex:Trabajador .

Consulta:
SELECT ?type WHERE {
  ex:kg_demo_trabajador a ?type .
  FILTER (?type = ex:Trabajador)
}

## Caso 3. Inferencia por subpropiedad

Axioma:
ex:tieneTrabajador rdfs:subPropertyOf ex:tieneRelacionConActor

Dato base:
ex:kg_demo_contrato ex:tieneTrabajador ex:kg_demo_trabajador .

Inferencia esperada:
ex:kg_demo_contrato ex:tieneRelacionConActor ex:kg_demo_trabajador .

Consulta:
SELECT ?actor WHERE {
  ex:kg_demo_contrato ex:tieneRelacionConActor ?actor .
}

## Caso 4. Inferencia por dominio (NormaJuridica)

Axioma de dominio:
ex:otorgaDerecho rdfs:domain ex:NormaJuridica

Dato base insertado:
ex:kg_demo_ley ex:otorgaDerecho ex:kg_demo_derecho .

Inferencia esperada:
ex:kg_demo_ley a ex:NormaJuridica .

Consulta:
SELECT ?type WHERE {
  ex:kg_demo_ley a ?type .
  FILTER (?type = ex:NormaJuridica)
}

## Caso 5. Inferencia por rango (DerechoLaboral)

Axioma de rango:
ex:otorgaDerecho rdfs:range ex:DerechoLaboral

Dato base:
ex:kg_demo_ley ex:otorgaDerecho ex:kg_demo_derecho .

Inferencia esperada:
ex:kg_demo_derecho a ex:DerechoLaboral .

Consulta:
SELECT ?type WHERE {
  ex:kg_demo_derecho a ?type .
  FILTER (?type = ex:DerechoLaboral)
}

## Evidencia de ejecucion

Usa la salida JSON de graphdb_setup_and_demo.py y capturas de GraphDB Workbench para el informe.
