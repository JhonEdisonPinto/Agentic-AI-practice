"""
Tools para el RAG de normativa laboral colombiana.
Estas herramientas extienden las capacidades del sistema RAG.
"""
from typing import List, Dict, Optional, Any
from langchain_core.tools import tool
from langchain_core.documents import Document
import re
import os
from src.config import init_groq_llm


# -----------------------------------------------------------------------------
# ROUTING TOOLS (SCHEMA-ONLY)
# -----------------------------------------------------------------------------
# Estas tools NO ejecutan lógica de negocio ni acceden a vectorstore.
# Su única responsabilidad es exponer un contrato limpio para `llm.bind_tools(...)`
# durante el routing en `tool_calling_node`.
#
# Diferencia clave frente a las tools de ejecución de este mismo archivo:
# - routing_*: interfaz pública para el LLM (parámetros de negocio serializables).
# - tools de ejecución: lógica real (búsqueda, cálculo, extracción), con dependencias
#   internas inyectadas por el runtime (por ejemplo vectorstore).
#
# Beneficio: se evita que el modelo vea parámetros internos (ej. vectorstore),
# lo que reduce errores de validación y acoplamiento con el proveedor del LLM.


@tool("resume_document")
def routing_resume_document(
    doc_type: str,
    doc_number: str,
    doc_year: int | None = None,
    doc_id: str | None = None,
) -> str:
    """Schema de routing para solicitar resumen de documento."""
    return "routing_only"


@tool("calculate_prestaciones_sociales")
def routing_calculate_prestaciones_sociales(
    salario_detectado: float | None = None,
    dias_trabajados: int | None = None,
    años_servicio: float | None = None,
) -> str:
    """Schema de routing para solicitar cálculo de prestaciones."""
    return "routing_only"


@tool("extract_specific_article")
def routing_extract_specific_article(
    article_number: str,
    doc_type: str,
    doc_number: str,
    doc_year: int | None = None,
    doc_id: str | None = None,
) -> str:
    """Schema de routing para extracción de artículo específico."""
    return "routing_only"


@tool("compare_documents")
def routing_compare_documents(doc_id1: str, doc_id2: str, topic: str) -> str:
    """Schema de routing para comparación entre documentos."""
    return "routing_only"


@tool("search_by_document_type")
def routing_search_by_document_type(
    doc_type: str,
    doc_number: str,
    doc_year: int | None = None,
) -> str:
    """Schema de routing para búsqueda por tipo y número."""
    return "routing_only"


@tool("search_by_year_range")
def routing_search_by_year_range(start_year: int, end_year: int) -> str:
    """Schema de routing para búsqueda por rango de años."""
    return "routing_only"


# Lista usada por graph.py para bind_tools en la fase de routing.
ROUTING_TOOLS = [
    routing_resume_document,
    routing_calculate_prestaciones_sociales,
    routing_extract_specific_article,
    routing_compare_documents,
    routing_search_by_document_type,
    routing_search_by_year_range,
]


def _retrieve_docs(
    vectorstore: Any,
    query: str,
    k: int,
    filter_dict: Optional[Dict[str, Any]] = None,
) -> List[Document]:
    """Recupera documentos con Similarity o MMR según RETRIEVAL_STRATEGY."""
    strategy = os.getenv("RETRIEVAL_STRATEGY", "similarity").strip().lower()
    mmr_fetch_k = int(os.getenv("MMR_FETCH_K", max(k * 4, 20)))
    mmr_lambda_mult = float(os.getenv("MMR_LAMBDA_MULT", 0.5))

    if strategy == "mmr":
        if filter_dict:
            return vectorstore.max_marginal_relevance_search(
                query,
                k=k,
                fetch_k=mmr_fetch_k,
                lambda_mult=mmr_lambda_mult,
                filter=filter_dict,
            )
        return vectorstore.max_marginal_relevance_search(
            query,
            k=k,
            fetch_k=mmr_fetch_k,
            lambda_mult=mmr_lambda_mult,
        )

    if filter_dict:
        return vectorstore.similarity_search(query, k=k, filter=filter_dict)
    return vectorstore.similarity_search(query, k=k)


# -----------------------------------------------------------------------------
# EXECUTION TOOLS (LÓGICA REAL)
# -----------------------------------------------------------------------------


@tool
def search_by_document_type(query: str, doc_type: str, vectorstore: Any = None) -> List[Document]:
    """
    Busca documentos por tipo específico (LEY, DECRETO, SENTENCIA).
    
    Args:
        query: Consulta de búsqueda
        doc_type: Tipo de documento (LEY, DECRETO, SENTENCIA)
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Lista de documentos del tipo especificado
    """

    if vectorstore is None:
        raise ValueError("vectorstore is required for search_by_document_type")

    # Recupera hasta 10 fragmentos por similaridad y filtra los primeros 5 que coincidan
    # con el tipo de documento indicado. El filtrado es post-búsqueda (no usa filtros de ChromaDB),
    # lo que implica que k=10 puede no ser suficiente si el tipo buscado es poco frecuente.
    results = vectorstore.similarity_search(query, k=10)
    
    # Filtrar por tipo de documento
    filtered = [
        doc for doc in results 
        if doc.metadata.get("tipo_documento", "").upper() == doc_type.upper()
    ]
    
    return filtered[:5]


@tool
def search_by_year_range(query: str, start_year: int, end_year: int, vectorstore: Any = None) -> List[Document]:
    """
    Busca documentos dentro de un rango de años.
    
    Args:
        query: Consulta de búsqueda
        start_year: Año inicial
        end_year: Año final
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Lista de documentos en el rango de años
    """

    if vectorstore is None:
        raise ValueError("vectorstore is required for search_by_year_range")

    # Similar a search_by_document_type pero filtra por rango de años.
    # Recupera hasta 20 fragmentos para compensar que el filtrado es post-búsqueda.
    # Los valores de metadata "año" se convierten a int; entradas no numéricas se descartan silenciosamente.
    results = vectorstore.similarity_search(query, k=20)
    
    # Filtrar por rango de años
    filtered = []
    for doc in results:
        año_str = doc.metadata.get("año", "")
        if año_str:
            try:
                año = int(año_str)
                if start_year <= año <= end_year:
                    filtered.append(doc)
            except ValueError:
                continue
    
    return filtered[:5]


@tool
def calculate_prestaciones_sociales(
    salario_mensual: float,
    dias_trabajados: int,
    años_servicio: float
) -> Dict[str, float]:
    """
    Calcula las prestaciones sociales de un trabajador colombiano.
    
    Args:
        salario_mensual: Salario mensual del trabajador
        dias_trabajados: Días trabajados en el período
        años_servicio: Años de servicio (puede incluir decimales)
        
    Returns:
        Diccionario con el desglose de prestaciones
    """
    # Calcula cesantías, intereses (12% anual), prima de servicios y vacaciones
    # conforme al Código Sustantivo del Trabajo colombiano.
    # Usa año comercial de 360 días. El parámetro años_servicio se incluye en el resultado
    # pero no interviene en ningún cálculo; toda la base es dias_trabajados/360.
    
    # Cálculo de cesantías (1 mes de salario por año)
    cesantias = (salario_mensual * dias_trabajados) / 360
    
    # Intereses sobre cesantías (12% anual)
    intereses_cesantias = cesantias * 0.12
    
    # Prima de servicios (1 mes de salario por año, pagada en 2 cuotas)
    prima_servicios = (salario_mensual * dias_trabajados) / 360
    
    # Vacaciones (15 días hábiles por año)
    dias_vacaciones = (dias_trabajados / 360) * 15
    vacaciones = (salario_mensual / 30) * dias_vacaciones
    
    total = cesantias + intereses_cesantias + prima_servicios + vacaciones
    
    return {
        "cesantias": round(cesantias, 2),
        "intereses_cesantias": round(intereses_cesantias, 2),
        "prima_servicios": round(prima_servicios, 2),
        "vacaciones": round(vacaciones, 2),
        "total_prestaciones": round(total, 2),
        "dias_trabajados": dias_trabajados,
        "años_servicio": round(años_servicio, 2)
    }


@tool
def extract_specific_article(doc_id: str, article_number: str, vectorstore: Any = None) -> Optional[str]:
    """
    Extrae un artículo específico de un documento legal.
    
    Args:
        doc_id: ID del documento (ej: "LEY_1010_2006" o "LEY_1010")
        article_number: Número del artículo a extraer
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Contenido del artículo o None si no se encuentra
    """

    if vectorstore is None:
        raise ValueError("vectorstore is required for extract_specific_article")

    # Localiza el chunk que contiene un artículo específico dentro de un documento legal.
    # Estrategia en tres niveles de estrictez:
    #   1. Metadata completa (tipo + número + año) con coincidencia de patrón textual.
    #   2. Solo número de documento (ignora tipo y año) con coincidencia de patrón.
    #   3. Sin filtro de metadata: acepta cualquier chunk que mencione el artículo
    #      y contenga el número o tipo del documento en su texto.
    # El tercer nivel es un fallback de último recurso ante metadata inconsistente.
    
    # Parsear el document ID
    parts = doc_id.split("_")
    if len(parts) < 2:
        return None
    
    doc_type = parts[0]
    doc_number = parts[1]
    doc_year = parts[2] if len(parts) >= 3 else None
    
    # Crear queries muy específicas que incluyan el texto literal del artículo
    queries = [
        f"Artículo {article_number}º",  # Query más simple y directa
        f"ARTÍCULO {article_number}º",
        f"Artículo {article_number}º. {doc_type} {doc_number}",  # Con punto después del número
        f"Artículo {article_number}",  # Sin símbolo
        f"{doc_type} {doc_number} Artículo {article_number}",
    ]
    
    if doc_year:
        queries.extend([
            f"{doc_type} {doc_number} {doc_year} Artículo {article_number}º",
            f"ley {doc_number} de {doc_year} artículo {article_number}",
        ])
    
    all_results = []
    
    # Se generan múltiples variantes del query para cubrir diferencias tipográficas
    # frecuentes en documentos escaneados: símbolo ordinal (º/°), abreviatura "Art.",
    # mayúsculas/minúsculas y presencia o ausencia de punto tras el número.

    # Probar cada query con un k mayor para capturar más chunks
    for query in queries:
        try:
            docs = vectorstore.similarity_search(query, k=50)  # Aumentado de 30 a 50
            all_results.extend(docs)
        except Exception as e:
            continue
    
    # Eliminar duplicados manteniendo el orden
    seen_contents = set()
    unique_results = []
    for doc in all_results:
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            unique_results.append(doc)
    
    # Filtrar por metadata del documento correcto
    filtered_results = []
    for doc in unique_results:
        meta = doc.metadata
        meta_tipo = str(meta.get("tipo_documento", "")).upper()
        meta_numero = str(meta.get("numero", ""))
        meta_año = str(meta.get("año", ""))
        
        # Verificar que coincida el tipo y número
        if meta_tipo == doc_type.upper() and meta_numero == doc_number:
            # Si tenemos año, verificarlo también
            if doc_year is None or meta_año == doc_year:
                filtered_results.append(doc)
    
    # Buscar el artículo específico en los resultados filtrados
    patterns = [
        f"Artículo {article_number}º",  # Formato principal: "Artículo 4º"
        f"ARTÍCULO {article_number}º",  # Mayúsculas: "ARTÍCULO 4º"
        f"[Aa]rtículo {article_number}º",  # Case insensitive
        f"Artículo {article_number}\\.",  # Con punto directo: "Artículo 4."
        f"ARTÍCULO {article_number}\\.",
        f"Art\\. {article_number}º",  # Abreviatura: "Art. 4º"
        f"Artículo {article_number}[°º]",  # Con cualquier variante de grado
    ]
    
    # Primero intentar en resultados filtrados (más precisos)
    for doc in filtered_results:
        content = doc.page_content
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return content
    
    # Si no encontramos con filtros estrictos, buscar en TODOS los resultados únicos
    # (puede que el metadata esté mal pero el contenido sí tenga el artículo)
    print(f"      ℹ️ Buscando en {len(unique_results)} documentos únicos...")
    for doc in unique_results:
        if str(doc.metadata.get("numero", "")) == doc_number:
            content = doc.page_content
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    print(f"      ✓ Encontrado en documento con número {doc_number}")
                    return content
    
    # Último intento: buscar el patrón en CUALQUIER documento sin filtros
    # (muy relajado, solo para casos donde metadata esté completamente mal)
    print(f"      ℹ️ Último intento: buscando en todos los documentos sin filtros...")
    for doc in unique_results:
        content = doc.page_content
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                # Verificar que al menos mencione el documento correcto en el contenido
                if doc_number in content or doc_type.lower() in content.lower():
                    print(f"      ✓ Encontrado en documento: {doc.metadata.get('id_documento', 'desconocido')}")
                    return content
    
    return None


def select_dynamic_k(query: str, classification: str, tool_results: Optional[Dict] = None, min_k: int = 1, max_k: int = 10) -> int:
    """
    Selecciona dinámicamente el número de vecinos `k` usando un LLM (Groq por defecto).

    Devuelve un entero entre `min_k` y `max_k`. Si falla la llamada al LLM, aplica
    una estrategia de fallback simple.
    """

    # Función utilitaria (no expuesta como @tool) que delega en el LLM de Groq la decisión
    # de cuántos fragmentos recuperar (k), según la query, su clasificación y la herramienta
    # detectada. El resultado se fuerza al rango [min_k, max_k].
    # Fallback determinista si el LLM falla o no devuelve un entero:
    #   - "legal_specific" → k=5, "calculation" → k=2, resto → k=3.

    try:
        llm = init_groq_llm()

        tool_used = tool_results.get("tool_used") if tool_results else "None"
        prompt = f"""Determina cuántos documentos (k) deben recuperarse para una consulta.
    Responde SOLO con un número entero entre {min_k} y {max_k} (sin texto adicional).

    Consulta: "{query}"
    Clasificación: {classification}
    Herramienta detectada: {tool_used}

    Consideraciones (no escribirlas en la respuesta):
    - Si la consulta pide un documento específico o un artículo concreto, devolver 3.
    - Si pide resumen de un documento, devolver entre 1 y 3.
    - Para comparaciones o búsquedas generales amplias, 5-10 puede ser apropiado.
    - Para cálculos, 1-3 suele bastar.
    """

        response = llm.invoke(prompt)
        content = response.content.strip()

        # Extraer primer entero del texto de salida
        m = re.search(r"(\d+)", content)
        if m:
            k = int(m.group(1))
            k = max(min_k, min(max_k, k))
            print(f"   ℹ️ select_dynamic_k: LLM sugirió k={k}")
            return k
        else:
            raise ValueError("No se encontró entero en la respuesta del LLM")

    except Exception as e:
        print(f"   ⚠️ select_dynamic_k falló: {e} — aplicando fallback")
        # Fallback simple
        if classification == "legal_specific":
            return 5
        if classification == "calculation":
            return 2
        return 3


@tool
def compare_documents(doc_id1: str, doc_id2: str, topic: str, vectorstore: Any = None) -> Dict[str, any]:
    """
    Compara dos documentos legales sobre un tema específico.
    OPTIMIZADO: Limita la cantidad de información para evitar exceder límites del modelo.
    
    Args:
        doc_id1: ID del primer documento (ej: "LEY_1010" o "LEY_1010_2006")
        doc_id2: ID del segundo documento (ej: "DECRETO_1072" o "DECRETO_1072_2015")
        topic: Tema a comparar
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Diccionario con información comparativa (optimizada)
    """

    if vectorstore is None:
        raise ValueError("vectorstore is required for compare_documents")

    # Compara dos documentos legales sobre un tema usando búsquedas independientes por documento.
    # Para controlar el tamaño del contexto enviado al modelo, aplica dos límites duros:
    #   - MAX_FRAGMENTS_PER_DOC = 5 fragmentos por documento
    #   - MAX_CHARS_PER_FRAGMENT = 800 caracteres por fragmento
    # La búsqueda interna (search_document) intenta tres estrategias en orden descendente
    # de precisión: filtro por id_documento, filtro por tipo+número, y búsqueda manual con
    # filtrado post-hoc.
    
    # Configuración optimizada
    MAX_FRAGMENTS_PER_DOC = 5  # Reducido de 10 a 5
    MAX_CHARS_PER_FRAGMENT = 800  # Limitar cada fragmento
    
    def search_document(doc_id: str, query: str):
        """Busca un documento usando estrategias progresivas y consultas híbridas."""
        results = []

        parts = doc_id.split("_")
        doc_type = parts[0] if len(parts) >= 1 else ""
        doc_number = parts[1] if len(parts) >= 2 else ""
        doc_year = parts[2] if len(parts) >= 3 else ""

        # Usar más de una consulta ayuda cuando el tema (topic) no aparece literal
        # en todos los fragmentos del documento, pero sí existe contenido útil.
        # Agregar variantes temáticas para capturar sinónimos y conceptos relacionados
        query_variants = [
            query,
            f"{doc_type} {doc_number} {query}".strip(),
            f"{doc_type} {doc_number}".strip(),
            f"{doc_type} {doc_number} {doc_year}".strip(),
        ]
        
        # Agregar variantes semánticas del tema (si es acoso laboral → hostigamiento, conducta hostil, etc)
        theme_variants = []
        query_lower = query.lower()
        if "acoso" in query_lower and "laboral" in query_lower:
            theme_variants = [
                "hostigamiento laboral",
                "conducta hostil trabajo",
                "maltrato trabajador",
                "comportamiento abusivo",
                "victimización laboral"
            ]
        elif "jornada" in query_lower or "horario" in query_lower:
            theme_variants = [
                "horas de trabajo",
                "tiempo de labores",
                "duración jornada",
                "regimen horario"
            ]
        elif "salario" in query_lower or "remuneración" in query_lower:
            theme_variants = [
                "pago de salarios",
                "remuneración trabajador",
                "prestaciones económicas",
                "compensación laboral"
            ]
        
        # Combinar variantes
        query_variants.extend([f"{doc_type} {doc_number} {v}".strip() for v in theme_variants])
        query_variants.extend(theme_variants)
        query_variants = [q for q in query_variants if q]  # Remover vacías
        
        # Estrategia 1: Búsqueda con ID exacto
        for qv in query_variants:
            try:
                partial = _retrieve_docs(
                    vectorstore=vectorstore,
                    query=qv,
                    k=MAX_FRAGMENTS_PER_DOC,
                    filter_dict={"id_documento": {"$eq": doc_id}},
                )
                if partial:
                    results.extend(partial)
            except Exception:
                continue

        if results:
            dedup = []
            seen = set()
            for doc in results:
                key = doc.page_content
                if key in seen:
                    continue
                seen.add(key)
                dedup.append(doc)
                if len(dedup) >= MAX_FRAGMENTS_PER_DOC:
                    break
            return dedup
        
        # Estrategia 2: Buscar por tipo y número (sin año)
        if not results:
            try:
                if len(parts) >= 2:
                    for qv in query_variants:
                        partial = _retrieve_docs(
                            vectorstore=vectorstore,
                            query=qv,
                            k=MAX_FRAGMENTS_PER_DOC,
                            filter_dict={
                                "$and": [
                                    {"tipo_documento": {"$eq": doc_type}},
                                    {"numero": {"$eq": doc_number}}
                                ]
                            },
                        )
                        if partial:
                            results.extend(partial)
            except Exception:
                pass

            if results:
                dedup = []
                seen = set()
                for doc in results:
                    key = doc.page_content
                    if key in seen:
                        continue
                    seen.add(key)
                    dedup.append(doc)
                    if len(dedup) >= MAX_FRAGMENTS_PER_DOC:
                        break
                return dedup
        
        # Estrategia 3: Búsqueda manual (fallback)
        if not results:
            try:
                if len(parts) >= 2:
                    all_docs = []
                    for qv in query_variants:
                        batch = _retrieve_docs(vectorstore=vectorstore, query=qv, k=20)
                        all_docs.extend(batch)
                    
                    # Filtrado case-insensitive para tolerar variaciones en metadata.
                    results = [
                        doc for doc in all_docs
                        if (
                            str(doc.metadata.get("tipo_documento", "")).upper() == doc_type.upper()
                            and str(doc.metadata.get("numero", "")) == str(doc_number)
                        )
                    ][:MAX_FRAGMENTS_PER_DOC]
            except Exception:
                pass
        
        # Estrategia 4: Buscar SOLO por tipo de documento como último recurso
        # Esto es útil cuando el número o campo de metadata no coincide exactamente
        if not results:
            try:
                if doc_type:
                    # Buscar con todas las variantes para capturar más candidatos
                    all_docs = []
                    for qv in query_variants:
                        try:
                            batch = _retrieve_docs(
                                vectorstore=vectorstore,
                                query=qv,
                                k=100  # Aumentar significativamente para tener muchos candidatos
                            )
                            all_docs.extend(batch)
                        except Exception:
                            continue
                    
                    # Filtrar por tipo de documento (case-insensitive)
                    type_matched = [
                        doc for doc in all_docs
                        if str(doc.metadata.get("tipo_documento", "")).upper() == doc_type.upper()
                    ]
                    
                    # Si tenemos documentos del tipo correcto, priorizar por relevancia
                    if type_matched:
                        query_lower = query.lower()
                        query_words = set(w.lower() for w in query.split() if len(w) > 2)
                        
                        def score_relevance(doc):
                            """Puntúa fragmento por coincidencia de palabras clave y variantes temáticas"""
                            content_lower = doc.page_content.lower()
                            
                            # Puntaje base: palabras clave del query
                            score = sum(1 for word in query_words if word in content_lower)
                            
                            # Puntaje adicional por sinónimos/variantes temáticas
                            theme_synonyms = {
                                "acoso": ["hostig", "maltrat", "abusiv", "conduct hostil", "intimidac"],
                                "laboral": ["trabajo", "empleado", "obrero", "jornada"],
                                "jornada": ["horas", "horario", "tiempo trabajo", "duración"],
                                "salario": ["pago", "remuneración", "prestación", "compensación", "sueldo"]
                            }
                            
                            for theme, synonyms in theme_synonyms.items():
                                if theme in query_lower:
                                    syn_matches = sum(1 for syn in synonyms if syn in content_lower)
                                    score += syn_matches * 0.5  # Peso menor que coincidencias exactas
                            
                            return score
                        
                        # Deduplicar y puntuar
                        seen_content = set()
                        scored_docs = []
                        for doc in type_matched:
                            key = doc.page_content
                            if key not in seen_content:
                                seen_content.add(key)
                                rel_score = score_relevance(doc)
                                scored_docs.append((doc, rel_score))
                        
                        # Ordenar por relevancia (descendente) separando:
                        # 1. Documentos con alta relevancia (score > 0)
                        # 2. Documentos sin relevancia directa pero de tipo correcto
                        high_rel = [d for d, s in scored_docs if s > 0]
                        low_rel = [d for d, s in scored_docs if s == 0]
                        
                        # Combinar: primero los relevantes, luego los demás
                        results = (high_rel + low_rel)[:MAX_FRAGMENTS_PER_DOC]
            except Exception:
                pass
        
        return results
    
    def truncate_content(content: str, max_chars: int) -> str:
        """Trunca el contenido manteniendo las partes más relevantes"""
        if len(content) <= max_chars:
            return content
        # Mantener el inicio (más relevante semánticamente)
        return content[:max_chars] + "..."
    
    # Buscar en ambos documentos
    results1 = search_document(doc_id1, topic)
    results2 = search_document(doc_id2, topic)
    
    # Optimizar contenido: limitar y truncar fragmentos
    contenido1_optimizado = [
        truncate_content(doc.page_content, MAX_CHARS_PER_FRAGMENT) 
        for doc in results1[:MAX_FRAGMENTS_PER_DOC]
    ]
    
    contenido2_optimizado = [
        truncate_content(doc.page_content, MAX_CHARS_PER_FRAGMENT) 
        for doc in results2[:MAX_FRAGMENTS_PER_DOC]
    ]
    
    comparison = {
        "documento1": {
            "id": doc_id1,
            "fragmentos_encontrados": len(results1),
            "contenido": contenido1_optimizado,  # Contenido optimizado
            "total_fragmentos_disponibles": len(results1)  # Info para el usuario
        },
        "documento2": {
            "id": doc_id2,
            "fragmentos_encontrados": len(results2),
            "contenido": contenido2_optimizado,  # Contenido optimizado
            "total_fragmentos_disponibles": len(results2)
        },
        "tema_comparacion": topic,
        "optimizacion": {
            "fragmentos_por_documento": MAX_FRAGMENTS_PER_DOC,
            "caracteres_maximos_por_fragmento": MAX_CHARS_PER_FRAGMENT,
            "nota": "Contenido optimizado para evitar exceder límites del modelo"
        }
    }
    
    return comparison


@tool
def resume_document(doc_id: str, vectorstore: Any = None) -> Dict[str, any]:
    """
    Resume el contenido de un documento específico.
    
    Args:
        doc_id: ID del documento (ej: "LEY_1010_2006" o "DECRETO_36_2016")
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Diccionario con el contenido completo del documento y metadatos para que el LLM lo resuma
    """

    if vectorstore is None:
        raise ValueError("vectorstore is required for resume_document")

    # Recupera todos los fragmentos disponibles de un documento para que el LLM los resuma.
    # El contenido acumulado se limita a 15 000 caracteres (MAX_CHARS) para evitar
    # el error 413 al invocar el modelo; los fragmentos sobrantes se truncan con aviso explícito.
    # El campo "contenido_completo" devuelto puede estar truncado pese a su nombre,
    # pero se incluye un aviso claro para que el LLM lo tenga en cuenta al generar el resumen.
    
    results = []
    
    # Estrategia 1: Búsqueda con filtro de metadata por ID exacto (EFICIENTE)
    try:
        filter_dict = {"id_documento": {"$eq": doc_id}}
        # Buscar con filtro - recuperar fragmentos representativos del documento
        results = vectorstore.similarity_search(
            query="contenido documento",  # Query genérica
            k=50,  # Número reducido para evitar exceder límites del modelo
            filter=filter_dict
        )
    except Exception as e:
        print(f"⚠️ Estrategia 1 falló: {e}")
    
    # Estrategia 2: Por tipo y número usando filtros de metadata
    if not results:
        try:
            parts = doc_id.split("_")
            if len(parts) >= 2:
                doc_type = parts[0]
                doc_number = parts[1]
                
                # Usar filtro de ChromaDB
                filter_dict = {
                    "$and": [
                        {"tipo_documento": {"$eq": doc_type}},
                        {"numero": {"$eq": doc_number}}
                    ]
                }
                
                results = vectorstore.similarity_search(
                    query="contenido documento",
                    k=50,
                    filter=filter_dict
                )
        except Exception as e:
            print(f"⚠️ Estrategia 2 falló: {e}")
    
    # Estrategia 3: Búsqueda manual (fallback) - SOLO si las anteriores fallan
    if not results:
        try:
            parts = doc_id.split("_")
            if len(parts) >= 2:
                doc_type = parts[0]
                doc_number = parts[1]
                
                # Búsqueda amplia sin filtros
                all_docs = vectorstore.similarity_search(
                    query=f"{doc_type.lower()} {doc_number}",
                    k=100
                )
                
                # Filtrado manual
                results = [
                    doc for doc in all_docs 
                    if doc.metadata.get("id_documento", "") == doc_id or
                       (doc.metadata.get("tipo_documento", "") == doc_type and
                        str(doc.metadata.get("numero", "")) == doc_number)
                ]
        except Exception as e:
            print(f"⚠️ Estrategia 3 falló: {e}")
    
    if not results:
        return {
            "id_documento": doc_id,
            "titulo": "Documento no encontrado",
            "tipo_documento": "Desconocido",
            "año": "No especificado",
            "contenido_completo": f"No se encontraron fragmentos para el documento {doc_id}",
            "fragmentos_principales": [],
            "fragmentos_encontrados": 0,
            "metadatos": {}
        }
    
    # Compilar fragmentos con límite de tamaño
    metadatos = results[0].metadata if results else {}
    
    # Juntar contenido con límite de caracteres para evitar error 413
    fragmentos_ordenados = [doc.page_content for doc in results]
    contenido_completo = ""
    MAX_CHARS = 15000  # Límite de caracteres para evitar exceder el modelo
    
    for fragmento in fragmentos_ordenados:
        if len(contenido_completo) + len(fragmento) + 10 < MAX_CHARS:
            contenido_completo += fragmento + "\n\n---\n\n"
        else:
            # Truncar si excede el límite
            contenido_completo += "\n\n[... contenido truncado para ajustarse al modelo ...]"
            break
    
    # Devolver el contenido limitado para que el LLM lo resuma
    return {
        "id_documento": doc_id,
        "titulo": metadatos.get("titulo", f"{metadatos.get('tipo_documento', 'Documento')} {metadatos.get('numero', '')}"),
        "tipo_documento": metadatos.get("tipo_documento", "Desconocido"),
        "año": metadatos.get("año", "No especificado"),
        "contenido_completo": contenido_completo,  # Contenido sin truncar
        "fragmentos_principales": fragmentos_ordenados[:10],  # Primeros 10 fragmentos
        "fragmentos_encontrados": len(results),
        "metadatos": metadatos
    }



