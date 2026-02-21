"""
Tools para el RAG de normativa laboral colombiana.
Estas herramientas extienden las capacidades del sistema RAG.
"""
from typing import List, Dict, Optional
from langchain_core.tools import tool
from langchain_core.documents import Document
import re


@tool
def search_by_document_type(query: str, doc_type: str, vectorstore) -> List[Document]:
    """
    Busca documentos por tipo específico (LEY, DECRETO, SENTENCIA).
    
    Args:
        query: Consulta de búsqueda
        doc_type: Tipo de documento (LEY, DECRETO, SENTENCIA)
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Lista de documentos del tipo especificado
    """
    results = vectorstore.similarity_search(query, k=10)
    
    # Filtrar por tipo de documento
    filtered = [
        doc for doc in results 
        if doc.metadata.get("tipo_documento", "").upper() == doc_type.upper()
    ]
    
    return filtered[:5]


@tool
def search_by_year_range(query: str, start_year: int, end_year: int, vectorstore) -> List[Document]:
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
def extract_specific_article(doc_id: str, article_number: str, vectorstore) -> Optional[str]:
    """
    Extrae un artículo específico de un documento legal.
    
    Args:
        doc_id: ID del documento (ej: "LEY_1010_2006" o "LEY_1010")
        article_number: Número del artículo a extraer
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Contenido del artículo o None si no se encuentra
    """
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


@tool
def compare_documents(doc_id1: str, doc_id2: str, topic: str, vectorstore) -> Dict[str, any]:
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
    
    # Configuración optimizada
    MAX_FRAGMENTS_PER_DOC = 5  # Reducido de 10 a 5
    MAX_CHARS_PER_FRAGMENT = 800  # Limitar cada fragmento
    
    def search_document(doc_id: str, query: str):
        """Busca un documento usando estrategias progresivas"""
        results = []
        
        # Estrategia 1: Búsqueda con ID exacto
        try:
            results = vectorstore.similarity_search(
                query,
                k=MAX_FRAGMENTS_PER_DOC,  # Usar configuración optimizada
                filter={"id_documento": {"$eq": doc_id}}
            )
        except Exception as e:
            pass
        
        # Estrategia 2: Buscar por tipo y número (sin año)
        if not results:
            try:
                parts = doc_id.split("_")
                if len(parts) >= 2:
                    doc_type = parts[0]
                    doc_number = parts[1]
                    
                    results = vectorstore.similarity_search(
                        query,
                        k=MAX_FRAGMENTS_PER_DOC,
                        filter={
                            "$and": [
                                {"tipo_documento": {"$eq": doc_type}},
                                {"numero": {"$eq": doc_number}}
                            ]
                        }
                    )
            except Exception as e:
                pass
        
        # Estrategia 3: Búsqueda manual (fallback)
        if not results:
            try:
                parts = doc_id.split("_")
                if len(parts) >= 2:
                    doc_type = parts[0]
                    doc_number = parts[1]
                    
                    all_docs = vectorstore.similarity_search(
                        f"{doc_type.lower()} {doc_number} {query}",
                        k=20  # Reducido de 100 a 20
                    )
                    
                    results = [
                        doc for doc in all_docs
                        if (doc.metadata.get("tipo_documento", "") == doc_type and
                            str(doc.metadata.get("numero", "")) == doc_number)
                    ][:MAX_FRAGMENTS_PER_DOC]
            except Exception as e:
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
def resume_document(doc_id: str, vectorstore) -> Dict[str, any]:
    """
    Resume el contenido de un documento específico.
    
    Args:
        doc_id: ID del documento (ej: "LEY_1010_2006" o "DECRETO_36_2016")
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Diccionario con el contenido completo del documento y metadatos para que el LLM lo resuma
    """
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



