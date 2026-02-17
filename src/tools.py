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
    results = []
    query = f"articulo {article_number}"
    
    # Estrategia 1: Búsqueda con ID exacto
    try:
        results = vectorstore.similarity_search(
            query,
            k=20,
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
                    k=20,
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
                    k=100
                )
                
                results = [
                    doc for doc in all_docs
                    if (doc.metadata.get("tipo_documento", "") == doc_type and
                        str(doc.metadata.get("numero", "")) == doc_number)
                ][:20]
        except Exception as e:
            pass
    
    # Buscar el artículo específico en los resultados
    for doc in results:
        content = doc.page_content
        # Buscar patrones como "Artículo X", "ARTÍCULO X", "Art. X"
        patterns = [
            f"Artículo {article_number}[.:]",
            f"ARTÍCULO {article_number}[.:]",
            f"Art\\. {article_number}[.:]",
            f"Artículo {article_number}\\.",
        ]
        
        for pattern in patterns:
            if re.search(pattern, content, re.IGNORECASE):
                return content
    
    return None


@tool
def compare_documents(doc_id1: str, doc_id2: str, topic: str, vectorstore) -> Dict[str, any]:
    """
    Compara dos documentos legales sobre un tema específico.
    
    Args:
        doc_id1: ID del primer documento (ej: "LEY_1010" o "LEY_1010_2006")
        doc_id2: ID del segundo documento (ej: "DECRETO_1072" o "DECRETO_1072_2015")
        topic: Tema a comparar
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Diccionario con información comparativa
    """
    
    def search_document(doc_id: str, query: str):
        """Busca un documento usando estrategias progresivas"""
        results = []
        
        # Estrategia 1: Búsqueda con ID exacto
        try:
            results = vectorstore.similarity_search(
                query,
                k=10,
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
                        k=10,
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
                        k=100
                    )
                    
                    results = [
                        doc for doc in all_docs
                        if (doc.metadata.get("tipo_documento", "") == doc_type and
                            str(doc.metadata.get("numero", "")) == doc_number)
                    ][:10]
            except Exception as e:
                pass
        
        return results
    
    # Buscar en ambos documentos
    results1 = search_document(doc_id1, topic)
    results2 = search_document(doc_id2, topic)
    
    comparison = {
        "documento1": {
            "id": doc_id1,
            "fragmentos_encontrados": len(results1),
            "contenido": [doc.page_content for doc in results1]  # Contenido completo de todos los fragmentos
        },
        "documento2": {
            "id": doc_id2,
            "fragmentos_encontrados": len(results2),
            "contenido": [doc.page_content for doc in results2]  # Contenido completo de todos los fragmentos
        },
        "tema_comparacion": topic
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
        # Buscar con filtro - recuperar todos los fragmentos del documento
        results = vectorstore.similarity_search(
            query="contenido documento",  # Query genérica
            k=500,  # Suficientes fragmentos para documento completo
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
                    k=500,
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
                    k=1000
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
    
    # Compilar TODOS los fragmentos (sin truncar)
    metadatos = results[0].metadata if results else {}
    
    # Juntar todo el contenido disponible
    fragmentos_ordenados = [doc.page_content for doc in results]
    contenido_completo = "\n\n---\n\n".join(fragmentos_ordenados)
    
    # Devolver el contenido completo para que el LLM lo resuma
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



