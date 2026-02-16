"""
Tools para el RAG de normativa laboral colombiana.
Estas herramientas extienden las capacidades del sistema RAG.
"""
from typing import List, Dict, Optional
from datetime import datetime
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
        doc_id: ID del documento (ej: "LEY_1010_2006")
        article_number: Número del artículo a extraer
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Contenido del artículo o None si no se encuentra
    """
    # Buscar por ID de documento
    results = vectorstore.similarity_search(
        f"articulo {article_number}",
        k=20,
        filter={"id_documento": doc_id}
    )
    
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
        doc_id1: ID del primer documento
        doc_id2: ID del segundo documento
        topic: Tema a comparar
        vectorstore: Vectorstore de ChromaDB
        
    Returns:
        Diccionario con información comparativa
    """
    # Buscar en ambos documentos
    results1 = vectorstore.similarity_search(
        topic,
        k=5,
        filter={"id_documento": doc_id1}
    )
    
    results2 = vectorstore.similarity_search(
        topic,
        k=5,
        filter={"id_documento": doc_id2}
    )
    
    comparison = {
        "documento1": {
            "id": doc_id1,
            "fragmentos_encontrados": len(results1),
            "contenido": [doc.page_content[:200] for doc in results1[:2]]
        },
        "documento2": {
            "id": doc_id2,
            "fragmentos_encontrados": len(results2),
            "contenido": [doc.page_content[:200] for doc in results2[:2]]
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
    
    # Estrategia 1: Búsqueda por ID exacto
    try:
        # Buscar todos los documentos del ID
        all_docs = vectorstore.similarity_search("documento ley decreto sentencia normativa legal", k=100)
        results = [
            doc for doc in all_docs 
            if doc.metadata.get("id_documento", "") == doc_id
        ]
    except Exception as e:
        pass
    
    # Estrategia 2: Por tipo y número si no encontramos el ID exacto
    if not results:
        try:
            parts = doc_id.split("_")
            if len(parts) >= 2:
                doc_type = parts[0]
                doc_number = parts[1]
                
                all_docs = vectorstore.similarity_search("documento ley decreto sentencia normativa", k=150)
                results = [
                    doc for doc in all_docs 
                    if doc.metadata.get("tipo_documento", "") == doc_type and
                       str(doc.metadata.get("numero", "")) == doc_number
                ]
        except Exception as e:
            pass
    
    # Estrategia 3: Con query descriptiva
    if not results:
        try:
            parts = doc_id.split("_")
            if len(parts) >= 2:
                doc_type = parts[0].lower()
                doc_number = parts[1]
                query = f"{doc_type} número {doc_number}"
                
                docs_with_scores = vectorstore.similarity_search_with_score(query, k=100)
                results = [doc for doc, score in docs_with_scores]
                results = [
                    doc for doc in results 
                    if str(doc.metadata.get("numero", "")) == doc_number and
                       doc.metadata.get("tipo_documento", "") == parts[0]
                ]
        except Exception as e:
            pass
    
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


# Diccionario de tools para fácil acceso
AVAILABLE_TOOLS = {
    "search_by_document_type": search_by_document_type,
    "search_by_year_range": search_by_year_range,
    "calculate_prestaciones_sociales": calculate_prestaciones_sociales,
    "extract_specific_article": extract_specific_article,
    "compare_documents": compare_documents,
    "resume_document": resume_document,
}


def get_tool_descriptions() -> str:
    """
    Retorna descripciones de todas las herramientas disponibles.
    """
    return """
Herramientas disponibles:

1. search_by_document_type(query, doc_type, vectorstore)
   - Busca por tipo específico: LEY, DECRETO, SENTENCIA
   
2. search_by_year_range(query, start_year, end_year, vectorstore)
   - Busca documentos en un rango de años
   
3. calculate_prestaciones_sociales(salario_mensual, dias_trabajados, años_servicio)
   - Calcula cesantías, prima, vacaciones e intereses
   
4. extract_specific_article(doc_id, article_number, vectorstore)
   - Extrae un artículo específico de un documento
   
5. compare_documents(doc_id1, doc_id2, topic, vectorstore)
   - Compara dos documentos sobre un tema

6. resume_document(doc_id, vectorstore)
   - Resume el contenido de un documento específico
"""
