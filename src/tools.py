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


# Diccionario de tools para fácil acceso
AVAILABLE_TOOLS = {
    "search_by_document_type": search_by_document_type,
    "search_by_year_range": search_by_year_range,
    "calculate_prestaciones_sociales": calculate_prestaciones_sociales,
    "extract_specific_article": extract_specific_article,
    "compare_documents": compare_documents,
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
"""
