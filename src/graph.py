"""
LangGraph workflow para RAG de normativa laboral colombiana.

Flujo:
1. Classify: Clasifica la consulta del usuario
2. Retrieve: Recupera documentos relevantes de ChromaDB
3. Tool Calling: Ejecuta herramientas especializadas si es necesario
4. Generate: Genera respuesta usando el contexto
5. Verify: Verifica la calidad y exactitud de la respuesta
"""
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document

from langgraph.graph import END, StateGraph

from src.config import init_embeddings, init_gemini_llm, init_groq_llm
from src.vectorstore import load_chroma_index
import os
import re


class RAGState(TypedDict):
    """Estado compartido entre todos los nodos del grafo."""
    query: str  # Consulta original del usuario
    classification: str  # Clasificación de la consulta
    documents: List[Document]  # Documentos recuperados
    tool_results: Optional[Dict[str, Any]]  # Resultados de tools ejecutadas
    answer: str  # Respuesta generada
    verification: Dict[str, Any]  # Resultados de verificación
    metadata: Dict[str, Any]  # Metadata adicional


def classify_node(state: RAGState) -> RAGState:
    """
    Clasifica la consulta del usuario para determinar el tipo de pregunta.
    
    Categorías:
    - legal_specific: Pregunta sobre normativa específica
    - procedural: Pregunta sobre procedimientos o trámites
    - general: Pregunta general sobre derechos laborales
    - calculation: Pregunta que requiere cálculos
    - resume: Pregunta que requiere un resumen de un documento
    """
    query = state["query"]
    
    print(f"\n🔍 CLASIFICANDO: {query}")
    
    llm = init_gemini_llm()
    
    classification_prompt = f"""Clasifica la siguiente consulta laboral en UNA de estas categorías:

1. legal_specific: Pregunta sobre una ley, decreto o sentencia específica
2. procedural: Pregunta sobre procedimientos, trámites o pasos a seguir
3. general: Pregunta general sobre derechos, obligaciones o conceptos laborales
4. calculation: Pregunta que requiere cálculos (liquidaciones, pagos, etc.)
5. resume: Pregunta que requiere un resumen de un documento específico

Consulta: "{query}"

Responde SOLO con el nombre de la categoría (sin explicaciones):"""

    try:
        response = llm.invoke(classification_prompt)
        classification = response.content.strip().lower()
        
        # Validar clasificación
        valid_classifications = ["legal_specific", "procedural", "general", "calculation", "resume"]
        if classification not in valid_classifications:
            classification = "general"
        
        print(f"   ✓ Clasificación: {classification}")
        
    except Exception as e:
        print(f"   ⚠️ Error en clasificación (usando clasificación simple): {str(e)[:100]}")
        # Clasificación simple basada en palabras clave
        query_lower = query.lower()
        if any(word in query_lower for word in ["calcular", "liquidar", "cuánto", "pagar"]):
            classification = "calculation"
        elif any(word in query_lower for word in ["resumen", "resume", "resumir"]):
            classification = "resume"
        elif any(word in query_lower for word in ["ley", "decreto", "artículo", "sentencia"]):
            classification = "legal_specific"
        elif any(word in query_lower for word in ["cómo", "procedimiento", "trámite", "pasos"]):
            classification = "procedural"
        else:
            classification = "general"
        print(f"   ✓ Clasificación (fallback): {classification}")
    
    state["classification"] = classification
    state["metadata"] = {"classification_method": "gemini"}
    
    return state


def tool_calling_node(state: RAGState) -> RAGState:
    """
    Determina si se necesita ejecutar alguna herramienta especializada.
    Detecta todos los casos y prepara los parámetros para ejecutar las herramientas.
    """
    query = state["query"]
    classification = state.get("classification", "general")
    
    print(f"\n🔧 EVALUANDO HERRAMIENTAS ESPECIALIZADAS")
    
    tool_results = None
    
    # 0. HERRAMIENTA: resume_document (PRIORIDAD: Detectar primero para evitar confusiones)
    # Detectar consultas que requieren resumen de documento
    if classification == "resume":
        doc_type = None
        doc_number = None
        doc_year = None
        doc_id = None
        
        # Patrón 1: Sentencias (C-1234, T-1234, SU-1234, etc.)
        sentencia_match = re.search(r'resumen\s+(?:de\s+)?(?:la\s+)?sentencia\s+([CT])-?(\d+)(?:\s+de\s+(\d{4}))?', query, re.IGNORECASE)
        if sentencia_match:
            sentencia_prefix = sentencia_match.group(1).upper()
            sentencia_num = sentencia_match.group(2)
            doc_year = sentencia_match.group(3) if sentencia_match.group(3) else None
            
            doc_type = "SENTENCIA"
            doc_number = f"{sentencia_prefix}{sentencia_num}"
            
            if doc_year:
                doc_id = f"SENTENCIA_{doc_number}_{doc_year}"
            else:
                doc_id = f"SENTENCIA_{doc_number}"
            
            print(f"   ✓ Detectado: Resumen de sentencia - {doc_id}")
        
        # Patrón 2: LEY/DECRETO (ej: "resumen de la ley 1010" o "resumen del decreto 36 de 2016")
        if not doc_id:
            doc_pattern = re.search(r'resumen\s+(?:de\s+)?(?:(?:la|del)\s+)?(ley|decreto|acto legislativo)\s+(\d+)(?:\s+de\s+(\d{4}))?', query, re.IGNORECASE)
            if doc_pattern:
                doc_type = doc_pattern.group(1).upper()
                doc_number = doc_pattern.group(2)
                doc_year = doc_pattern.group(3) if doc_pattern.group(3) else None
                
                if doc_year:
                    doc_id = f"{doc_type}_{doc_number}_{doc_year}"
                else:
                    doc_id = f"{doc_type}_{doc_number}"
                
                print(f"   ✓ Detectado: Resumen de documento - {doc_id}")
        
        # Patrón 3: Estructura inversa (ej: "ley 1010 resumen" o "decreto 36 de 2016 resumen")
        if not doc_id:
            inverse_pattern = re.search(r'(ley|decreto|sentencia|acto legislativo)\s+([CT])?-?(\d+)(?:\s+de\s+(\d{4}))?.*resumen', query, re.IGNORECASE)
            if inverse_pattern:
                if inverse_pattern.group(2):  # Es una sentencia
                    doc_type = "SENTENCIA"
                    doc_number = f"{inverse_pattern.group(2).upper()}{inverse_pattern.group(3)}"
                    doc_year = inverse_pattern.group(4) if inverse_pattern.group(4) else None
                else:  # Es un LEY/DECRETO
                    doc_type = inverse_pattern.group(1).upper()
                    doc_number = inverse_pattern.group(3)
                    doc_year = inverse_pattern.group(4) if inverse_pattern.group(4) else None
                
                if doc_year:
                    doc_id = f"{doc_type}_{doc_number}_{doc_year}"
                else:
                    doc_id = f"{doc_type}_{doc_number}"
                
                print(f"   ✓ Detectado: Resumen de documento - {doc_id}")
        
        if doc_id:
            tool_results = {
                "tool_used": "resume_document",
                "doc_id": doc_id,
                "doc_type": doc_type,
                "doc_number": doc_number,
                "doc_year": doc_year
            }
    
    # 1. HERRAMIENTA: calculate_prestaciones_sociales
    if not tool_results and classification == "calculation":
        if any(word in query.lower() for word in ["prestaciones", "cesantías", "prima", "liquidación"]):
            print("   ✓ Detectado: Cálculo de prestaciones sociales")
            # Extraer valores si están en la consulta
            salary_match = re.search(r'[\$]?([\d,]+(?:\.\d{2})?)', query)
            if salary_match:
                try:
                    salario = float(salary_match.group(1).replace(',', ''))
                    tool_results = {
                        "tool_used": "calculate_prestaciones_sociales",
                        "requires_user_input": True,
                        "message": "Necesito más información para el cálculo",
                        "salario_detectado": salario
                    }
                    print(f"      Salario detectado: ${salario:,.2f}")
                except:
                    pass
    
    # 2. HERRAMIENTA: extract_specific_article
    # Detectar consultas sobre artículos específicos (ej: "artículo 5 de la ley 1010", "artículo 2 del decreto 1072")
    if not tool_results and classification != "resume":
        article_match = re.search(
            r'art[íi]culo\s+(\d+)\s+.*?\b(ley|decreto)\s+(\d+)(?:\s+de\s+(\d{4}))?',
            query,
            re.IGNORECASE
        )
        if article_match:
            article_num = article_match.group(1)
            doc_type = article_match.group(2).upper()
            doc_num = article_match.group(3)
            doc_year = article_match.group(4) if article_match.group(4) else None
            
            # Construir ID del documento
            if doc_year:
                doc_id = f"{doc_type}_{doc_num}_{doc_year}"
                print(f"   ✓ Detectado: Extracción de artículo {article_num} de {doc_id}")
            else:
                doc_id = f"{doc_type}_{doc_num}"
                print(f"   ✓ Detectado: Extracción de artículo {article_num} de {doc_id} (sin año)")
            
            tool_results = {
                "tool_used": "extract_specific_article",
                "doc_id": doc_id,
                "article_number": article_num,
                "doc_type": doc_type,
                "doc_number": doc_num,
                "doc_year": doc_year
            }
    
    # 3. HERRAMIENTA: compare_documents
    # Detectar comparaciones entre documentos
    if not tool_results:
        compare_patterns = [
            r'compar[ae]r?\s+(?:la\s+)?(ley|decreto)\s+(\d+).*(?:con|y|vs).*(?:la\s+)?(ley|decreto)\s+(\d+)',
            r'diferencias?\s+entre\s+(?:la\s+)?(ley|decreto)\s+(\d+).*(?:y|con).*(?:la\s+)?(ley|decreto)\s+(\d+)'
        ]
        
        for pattern in compare_patterns:
            compare_match = re.search(pattern, query, re.IGNORECASE)
            if compare_match:
                doc1_type = compare_match.group(1).upper()
                doc1_num = compare_match.group(2)
                doc2_type = compare_match.group(3).upper()
                doc2_num = compare_match.group(4)
                
                # Extraer tema de comparación (palabras clave)
                topic_words = []
                for word in query.lower().split():
                    if word not in ['comparar', 'diferencia', 'entre', 'con', 'la', 'el', 'de', 'ley', 'decreto', 'y']:
                        if not word.isdigit():
                            topic_words.append(word)
                topic = " ".join(topic_words[:3]) if topic_words else "contenido general"
                
                doc1_id = f"{doc1_type}_{doc1_num}"
                doc2_id = f"{doc2_type}_{doc2_num}"
                
                print(f"   ✓ Detectado: Comparación entre {doc1_id} y {doc2_id}")
                print(f"      Tema: {topic}")
                
                tool_results = {
                    "tool_used": "compare_documents",
                    "doc_id1": doc1_id,
                    "doc_id2": doc2_id,
                    "topic": topic
                }
                break
    
    # 4. HERRAMIENTA: search_by_document_type
    # Patrón 1: LEY/DECRETO con número y año opcional
    if not tool_results and classification != "resume":
        doc_type_match = re.search(r'(ley|decreto)\s+(\d+)(?:\s+de\s+(\d{4}))?', query, re.IGNORECASE)
        if doc_type_match:
            doc_type = doc_type_match.group(1).upper()
            doc_num = doc_type_match.group(2)
            doc_year = doc_type_match.group(3) if doc_type_match.group(3) else None
            
            if doc_year:
                print(f"   ✓ Detectado: Búsqueda por documento específico - {doc_type} {doc_num} DE {doc_year}")
            else:
                print(f"   ✓ Detectado: Búsqueda por documento específico - {doc_type} {doc_num}")
                
            tool_results = {
                "tool_used": "search_by_document_type",
                "doc_type": doc_type,
                "doc_number": doc_num,
                "doc_year": doc_year
            }
    
    # Patrón 2: SENTENCIA (formato C-200, T-1234, etc.)
    if not tool_results:
        sentencia_match = re.search(r'sentencia\s+([CT])-?(\d+)(?:\s+de\s+(\d{4}))?', query, re.IGNORECASE)
        if sentencia_match:
            sentencia_prefix = sentencia_match.group(1).upper()
            sentencia_num = sentencia_match.group(2)
            doc_year = sentencia_match.group(3) if sentencia_match.group(3) else None
            
            # El número de sentencia incluye el prefijo: C200, T1234
            doc_num = f"{sentencia_prefix}{sentencia_num}"
            
            if doc_year:
                print(f"   ✓ Detectado: Búsqueda por sentencia específica - SENTENCIA_{doc_num}_{doc_year}")
            else:
                print(f"   ✓ Detectado: Búsqueda por sentencia específica - SENTENCIA_{doc_num}")
            
            tool_results = {
                "tool_used": "search_by_document_type",
                "doc_type": "SENTENCIA",
                "doc_number": doc_num,
                "doc_year": doc_year
            }
    
    # 5. HERRAMIENTA: search_by_year_range
    # Detectar rango de años
    if not tool_results and classification != "resume":
        year_match = re.findall(r'\b(?:19|20)\d{2}\b', query)
        if len(year_match) >= 2:
            years = sorted([int(y) for y in year_match])
            print(f"   ✓ Detectado: Búsqueda por rango de años - {years[0]} a {years[-1]}")
            tool_results = {
                "tool_used": "search_by_year_range",
                "start_year": years[0],
                "end_year": years[-1]
            }
    
    if tool_results:
        print(f"   ✓ Herramienta seleccionada: {tool_results.get('tool_used', 'N/A')}")
    else:
        print("   • No se requieren herramientas especializadas")
    
    state["tool_results"] = tool_results
    return state


def retrieve_node(state: RAGState) -> RAGState:
    """
    Recupera documentos relevantes de ChromaDB basándose en la consulta.
    Ejecuta las herramientas especializadas cuando son necesarias.
    Mantiene la búsqueda por metadata para consultas específicas.
    """
    query = state["query"]
    classification = state.get("classification", "general")
    tool_results = state.get("tool_results")
    
    print(f"\n📚 RECUPERANDO DOCUMENTOS")
    
    try:
        # Cargar índice
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "normativa_laboral")
        
        embedding_fn = init_embeddings()
        vectorstore = load_chroma_index(persist_dir, embedding_fn, collection_name)
        
        # Determinar número de documentos a recuperar según clasificación
        k = 5 if classification == "legal_specific" else 3
        
        # EJECUTAR HERRAMIENTAS ESPECIALIZADAS
        documents = []
        filter_dict = None
        
        if tool_results:
            tool_used = tool_results.get("tool_used")
            
            # 1. Ejecutar herramienta extract_specific_article
            if tool_used == "extract_specific_article":
                from src.tools import extract_specific_article
                print(f"   🔧 Ejecutando: extract_specific_article")
                
                doc_id = tool_results.get("doc_id")
                article_number = tool_results.get("article_number")
                
                # La herramienta devuelve el contenido del artículo
                article_content = extract_specific_article.invoke({
                    "doc_id": doc_id,
                    "article_number": article_number,
                    "vectorstore": vectorstore
                })
                
                if article_content:
                    # Crear documento con el artículo encontrado
                    doc = Document(
                        page_content=article_content,
                        metadata={
                            "id_documento": doc_id,
                            "tipo_documento": tool_results.get("doc_type"),
                            "articulo": article_number,
                            "source": "extract_specific_article"
                        }
                    )
                    documents = [doc]
                    print(f"      ✓ Artículo {article_number} extraído exitosamente")
                else:
                    print(f"      ⚠️ Artículo {article_number} no encontrado en {doc_id}")
                    # Para extracción de artículos, si no se encuentra, devolver mensaje
                    # NO hacer búsqueda genérica porque el usuario pidió algo específico
                    doc = Document(
                        page_content=f"No se pudo encontrar el Artículo {article_number} en {doc_id}. El formato del artículo en el documento puede ser diferente o el artículo puede no existir en la base de datos.",
                        metadata={
                            "id_documento": doc_id,
                            "tipo_documento": tool_results.get("doc_type"),
                            "articulo": article_number,
                            "source": "extract_specific_article",
                            "not_found": True
                        }
                    )
                    documents = [doc]
                    # No establecer filter_dict para evitar búsqueda adicional
            
            # 2. Ejecutar herramienta compare_documents
            elif tool_used == "compare_documents":
                from src.tools import compare_documents
                print(f"   🔧 Ejecutando: compare_documents")
                
                doc_id1 = tool_results.get("doc_id1")
                doc_id2 = tool_results.get("doc_id2")
                topic = tool_results.get("topic")
                
                comparison_result = compare_documents.invoke({
                    "doc_id1": doc_id1,
                    "doc_id2": doc_id2,
                    "topic": topic,
                    "vectorstore": vectorstore
                })
                
                # Guardar resultado de comparación en tool_results para usarlo en generate
                state["tool_results"]["comparison_result"] = comparison_result
                print(f"      ✓ Comparación completada")
                print(f"         Doc1: {comparison_result['documento1']['fragmentos_encontrados']} fragmentos")
                print(f"         Doc2: {comparison_result['documento2']['fragmentos_encontrados']} fragmentos")
                
                # Crear documentos directamente desde los resultados de la comparación
                doc1_contenido = comparison_result['documento1'].get('contenido', [])
                doc2_contenido = comparison_result['documento2'].get('contenido', [])
                
                for i, contenido in enumerate(doc1_contenido, 1):
                    doc = Document(
                        page_content=contenido,
                        metadata={
                            "id_documento": doc_id1,
                            "source": "compare_documents",
                            "documento": 1,
                            "fragmento": i
                        }
                    )
                    documents.append(doc)
                
                for i, contenido in enumerate(doc2_contenido, 1):
                    doc = Document(
                        page_content=contenido,
                        metadata={
                            "id_documento": doc_id2,
                            "source": "compare_documents",
                            "documento": 2,
                            "fragmento": i
                        }
                    )
                    documents.append(doc)
                
                print(f"      ✓ {len(documents)} documentos creados para contexto")
            
            # 3. Ejecutar herramienta search_by_year_range
            elif tool_used == "search_by_year_range":
                from src.tools import search_by_year_range
                print(f"   🔧 Ejecutando: search_by_year_range")
                
                start_year = tool_results.get("start_year")
                end_year = tool_results.get("end_year")
                
                # La herramienta ya hace la búsqueda y filtrado
                documents = search_by_year_range.invoke({
                    "query": query,
                    "start_year": start_year,
                    "end_year": end_year,
                    "vectorstore": vectorstore
                })
                
                print(f"      ✓ {len(documents)} documentos encontrados en rango {start_year}-{end_year}")
            
            # 4. Ejecutar herramienta resume_document
            elif tool_used == "resume_document":
                from src.tools import resume_document
                print(f"   🔧 Ejecutando: resume_document")
                
                doc_id = tool_results.get("doc_id")
                
                # La herramienta devuelve un diccionario con información del documento
                resume_result = resume_document.invoke({
                    "doc_id": doc_id,
                    "vectorstore": vectorstore
                })
                
                # Guardar resultado del resumen en tool_results para usarlo en generate
                state["tool_results"]["resume_result"] = resume_result
                print(f"      ✓ Contenido del documento recuperado")
                print(f"         Título: {resume_result.get('titulo', 'Sin título')}")
                print(f"         Fragmentos encontrados: {resume_result.get('fragmentos_encontrados', 0)}")
                
                # Crear documento con el CONTENIDO COMPLETO para que el LLM lo resuma
                doc = Document(
                    page_content=resume_result.get("contenido_completo", "No se encontró información disponible"),
                    metadata={
                        "id_documento": doc_id,
                        "titulo": resume_result.get("titulo"),
                        "tipo_documento": resume_result.get("tipo_documento"),
                        "año": resume_result.get("año"),
                        "source": "resume_document",
                        "fragmentos_encontrados": resume_result.get("fragmentos_encontrados", 0)
                    }
                )
                documents = [doc]
                
                if resume_result.get("fragmentos_encontrados", 0) == 0:
                    print(f"      ⚠️ No se encontraron fragmentos en la base de datos para este documento")
            
            # 5. Herramienta search_by_document_type - mantener búsqueda por metadata
            elif tool_used == "search_by_document_type":
                doc_type = tool_results.get("doc_type")
                doc_number = tool_results.get("doc_number")
                doc_year = tool_results.get("doc_year")
                
                print(f"   🎯 Búsqueda con metadata: {doc_type} {doc_number}")
                
                # Construir el ID del documento que buscamos
                # Formato esperado en metadata: "LEY_1010_2006", "DECRETO_1072_2015", etc.
                if doc_year:
                    target_id = f"{doc_type}_{doc_number}_{doc_year}"
                    print(f"      ID objetivo: {target_id}")
                else:
                    target_id = f"{doc_type}_{doc_number}"
                    print(f"      ID objetivo (sin año): {target_id}")
                
                # Intentar búsqueda con filtro exacto - MANTENER LÓGICA ACTUAL
                try:
                    # Estrategia 1: Búsqueda directa con ID completo
                    if doc_year:
                        filter_dict = {"id_documento": {"$eq": target_id}}
                    else:
                        # Solo tenemos tipo y número, no año
                        filter_dict = {
                            "$and": [
                                {"tipo_documento": {"$eq": doc_type}},
                                {"numero": {"$eq": doc_number}}
                            ]
                        }
                except Exception as filter_error:
                    print(f"      ⚠️ Error construyendo filtros: {filter_error}")
                    filter_dict = None
        
        # EJECUTAR BÚSQUEDA si no se obtuvieron documentos de herramientas
        if not documents:
            try:
                if filter_dict:
                    # Búsqueda con filtros de metadata
                    docs_with_scores = vectorstore.similarity_search_with_score(
                        query, k=k, filter=filter_dict
                    )
                    
                    # Si no encontramos con el filtro exacto, intentar estrategias alternativas
                    if not docs_with_scores and tool_results:
                        tool_used = tool_results.get("tool_used")
                        
                        if tool_used == "search_by_document_type":
                            doc_type = tool_results.get("doc_type")
                            doc_number = tool_results.get("doc_number")
                            
                            print(f"      ⚠️ Estrategia 2: Buscando solo por tipo y número")
                            filter_dict = {
                                "$and": [
                                    {"tipo_documento": {"$eq": doc_type}},
                                    {"numero": {"$eq": doc_number}}
                                ]
                            }
                            docs_with_scores = vectorstore.similarity_search_with_score(
                                query, k=k, filter=filter_dict
                            )
                            
                            # Estrategia 3: Si aún no encontramos, buscar por tipo solamente
                            if not docs_with_scores:
                                print(f"      ⚠️ Estrategia 3: Buscando solo por tipo: {doc_type}")
                                filter_dict = {"tipo_documento": {"$eq": doc_type}}
                                docs_with_scores = vectorstore.similarity_search_with_score(
                                    query, k=k*2, filter=filter_dict
                                )
                else:
                    # Búsqueda por similitud estándar
                    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
                
                documents = [doc for doc, score in docs_with_scores]
                scores = [score for doc, score in docs_with_scores]
                
            except Exception as search_error:
                print(f"      ⚠️ Error en búsqueda con filtros: {search_error}")
                # Fallback final: búsqueda sin filtros
                docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
                documents = [doc for doc, score in docs_with_scores]
                scores = [score for doc, score in docs_with_scores]
        else:
            # Ya tenemos documentos de una herramienta
            scores = [0.0] * len(documents)  # No hay scores si vienen de herramienta
        
        # LOGGING Y ACTUALIZACIÓN DEL ESTADO
        print(f"   ✓ {len(documents)} documentos recuperados")
        for i, (doc, score) in enumerate(zip(documents, scores), 1):
            doc_id = doc.metadata.get("id_documento", "N/A")
            tipo = doc.metadata.get("tipo_documento", "N/A")
            if score > 0:
                print(f"      {i}. {doc_id} ({tipo}) - score: {score:.4f}")
            else:
                print(f"      {i}. {doc_id} ({tipo}) - [herramienta]")
        
        state["documents"] = documents
        state["metadata"]["retrieval_scores"] = scores
        state["metadata"]["num_retrieved"] = len(documents)
        state["metadata"]["used_filter"] = filter_dict is not None
        if tool_results:
            state["metadata"]["tool_executed"] = tool_results.get("tool_used")
        
    except Exception as e:
        print(f"   ⚠️ Error en recuperación: {e}")
        import traceback
        traceback.print_exc()
        state["documents"] = []
        state["metadata"]["retrieval_error"] = str(e)
    
    return state


def generate_node(state: RAGState) -> RAGState:
    """
    Genera una respuesta usando el contexto recuperado.
    """
    query = state["query"]
    documents = state.get("documents", [])
    classification = state.get("classification", "general")
    
    print(f"\n✍️  GENERANDO RESPUESTA")
    
    if not documents:
        state["answer"] = "Lo siento, no encontré información relevante para responder tu consulta."
        print("   ⚠️ No hay documentos para generar respuesta")
        return state
    
    try:
        llm = init_groq_llm()
        
        # OPTIMIZACIÓN: Para comparaciones, limitar la cantidad de contexto
        tool_results = state.get("tool_results")
        is_comparison = tool_results and tool_results.get("tool_used") == "compare_documents"
        
        # Construir contexto de manera optimizada
        if is_comparison:
            # Para comparaciones, limitar documentos a los 10 más relevantes
            limited_docs = documents[:10]
            # Truncar cada documento a 1000 caracteres para comparaciones
            context = "\n\n---\n\n".join([
                f"Documento {i+1} ({doc.metadata.get('id_documento', 'N/A')}):\n{doc.page_content[:1000]}{'...' if len(doc.page_content) > 1000 else ''}"
                for i, doc in enumerate(limited_docs)
            ])
            print(f"   📊 Contexto optimizado para comparación: {len(limited_docs)} docs, ~{len(context)} chars")
        else:
            # Para otras consultas, usar contexto normal
            context = "\n\n---\n\n".join([
                f"Documento {i+1} ({doc.metadata.get('id_documento', 'N/A')}):\n{doc.page_content}"
                for i, doc in enumerate(documents)
            ])
        
        # Agregar resultados de tools si existen
        tool_info = ""
        if tool_results:
            tool_used = tool_results.get("tool_used", "N/A")
            tool_info = f"\n\n{'='*60}\nHERRAMIENTA EJECUTADA: {tool_used}\n{'='*60}\n"
            
            # Agregar información específica según la herramienta
            if tool_used == "compare_documents":
                comparison = tool_results.get("comparison_result", {})
                tool_info += f"\nComparación entre documentos:\n"
                tool_info += f"- Documento 1: {comparison.get('documento1', {}).get('id', 'N/A')}\n"
                tool_info += f"  Fragmentos encontrados: {comparison.get('documento1', {}).get('fragmentos_encontrados', 0)}\n"
                tool_info += f"- Documento 2: {comparison.get('documento2', {}).get('id', 'N/A')}\n"
                tool_info += f"  Fragmentos encontrados: {comparison.get('documento2', {}).get('fragmentos_encontrados', 0)}\n"
                tool_info += f"- Tema de comparación: {comparison.get('tema_comparacion', 'N/A')}\n"
                
                # Agregar nota de optimización si existe
                if comparison.get('optimizacion'):
                    opt = comparison['optimizacion']
                    tool_info += f"\n⚙️ Optimización aplicada:\n"
                    tool_info += f"  - Max {opt.get('fragmentos_por_documento', 'N/A')} fragmentos por documento\n"
                    tool_info += f"  - Max {opt.get('caracteres_maximos_por_fragmento', 'N/A')} caracteres por fragmento\n"
            
            elif tool_used == "extract_specific_article":
                tool_info += f"\nArtículo específico extraído:\n"
                tool_info += f"- Documento: {tool_results.get('doc_id', 'N/A')}\n"
                tool_info += f"- Artículo número: {tool_results.get('article_number', 'N/A')}\n"
            
            elif tool_used == "calculate_prestaciones_sociales":
                tool_info += f"\nCálculo de prestaciones sociales:\n"
                if tool_results.get("requires_user_input"):
                    tool_info += f"- {tool_results.get('message', 'Información incompleta')}\n"
                    if tool_results.get('salario_detectado'):
                        tool_info += f"- Salario detectado: ${tool_results.get('salario_detectado'):,.2f}\n"
            
            elif tool_used == "search_by_year_range":
                tool_info += f"\nBúsqueda por rango de años:\n"
                tool_info += f"- Desde: {tool_results.get('start_year', 'N/A')}\n"
                tool_info += f"- Hasta: {tool_results.get('end_year', 'N/A')}\n"
            
            elif tool_used == "resume_document":
                resume_data = tool_results.get("resume_result", {})
                tool_info += f"\nResumen de documento:\n"
                tool_info += f"- Documento: {tool_results.get('doc_id', 'N/A')}\n"
                tool_info += f"- Título: {resume_data.get('titulo', 'Sin título')}\n"
                tool_info += f"- Tipo: {resume_data.get('tipo_documento', 'Desconocido')}\n"
                tool_info += f"- Año: {resume_data.get('año', 'No especificado')}\n"
                tool_info += f"- Fragmentos encontrados: {resume_data.get('fragmentos_encontrados', 0)}\n"
            
            elif tool_used == "search_by_document_type":
                doc_id_parts = [tool_results.get('doc_type', '')]
                if tool_results.get('doc_number'):
                    doc_id_parts.append(tool_results.get('doc_number'))
                if tool_results.get('doc_year'):
                    doc_id_parts.append(tool_results.get('doc_year'))
                tool_info += f"\nBúsqueda de documento específico:\n"
                tool_info += f"- ID: {'_'.join(doc_id_parts)}\n"
            
            tool_info += f"{'='*60}\n"
        
        # Prompt mejorado según clasificación y herramientas
        if tool_results and tool_results.get("tool_used") == "compare_documents":
            system_prompt = """Eres un experto en derecho laboral colombiano especializado en análisis comparativo.

TAREA: Compara los documentos legales proporcionados enfocándote en similitudes y diferencias.

Reglas breves:
1. Identifica puntos clave de cada documento sobre el tema
2. Señala diferencias y coincidencias específicas
3. Cita ambos documentos
4. Sé conciso pero preciso
5. Usa lenguaje claro"""
        elif tool_results and tool_results.get("tool_used") == "extract_specific_article":
            system_prompt = """Eres un experto en derecho laboral colombiano. Tu trabajo es responder preguntas basándote ÚNICAMENTE en el contexto proporcionado.

Reglas:
1. Estás extrayendo un ARTÍCULO ESPECÍFICO - proporciona su contenido completo
2. Explica brevemente qué establece este artículo
3. Cita el documento legal con precisión
4. Sé claro, preciso y profesional
5. Usa lenguaje accesible para el usuario"""
        elif tool_results and tool_results.get("tool_used") == "calculate_prestaciones_sociales":
            system_prompt = """Eres un experto en derecho laboral colombiano. Tu trabajo es responder preguntas basándote ÚNICAMENTE en el contexto proporcionado.

Reglas:
1. Estás ayudando con un CÁLCULO de prestaciones sociales
2. Cita las leyes y fórmulas relevantes del contexto
3. Si falta información para el cálculo, indícalo claramente
4. Sé claro, preciso y profesional
5. Usa lenguaje accesible para el usuario"""
        elif tool_results and tool_results.get("tool_used") == "resume_document":
            system_prompt = """Eres un experto en derecho laboral colombiano. Tu tarea es generar un resumen DETALLADO y COMPLETO de un documento legal.

INSTRUCCIONES CRÍTICAS PARA EL RESUMEN:
1. Lee TODO el contenido del documento proporcionado
2. EXTRAE los puntos clave y principales del documento
3. Organiza el resumen en secciones lógicas con subtítulos
4. INCLUYE:
   - Objetivo o propósito del documento
   - Contenido principal y disposiciones clave
   - Artículos o secciones significativas
   - Obligaciones o derechos establecidos
   - Vigencia y fechas relevantes
   - Cualquier excepción o aclaración importante
5. Usa un estilo profesional pero accesible
6. El resumen debe ser ÚTIL y INFORMATIVO para alguien que necesita entender rápidamente qué dice el documento
7. NO escribas frases genéricas como "el documento proporciona información" - SÉ ESPECÍFICO
8. Cita artículos o números de sección cuando sea relevante"""
        elif tool_results and tool_results.get("tool_used") == "search_by_year_range":
            system_prompt = """Eres un experto en derecho laboral colombiano especializado en búsquedas temporales.

TAREA: Responder sobre normativa publicada en un rango de años específico.

Reglas:
1. Lista los documentos encontrados en el rango solicitado
2. Para cada documento, indica: tipo, número, año y tema principal
3. Si el tema específico preguntado está en los documentos, resáltalo
4. Si no hay información sobre el tema específico, indica qué documentos se encontraron pero no tratan ese tema
5. Sé específico y útil"""
        else:
            system_prompt = """Eres un experto en derecho laboral colombiano. Tu trabajo es responder preguntas basándote ÚNICAMENTE en el contexto proporcionado.

Reglas:
1. Responde SOLO con información del contexto
2. Cita las leyes, decretos o sentencias específicas cuando sea relevante
3. Si la información no está en el contexto, di que no tienes esa información
4. Sé claro, preciso y profesional
5. Usa lenguaje accesible para el usuario"""

        # Crear prompts específicos según el tipo de herramienta
        if tool_results and tool_results.get("tool_used") == "compare_documents":
            # Prompt optimizado para comparaciones
            comparison = tool_results.get("comparison_result", {})
            user_prompt = f"""COMPARACIÓN DE DOCUMENTOS LEGALES:

Documento 1: {comparison.get('documento1', {}).get('id', 'N/A')}
Documento 2: {comparison.get('documento2', {}).get('id', 'N/A')}
Tema: {comparison.get('tema_comparacion', 'N/A')}

---FRAGMENTOS RELEVANTES---

{context}

---FIN DE FRAGMENTOS---

Pregunta: {query}

Responde comparando ambos documentos sobre el tema especificado. Sé conciso pero completo."""
        elif tool_results and tool_results.get("tool_used") == "resume_document":
            user_prompt = f"""A continuación se proporciona el contenido COMPLETO de un documento legal.

TAREA: Realiza un resumen DETALLADO y ORGANIZADO del documento.

Estado del documento:
- Tipo: {tool_results.get('doc_type', 'N/A')}
- Número: {tool_results.get('doc_number', 'N/A')}
- Año: {tool_results.get('doc_year', 'N/A') if tool_results.get('doc_year') else 'N/A'}

---CONTENIDO DEL DOCUMENTO---

{context}

---FIN DEL CONTENIDO---

Genera un resumen COMPLETO que incluya:
✓ Objetivo y propósito del documento
✓ Puntos clave y disposiciones principales
✓ Contenido detallado de las secciones importantes
✓ Derechos u obligaciones establecidas
✓ Vigencia y aplicabilidad
✓ Información adicional relevante

Recuerda: Sé ESPECÍFICO. No escribas frases genéricas. Usa la información real del documento."""
        elif tool_results and tool_results.get("tool_used") == "search_by_year_range":
            start_year = tool_results.get('start_year', 'N/A')
            end_year = tool_results.get('end_year', 'N/A')
            user_prompt = f"""BÚSQUEDA DE NORMATIVA POR RANGO DE AÑOS: {start_year} - {end_year}

Se encontraron los siguientes documentos publicados en este período:

---DOCUMENTOS ENCONTRADOS---

{context}

---FIN DE DOCUMENTOS---

Pregunta del usuario: {query}

INSTRUCCIONES:
1. Lista los documentos encontrados mencionando: tipo, número y año
2. Indica qué temas trata cada documento basándote en el contenido mostrado
3. Si algún documento trata específicamente el tema preguntado (por ejemplo "jornada laboral"), resáltalo
4. Si ningún documento trata el tema específico, menciona qué documentos se encontraron en ese período

Responde de manera informativa y útil."""
        else:
            user_prompt = f"""Contexto de normativa laboral colombiana:

{context}{tool_info}

---

Pregunta del usuario: {query}

Proporciona una respuesta clara y precisa basada en el contexto:"""

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        response = llm.invoke(messages)
        answer = response.content.strip()
        
        # Agregar citas de fuentes al final de la respuesta
        sources_text = "\n\n" + "="*60 + "\n"
        sources_text += "📚 **FUENTES DE INFORMACIÓN**\n"
        sources_text += "="*60 + "\n\n"
        
        # Recolectar fuentes únicas
        sources_seen = set()
        sources_list = []
        
        for i, doc in enumerate(documents, 1):
            doc_id = doc.metadata.get('id_documento', 'N/A')
            
            # Evitar duplicados
            if doc_id in sources_seen:
                continue
            sources_seen.add(doc_id)
            
            # Extraer información del documento
            tipo_doc = doc.metadata.get('tipo_documento', 'Documento')
            año = doc.metadata.get('año', 'N/A')
            titulo = doc.metadata.get('titulo', '')
            
            # Construir la cita
            source_entry = f"{i}. **{tipo_doc}** {doc_id}"
            if año != 'N/A':
                source_entry += f" ({año})"
            if titulo:
                source_entry += f"\n   {titulo}"
            
            sources_list.append(source_entry)
        
        # Agregar las fuentes a la respuesta
        if sources_list:
            sources_text += "\n".join(sources_list)
            answer = answer + sources_text
        
        print(f"   ✓ Respuesta generada ({len(answer)} caracteres)")
        print(f"   ✓ Fuentes incluidas: {len(sources_list)}")
        
        state["answer"] = answer
        state["metadata"]["generation_model"] = "groq-llama-3.1-70b"
        state["metadata"]["sources_count"] = len(sources_list)
        
    except Exception as e:
        print(f"   ⚠️ Error en generación (usando respuesta basada en documentos): {str(e)[:100]}")
        # Fallback: crear respuesta simple basada en documentos
        if documents:
            answer = f"""Según los documentos de normativa laboral colombiana encontrados:

{documents[0].page_content[:500]}...

Nota: Esta es una extracción directa del documento. Para una respuesta más elaborada, se requiere configurar las API keys.

{"="*60}
📚 **FUENTES DE INFORMACIÓN**
{"="*60}

"""
            # Agregar fuentes incluso en fallback
            sources_seen = set()
            for i, doc in enumerate(documents, 1):
                doc_id = doc.metadata.get('id_documento', 'N/A')
                if doc_id not in sources_seen:
                    sources_seen.add(doc_id)
                    tipo_doc = doc.metadata.get('tipo_documento', 'Documento')
                    año = doc.metadata.get('año', 'N/A')
                    answer += f"{i}. **{tipo_doc}** {doc_id}"
                    if año != 'N/A':
                        answer += f" ({año})"
                    answer += "\n"
        else:
            answer = "No se encontró información relevante en la base de datos de normativa laboral."
        
        state["answer"] = answer
        state["metadata"]["generation_error"] = str(e)
        state["metadata"]["generation_model"] = "fallback-document-based"
    
    return state


def verify_node(state: RAGState) -> RAGState:
    """
    Verifica la calidad y exactitud de la respuesta generada.
    """
    answer = state.get("answer", "")
    documents = state.get("documents", [])
    query = state["query"]
    
    print(f"\n✅ VERIFICANDO RESPUESTA")
    
    verification = {
        "has_answer": len(answer) > 0,
        "answer_length": len(answer),
        "num_sources": len(documents),
        "quality_score": 0.0,
    }
    
    try:
        llm = init_gemini_llm()
        
        # Verificación de calidad
        verification_prompt = f"""Evalúa la calidad de esta respuesta sobre normativa laboral colombiana.

Pregunta: {query}
Respuesta: {answer}

Evalúa según estos criterios (responde con un número de 0 a 100):
1. ¿La respuesta es relevante para la pregunta?
2. ¿La respuesta está basada en información legal válida?
3. ¿La respuesta es clara y comprensible?
4. ¿La respuesta cita fuentes específicas cuando es apropiado?

Responde SOLO con un número del 0 al 100 que represente la calidad general:"""

        response = llm.invoke(verification_prompt)
        try:
            quality_score = float(response.content.strip()) / 100.0
            quality_score = max(0.0, min(1.0, quality_score))
        except:
            quality_score = 0.7 if len(answer) > 100 else 0.5
        
        verification["quality_score"] = quality_score
        verification["verification_method"] = "gemini"
        
        # Categorizar calidad
        if quality_score >= 0.8:
            verification["quality_level"] = "excellent"
            print(f"   ✓ Calidad: Excelente ({quality_score:.2%})")
        elif quality_score >= 0.6:
            verification["quality_level"] = "good"
            print(f"   ✓ Calidad: Buena ({quality_score:.2%})")
        else:
            verification["quality_level"] = "needs_improvement"
            print(f"   ⚠️ Calidad: Necesita mejorar ({quality_score:.2%})")
        
    except Exception as e:
        print(f"   ⚠️ Error en verificación: {e}")
        verification["verification_error"] = str(e)
        verification["quality_score"] = 0.5
    
    state["verification"] = verification
    
    return state


def build_graph():
    """
    Construye y compila el grafo de LangGraph con 5 nodos y tools integradas.
    """
    graph = StateGraph(RAGState)

    # Agregar nodos
    graph.add_node("classify", classify_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    # Definir flujo con tools
    graph.set_entry_point("classify")
    graph.add_edge("classify", "tool_calling")
    graph.add_edge("tool_calling", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()
