"""
LangGraph workflow para RAG de normativa laboral colombiana.

Cada nodo es una etapa del pipeline; el estado RAGState fluye entre ellos sin ramificaciones condicionales.

Flujo:
1. Classify: Clasifica la consulta del usuario (LLM)
2. Tool Calling: Selecciona herramientas especializadas mediante routing dirigido por LLM
3. Retrieve: Recupera documentos relevantes de ChromaDB
4. Generate: Genera respuesta usando el contexto
5. Verify: Verifica la calidad y exactitud de la respuesta
"""
from typing import TypedDict, List, Dict, Any, Optional
from langchain_core.documents import Document

from langgraph.graph import END, StateGraph

from src.config import init_embeddings, init_groq_llm, init_verification_llm
from src.generation_metrics import evaluate_generation_metrics
from src.ontology.kg_agent import KGAgent
from src.retrieval_metrics import evaluate_query_at_ks
from src.tools import ROUTING_TOOLS
from src.vectorstore import load_chroma_index
from src.query_transformer import transform_query
import os
import re


class RAGState(TypedDict):
    # Estado compartido e inmutable entre nodos. LangGraph lo pasa por referencia entre ejecuciones.
    # tool_results actúa como canal de comunicación entre tool_calling y retrieve/generate.
    # metadata acumula trazabilidad (modelo usado, scores, errores) sin afectar el flujo principal.
    query: str  # Consulta original del usuario
    classification: str  # Clasificación de la consulta
    query_type: Optional[str]  # Tipo de transformación: "hyde", "decomposition", "multi_query"
    transformed_queries: Optional[List[str]]  # Consultas transformadas/generadas
    documents: List[Document]  # Documentos recuperados
    tool_results: Optional[Dict[str, Any]]  # Resultados de tools ejecutadas
    kg_results: Optional[List[Dict[str, Any]]]  # Resultados estructurados desde GraphDB
    answer: str  # Respuesta generada
    verification: Dict[str, Any]  # Resultados de verificación
    metadata: Dict[str, Any]  # Metadata adicional


def classify_node(state: RAGState) -> RAGState:
    """
    Clasifica la consulta del usuario para determinar el tipo de pregunta.
    
    Categorías:
    - legal_specific: Pregunta sobre normativa específica
    - procedural: Pregunta sobre procedimientos o trámites
    - general_laboral: Pregunta general sobre derechos, obligaciones o conceptos laborales
    - general: Pregunta general que no tiene que ver con el ámbito laboral o es un saludo/conversación básica
    - calculation: Pregunta que requiere cálculos
    - resume: Pregunta que requiere un resumen de un documento
    """
    query = state["query"]
    
    print(f"\n[CLASSIFY] {query}")
    
    llm = init_groq_llm(temperature=0)
    
    classification_prompt = f"""Clasifica la siguiente consulta en UNA de estas categorías:

1. legal_specific: Pregunta sobre una ley, decreto o sentencia específica
2. procedural: Pregunta sobre procedimientos, trámites o pasos a seguir
3. general_laboral: Pregunta general sobre derechos, obligaciones o conceptos laborales
4. calculation: Pregunta que requiere cálculos (liquidaciones, pagos, etc.)
5. resume: Pregunta que requiere un resumen de un documento específico
6. general: Pregunta general que no tiene que ver con el ámbito laboral o es un saludo/conversación básica

Consulta: "{query}"

Responde SOLO con el nombre de la categoría (sin explicaciones):"""

    try:
        response = llm.invoke(classification_prompt)
        classification = response.content.strip().lower()
        
        # Validar clasificación
        valid_classifications = ["legal_specific", "procedural", "general_laboral", "general", "calculation", "resume"]
        if classification not in valid_classifications:
            classification = "general"
        
        print(f"   [OK] Clasificacion: {classification}")
        
    except Exception as e:
        print(f"   [WARN] Error en clasificacion (fallback simple): {str(e)[:100]}")
        # Clasificación simple basada en palabras clave
        # Fallback por keywords cuando el LLM principal falla o no está disponible.
        # El orden de los condicionales importa: "calculation" y "resume" se evalúan antes
        # que "legal_specific" para evitar colisiones con palabras como "ley" en contextos de cálculo.
        query_lower = query.lower()
        if any(word in query_lower for word in ["calcular", "liquidar", "cuánto", "pagar"]):
            classification = "calculation"
        elif any(word in query_lower for word in ["resumen", "resume", "resumir"]):
            classification = "resume"
        elif any(word in query_lower for word in ["ley", "decreto", "artículo", "sentencia"]):
            classification = "legal_specific"
        elif any(word in query_lower for word in ["cómo", "procedimiento", "trámite", "pasos"]):
            classification = "procedural"
        elif any(word in query_lower for word in ["trabajo", "laboral", "empleado", "empleador", "contrato", "salario", "despido", "renuncia", "vacaciones", "jornada"]):
            classification = "general_laboral"
        else:
            classification = "general"
        print(f"   [OK] Clasificacion (fallback): {classification}")
    
    state["classification"] = classification
    # Conserva metadata existente para no perder trazas o ground truth inyectado
    # desde scripts de evaluacion externos.
    state.setdefault("metadata", {})["classification_method"] = "groq_llama_3_3_70b_versatile"
    
    return state


def query_transform_node(state: RAGState) -> RAGState:
    """
    Transforma la consulta aplicando HyDE o Query Decomposition.
    
    Estrategias:
    - HyDE: Para preguntas cortas o ambiguas, genera un documento hipotético
    - Query Decomposition: Para consultas complejas, divide en sub-consultas
    - Multi-Query: Genera múltiples variaciones de la pregunta
    
    Este nodo se salta para consultas generales.
    """
    query = state["query"]
    classification = state.get("classification", "general")
    
    # Saltar transformación para consultas generales
    if classification in ["general"]:
        print(f"\n[TRANSFORM] Omitida (consulta general)")
        state["query_type"] = "none"
        state["transformed_queries"] = [query]
        state.setdefault("metadata", {})["query_transform_skipped"] = True
        return state
    
    try:
        print(f"\n[TRANSFORM] Iniciada")
        
        # Cargar vectorstore para transformación
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "normativa_laboral")
        embedding_fn = init_embeddings()
        vectorstore = load_chroma_index(persist_dir, embedding_fn, collection_name)
        
        # Inicializar LLM para transformación
        llm = init_groq_llm(temperature=0.1)
        
        # Aplicar transformación de consulta
        result = transform_query(
            question=query,
            llm=llm,
            vectorstore=vectorstore,
            k=4
        )
        
        # Actualizar estado con resultados
        state["query_type"] = result["query_type"]
        state["transformed_queries"] = result["transformed_queries"]
        
        # Guardar metadata de la transformación
        state.setdefault("metadata", {})["query_transform"] = result["metadata"]
        
        print(f"   [OK] Tipo: {state['query_type']}")
        print(f"   [OK] Consultas: {len(state['transformed_queries'])}")
        
    except Exception as e:
        print(f"   [WARN] Error en transformacion: {str(e)[:100]}")
        # Fallback: usar consulta original sin transformación
        state["query_type"] = "fallback"
        state["transformed_queries"] = [query]
        state.setdefault("metadata", {})["query_transform_error"] = str(e)
    
    return state


def tool_calling_node(state: RAGState) -> RAGState:
    """
    Determina si se necesita ejecutar alguna herramienta especializada.
    Implementa routing de tipo ReAct usando `llm.bind_tools(tools)` para que
    el modelo seleccione la herramienta adecuada a partir de la lista disponible.
    """
    query = state["query"]
    classification = state.get("classification", "general")
    
    print(f"\n🔧 EVALUANDO HERRAMIENTAS ESPECIALIZADAS (ReAct + bind_tools)")
    
    tool_results = None
    
    try:
        import json as _json

        llm = init_groq_llm(temperature=0)
        llm_with_tools = llm.bind_tools(ROUTING_TOOLS)

        routing_prompt = f"""Eres un agente router de herramientas para normativa laboral colombiana.

Tu tarea es decidir si debes llamar UNA herramienta de la lista disponible.
Si ninguna aplica, responde en texto corto: "none" y NO llames herramientas.

Reglas de selección:
- Si la clasificación es "resume", usa resume_document.
- Si la clasificación es "calculation" y habla de prestaciones/cesantías/prima/liquidación, usa calculate_prestaciones_sociales.
- Si pide un artículo concreto de una ley/decreto, usa extract_specific_article.
- Si pide comparar dos documentos, usa compare_documents.
- Si menciona una ley, decreto o sentencia específica (con número), usa search_by_document_type.
- Si menciona dos o más años como rango temporal, usa search_by_year_range.

Notas de argumentos:
- No incluyas `vectorstore`; ese parámetro lo inyecta el sistema posteriormente.
- Para sentencias usa formato doc_number tipo C200 (prefijo+número).
- doc_id sigue el formato TIPO_NUMERO o TIPO_NUMERO_AÑO.
- Si detectas salario, días trabajados o años de servicio, inclúyelos.

Clasificación actual: {classification}
Consulta del usuario: "{query}"""

        response = llm_with_tools.invoke(routing_prompt)
        tool_calls = getattr(response, "tool_calls", None) or []

        if tool_calls:
            first_call = tool_calls[0]
            tool_name = first_call.get("name", "none")
            params = first_call.get("args", {}) or {}
            print(f"   🤖 Agente ReAct seleccionó: {tool_name}")
            if params:
                print(f"      Parámetros: {_json.dumps(params, ensure_ascii=False)}")
        else:
            tool_name = "none"
            params = {}
            print("   🤖 Agente ReAct no seleccionó herramienta")
        
        # Construir tool_results según la herramienta seleccionada por el LLM
        if tool_name == "resume_document":
            doc_type = str(params.get("doc_type", "")).upper()
            doc_number = str(params.get("doc_number", ""))
            doc_year = params.get("doc_year")
            doc_id = params.get("doc_id")
            
            # Construir doc_id si el LLM no lo proporcionó
            if not doc_id:
                if doc_year:
                    doc_id = f"{doc_type}_{doc_number}_{doc_year}"
                else:
                    doc_id = f"{doc_type}_{doc_number}"
            
            tool_results = {
                "tool_used": "resume_document",
                "doc_id": doc_id,
                "doc_type": doc_type,
                "doc_number": doc_number,
                "doc_year": str(doc_year) if doc_year else None
            }
        
        elif tool_name == "calculate_prestaciones_sociales":
            # Extraer parámetros del LLM; usar defaults si no se proporcionan
            salary = params.get("salario_detectado")
            dias = params.get("dias_trabajados")
            años = params.get("años_servicio")
            
            # Defaults: contrato a término fijo, 30 días, 1 año de servicio
            defaults_used = []
            try:
                salary_val = float(salary) if salary is not None else None
            except (ValueError, TypeError):
                salary_val = None
            
            try:
                dias_val = int(dias) if dias is not None else 30
                if dias is None:
                    defaults_used.append("dias_trabajados=30")
            except (ValueError, TypeError):
                dias_val = 30
                defaults_used.append("dias_trabajados=30")
            
            try:
                años_val = float(años) if años is not None else 1.0
                if años is None:
                    defaults_used.append("años_servicio=1.0")
            except (ValueError, TypeError):
                años_val = 1.0
                defaults_used.append("años_servicio=1.0")
            
            tool_results = {
                "tool_used": "calculate_prestaciones_sociales",
                "salario_detectado": salary_val,
                "dias_trabajados": dias_val,
                "años_servicio": años_val,
                "defaults_used": defaults_used,
            }
            
            if salary_val is not None:
                print(f"      [OK] Salario: ${salary_val:,.2f}")
            else:
                print(f"      [INFO] Salario: sera inferido del corpus")
            print(f"      [OK] Dias: {dias_val}")
            print(f"      [OK] Anos: {anos_val}")
            if defaults_used:
                print(f"      [INFO] Valores por defecto: {', '.join(defaults_used)}")
        
        elif tool_name == "extract_specific_article":
            doc_type = str(params.get("doc_type", "")).upper()
            doc_number = str(params.get("doc_number", ""))
            doc_year = params.get("doc_year")
            doc_id = params.get("doc_id")
            article_number = str(params.get("article_number", ""))
            
            if not doc_id:
                if doc_year:
                    doc_id = f"{doc_type}_{doc_number}_{doc_year}"
                else:
                    doc_id = f"{doc_type}_{doc_number}"
            
            tool_results = {
                "tool_used": "extract_specific_article",
                "doc_id": doc_id,
                "article_number": article_number,
                "doc_type": doc_type,
                "doc_number": doc_number,
                "doc_year": str(doc_year) if doc_year else None
            }
        
        elif tool_name == "compare_documents":
            doc_id1 = str(params.get("doc_id1", ""))
            doc_id2 = str(params.get("doc_id2", ""))
            topic = str(params.get("topic", "contenido general"))
            
            tool_results = {
                "tool_used": "compare_documents",
                "doc_id1": doc_id1,
                "doc_id2": doc_id2,
                "topic": topic
            }
        
        elif tool_name == "search_by_document_type":
            doc_type = str(params.get("doc_type", "")).upper()
            doc_number = str(params.get("doc_number", ""))
            doc_year = params.get("doc_year")
            
            tool_results = {
                "tool_used": "search_by_document_type",
                "doc_type": doc_type,
                "doc_number": doc_number,
                "doc_year": str(doc_year) if doc_year else None
            }
        
        elif tool_name == "search_by_year_range":
            start_year = int(params.get("start_year", 0))
            end_year = int(params.get("end_year", 0))
            
            tool_results = {
                "tool_used": "search_by_year_range",
                "start_year": start_year,
                "end_year": end_year
            }
        
        # else: tool_name == "none" → tool_results queda None
    
    except Exception as e:
        print(f"   [WARN] Error en ReAct: {str(e)[:150]}")
        print(f"   [WARN] Usando fallback")
        tool_results = _tool_calling_fallback(query, classification)
    
    if tool_results:
        print(f"   [OK] Tool: {tool_results.get('tool_used', 'N/A')}")
    else:
        print("   • No se requieren herramientas especializadas")
    
    state["tool_results"] = tool_results
    return state


def _tool_calling_fallback(query: str, classification: str):
    """
    Fallback basado en patrones de texto para selección de herramientas.
    Se usa solo cuando el LLM routing falla.
    """
    tool_results = None
    
    # resume_document
    if classification == "resume":
        doc_match = re.search(
            r'(ley|decreto|sentencia|acto legislativo)\s+([CT])?-?(\d+)(?:\s+de\s+(\d{4}))?',
            query, re.IGNORECASE
        )
        if doc_match:
            if doc_match.group(2):
                doc_type = "SENTENCIA"
                doc_number = f"{doc_match.group(2).upper()}{doc_match.group(3)}"
            else:
                doc_type = doc_match.group(1).upper()
                doc_number = doc_match.group(3)
            doc_year = doc_match.group(4) if doc_match.group(4) else None
            doc_id = f"{doc_type}_{doc_number}_{doc_year}" if doc_year else f"{doc_type}_{doc_number}"
            tool_results = {
                "tool_used": "resume_document",
                "doc_id": doc_id, "doc_type": doc_type,
                "doc_number": doc_number, "doc_year": doc_year
            }
    
    # calculate_prestaciones_sociales
    if not tool_results and classification == "calculation":
        if any(w in query.lower() for w in ["prestaciones", "cesantías", "prima", "liquidación"]):
            defaults_used = []
            salary_val = None
            dias_val = 30
            años_val = 1.0
            
            salary_match = re.search(r'\$\s*([\d,\.]+)', query)
            if salary_match:
                try:
                    raw = salary_match.group(1).replace('.', '').replace(',', '')
                    salary_val = float(raw)
                except Exception:
                    pass
            
            dias_match = re.search(r'(\d+)\s*d[ií]as?\s*(?:de\s+)?(?:trabajo|trabajados)', query, re.IGNORECASE)
            if dias_match:
                dias_val = int(dias_match.group(1))
            else:
                defaults_used.append("dias_trabajados=30")
            
            años_match = re.search(r'(\d+(?:[.,]\d+)?)\s*años?\s*(?:de\s+)?servicio', query, re.IGNORECASE)
            if años_match:
                años_val = float(años_match.group(1).replace(',', '.'))
            else:
                defaults_used.append("años_servicio=1.0")
            
            tool_results = {
                "tool_used": "calculate_prestaciones_sociales",
                "salario_detectado": salary_val,
                "dias_trabajados": dias_val,
                "años_servicio": años_val,
                "defaults_used": defaults_used,
            }
    
    # extract_specific_article
    if not tool_results and classification != "resume":
        m = re.search(r'art[íi]culo\s+(\d+)\s+.*?\b(ley|decreto)\s+(\d+)(?:\s+de\s+(\d{4}))?', query, re.IGNORECASE)
        if m:
            doc_type = m.group(2).upper()
            doc_num = m.group(3)
            doc_year = m.group(4)
            doc_id = f"{doc_type}_{doc_num}_{doc_year}" if doc_year else f"{doc_type}_{doc_num}"
            tool_results = {
                "tool_used": "extract_specific_article",
                "doc_id": doc_id, "article_number": m.group(1),
                "doc_type": doc_type, "doc_number": doc_num, "doc_year": doc_year
            }
    
    # compare_documents
    if not tool_results:
        for pat in [
            r'compar[ae]r?\s+(?:la\s+)?(ley|decreto)\s+(\d+).*(?:con|y|vs).*(?:la\s+)?(ley|decreto)\s+(\d+)',
            r'diferencias?\s+entre\s+(?:la\s+)?(ley|decreto)\s+(\d+).*(?:y|con).*(?:la\s+)?(ley|decreto)\s+(\d+)'
        ]:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                tool_results = {
                    "tool_used": "compare_documents",
                    "doc_id1": f"{m.group(1).upper()}_{m.group(2)}",
                    "doc_id2": f"{m.group(3).upper()}_{m.group(4)}",
                    "topic": "contenido general"
                }
                break
    
    # search_by_document_type
    if not tool_results and classification != "resume":
        m = re.search(r'(ley|decreto)\s+(\d+)(?:\s+de\s+(\d{4}))?', query, re.IGNORECASE)
        if m:
            tool_results = {
                "tool_used": "search_by_document_type",
                "doc_type": m.group(1).upper(), "doc_number": m.group(2),
                "doc_year": m.group(3) if m.group(3) else None
            }
    
    # search_by_year_range
    if not tool_results and classification != "resume":
        years = re.findall(r'\b(?:19|20)\d{2}\b', query)
        if len(years) >= 2:
            ys = sorted(int(y) for y in years)
            tool_results = {
                "tool_used": "search_by_year_range",
                "start_year": ys[0], "end_year": ys[-1]
            }
    
    return tool_results


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
        # Selección dinámica de `k` utilizando un LLM (Groq) implementado en `select_dynamic_k`
        # k no es fijo: se delega a select_dynamic_k (Groq) para ajustarlo según complejidad de la consulta.
        # Esto balancea precisión (k pequeño para artículos específicos) vs. cobertura (k alto para resúmenes).
                
        from src.tools import select_dynamic_k
        k = select_dynamic_k(query, classification, tool_results, min_k=1, max_k=10)
        # El valor se guarda en metadata para trazabilidad desde la UI de Streamlit.
        state.setdefault("metadata", {})["retrieval_k"] = k
        
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
                    print(f"      [OK] Articulo {article_number} extraido")
                else:
                    print(f"      [WARN] Articulo {article_number} no encontrado en {doc_id}")
                    # Para extracción de artículos, si no se encuentra, devolver mensaje
                    # NO hacer búsqueda genérica porque el usuario pidió algo específico
                    # Se crea un Document de error en lugar de hacer búsqueda genérica.
                    # Decisión deliberada: si el usuario pidió un artículo específico, una respuesta genérica
                    # sería semánticamente incorrecta aunque devuelva texto relacionado.
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
                
                # El resultado de la comparación se inyecta de vuelta en state["tool_results"]
                # para que generate_node pueda construir el prompt con metadatos de ambos documentos.
                state["tool_results"]["comparison_result"] = comparison_result
                print(f"      [OK] Comparacion completada")
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
                
                print(f"      [OK] {len(documents)} docs creados")
            
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
                
                print(f"      [OK] {len(documents)} docs en rango {start_year}-{end_year}")
            
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
                print(f"      [OK] Contenido recuperado")
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
                    print(f"      [WARN] No hay fragmentos en base de datos")
            
            # 5. Ejecutar herramienta calculate_prestaciones_sociales
            elif tool_used == "calculate_prestaciones_sociales":
                from src.tools import calculate_prestaciones_sociales
                print(f"   🔧 Ejecutando: calculate_prestaciones_sociales")
                
                salary_val = tool_results.get("salario_detectado")
                dias_val = tool_results.get("dias_trabajados", 30)
                años_val = tool_results.get("años_servicio", 1.0)
                
                if salary_val is not None:
                    try:
                        calc_result = calculate_prestaciones_sociales.invoke({
                            "salario_mensual": float(salary_val),
                            "dias_trabajados": int(dias_val),
                            "años_servicio": float(años_val)
                        })
                        state["tool_results"]["calculation_result"] = calc_result
                        print(f"      [OK] Calculo exitoso")
                        print(f"         Cesantías: ${calc_result.get('cesantias', 0):,.2f}")
                        print(f"         Prima: ${calc_result.get('prima_servicios', 0):,.2f}")
                        print(f"         Vacaciones: ${calc_result.get('vacaciones', 0):,.2f}")
                        print(f"         Total: ${calc_result.get('total_prestaciones', 0):,.2f}")
                        
                        # Crear documento con los resultados del cálculo
                        defaults_used = tool_results.get("defaults_used", [])
                        defaults_note = ""
                        if defaults_used:
                            defaults_note = f"\n\nNota: Se usaron valores por defecto para: {', '.join(defaults_used)}. "
                            defaults_note += "Para obtener resultados más precisos, especifique: salario mensual, días trabajados y años de servicio."
                        
                        doc = Document(
                            page_content=(
                                f"Resultado del cálculo de prestaciones sociales:\n"
                                f"- Salario mensual: ${float(salary_val):,.2f}\n"
                                f"- Días trabajados: {dias_val}\n"
                                f"- Años de servicio: {años_val}\n\n"
                                f"Desglose:\n"
                                f"- Cesantías: ${calc_result.get('cesantias', 0):,.2f}\n"
                                f"- Intereses sobre cesantías (12%): ${calc_result.get('intereses_cesantias', 0):,.2f}\n"
                                f"- Prima de servicios: ${calc_result.get('prima_servicios', 0):,.2f}\n"
                                f"- Vacaciones: ${calc_result.get('vacaciones', 0):,.2f}\n"
                                f"- TOTAL PRESTACIONES: ${calc_result.get('total_prestaciones', 0):,.2f}"
                                f"{defaults_note}"
                            ),
                            metadata={
                                "source": "calculate_prestaciones_sociales",
                                "tipo_documento": "CÁLCULO"
                            }
                        )
                        documents = [doc]
                    except Exception as calc_error:
                        print(f"      [WARN] Error en calculo: {calc_error}")
                else:
                    print(f"      [INFO] Salario no disponible - usara corpus")
            
            # 6. Herramienta search_by_document_type - mantener búsqueda por metadata
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
                    print(f"      [WARN] Error en filtros: {filter_error}")
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
                    # Búsqueda en cascada: ID exacto → tipo+número → solo tipo.
                    # Compensa inconsistencias en metadata de ChromaDB (documentos sin año indexado, por ejemplo).
                    # Cada degradación amplía el scope de búsqueda, priorizando encontrar algo antes que devolver vacío.
                    if not docs_with_scores and tool_results:
                        tool_used = tool_results.get("tool_used")
                        
                        if tool_used == "search_by_document_type":
                            doc_type = tool_results.get("doc_type")
                            doc_number = tool_results.get("doc_number")
                            
                            print(f"      [WARN] Estrategia 2: Buscando por tipo")
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
                                print(f"      [WARN] Estrategia 3: Buscando por tipo: {doc_type}")
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
                print(f"      [WARN] Error en busqueda: {search_error}")
                # Fallback final: búsqueda sin filtros
                docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
                documents = [doc for doc, score in docs_with_scores]
                scores = [score for doc, score in docs_with_scores]
        else:
            # Ya tenemos documentos de una herramienta
            # Los documentos de herramientas no tienen score de similitud coseno.
            # Se asigna 0.0 como placeholder para mantener la estructura homogénea del estado
            # y evitar errores en el logging y en la UI que espera pares (doc, score).
            scores = [0.0] * len(documents)  # No hay scores si vienen de herramienta
        
        # LOGGING Y ACTUALIZACIÓN DEL ESTADO
        print(f"   [OK] {len(documents)} docs recuperados")
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

        # -----------------------------
        # Métricas de retrieval (opcional)
        # -----------------------------
        # Para calcular Recall@k y Precision@k se requiere ground truth por consulta.
        # Si no se inyecta en metadata, registramos solo los recuperados y omitimos score.
        retrieved_doc_ids = [
            doc.metadata.get("id_documento")
            for doc in documents
            if doc.metadata.get("id_documento")
        ]
        state["metadata"]["retrieved_doc_ids"] = retrieved_doc_ids

        ground_truth_doc_ids = state.get("metadata", {}).get("ground_truth_doc_ids")
        retrieval_eval_ks = state.get("metadata", {}).get("retrieval_eval_ks", (3, 5, 10))

        if isinstance(ground_truth_doc_ids, list) and ground_truth_doc_ids:
            try:
                ks = tuple(int(k_item) for k_item in retrieval_eval_ks)
                retrieval_metrics = evaluate_query_at_ks(
                    retrieved_doc_ids=retrieved_doc_ids,
                    relevant_doc_ids=ground_truth_doc_ids,
                    ks=ks,
                )
                state["metadata"]["retrieval_metrics"] = retrieval_metrics
                state["metadata"]["retrieval_metrics_enabled"] = True

                print("   [MTR] Retrieval metrics calculadas")
                for k_item in ks:
                    k_metrics = retrieval_metrics.get(k_item, {})
                    rec = k_metrics.get("recall", 0.0)
                    pre = k_metrics.get("precision", 0.0)
                    print(f"      k={k_item} -> recall={rec:.3f} precision={pre:.3f}")
            except Exception as metric_error:
                state["metadata"]["retrieval_metrics_error"] = str(metric_error)
                print(f"   [WARN] Error calculando retrieval metrics: {metric_error}")
        else:
            state["metadata"]["retrieval_metrics_enabled"] = False
            state["metadata"]["retrieval_metrics_skipped_reason"] = (
                "ground_truth_doc_ids no proporcionado"
            )
            print("   [MTR] Retrieval metrics omitidas: falta ground truth")
        
    except Exception as e:
        print(f"   [WARN] Error en recuperacion: {e}")
        import traceback
        traceback.print_exc()
        state["documents"] = []
        state["metadata"]["retrieval_error"] = str(e)
    
    return state


def kg_retrieve_node(state: RAGState) -> RAGState:
    """
    Recupera contexto estructurado desde GraphDB usando un agente KG.

    Este nodo no reemplaza la recuperacion semantica de ChromaDB.
    Su objetivo es enriquecer el contexto con hechos RDF/SPARQL.
    """
    query = state["query"]
    classification = state.get("classification", "general")

    # Para consultas generales no laborales evitamos consultas al KG.
    if classification == "general":
        state.setdefault("metadata", {})["kg_skipped"] = True
        return state

    print(f"\n[KG] Recuperando contexto estructurado (GraphDB)")

    try:
        kg_agent = KGAgent()
        kg_payload = kg_agent.retrieve(query)

        state["kg_results"] = kg_payload.get("rows", [])

        # Convertir filas RDF en Document para reutilizar generate_node actual.
        kg_docs = kg_agent.as_documents(query)

        # Unir resultados: primero KG y luego semantico para priorizar hechos.
        existing_docs = state.get("documents", [])
        state["documents"] = kg_docs + existing_docs

        md = state.setdefault("metadata", {})
        md["kg_enabled"] = kg_payload.get("enabled", False)
        md["kg_rows"] = len(kg_payload.get("rows", []))
        md["kg_error"] = kg_payload.get("error")
        md["kg_query_used"] = kg_payload.get("sparql")

        if kg_payload.get("error"):
            print(f"   [WARN] Error KG: {kg_payload['error']}")
        else:
            print(f"   [OK] Filas KG: {len(kg_payload.get('rows', []))}")

    except Exception as e:
        print(f"   [WARN] Error en KG: {e}")
        state.setdefault("metadata", {})["kg_error"] = str(e)

    return state


def generate_node(state: RAGState) -> RAGState:
    """
    Genera una respuesta usando el contexto recuperado.
    """
    query = state["query"]
    documents = state.get("documents", [])
    classification = state.get("classification", "general")
    
    print(f"\n[GEN] Generando respuesta")
    
    if not documents and classification != "general":
        state["answer"] = "Lo siento, no encontré información relevante para responder tu consulta."
        print("   [WARN] Sin documentos para generar")
        return state
    
    try:
        llm = init_groq_llm()
        
        # OPTIMIZACIÓN: Para comparaciones, limitar la cantidad de contexto
        tool_results = state.get("tool_results")
        is_comparison = tool_results and tool_results.get("tool_used") == "compare_documents"
        
        # Construir contexto de manera optimizada
        # Las comparaciones duplican el volumen de contexto (dos documentos completos).
        # Se limita a 10 docs y 1000 caracteres por fragmento para no exceder el context window de Groq/Llama.
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
                if tool_results.get('salario_detectado'):
                    tool_info += f"- Salario: ${tool_results.get('salario_detectado'):,.2f}\n"
                tool_info += f"- Días trabajados: {tool_results.get('dias_trabajados', 30)}\n"
                tool_info += f"- Años de servicio: {tool_results.get('años_servicio', 1.0)}\n"
                calc_result = tool_results.get("calculation_result")
                if calc_result:
                    tool_info += f"\nResultados del cálculo:\n"
                    tool_info += f"- Cesantías: ${calc_result.get('cesantias', 0):,.2f}\n"
                    tool_info += f"- Intereses cesantías: ${calc_result.get('intereses_cesantias', 0):,.2f}\n"
                    tool_info += f"- Prima de servicios: ${calc_result.get('prima_servicios', 0):,.2f}\n"
                    tool_info += f"- Vacaciones: ${calc_result.get('vacaciones', 0):,.2f}\n"
                    tool_info += f"- TOTAL: ${calc_result.get('total_prestaciones', 0):,.2f}\n"
                defaults_used = tool_results.get("defaults_used", [])
                if defaults_used:
                    tool_info += f"\n⚠️ Valores por defecto usados: {', '.join(defaults_used)}\n"
                    tool_info += f"Para resultados más precisos, el usuario debe especificar: salario, días trabajados y años de servicio.\n"
            
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
        # Cada herramienta activa un system_prompt distinto con instrucciones específicas.
        # Esto es crítico: el mismo LLM necesita rol y restricciones diferentes
        # para resumir, comparar, extraer o calcular correctamente.
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
2. Presenta los resultados del cálculo de forma clara y organizada
3. Cita las leyes y fórmulas relevantes del contexto
4. Si se usaron valores por defecto, SIEMPRE incluye al final una nota clara indicando:
   "⚠️ Para obtener resultados más precisos, especifique: salario mensual, días trabajados, años de servicio y tipo de contrato."
5. Sé claro, preciso y profesional
6. Usa lenguaje accesible para el usuario"""
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
        elif classification == "general":
            system_prompt = """Eres un asistente útil y amable. Tu trabajo es responder preguntas generales o saludos de manera concisa."""
        elif classification == "general_laboral":
            system_prompt = """Eres un experto en derecho laboral colombiano. Tu trabajo es responder preguntas generales sobre normativa laboral basándote en el contexto proporcionado.

Reglas:
1. Responde SOLO con información del contexto
2. Cita las leyes, decretos o sentencias específicas cuando sea relevante
3. Si la información no está en el contexto, di que no tienes esa información
4. Sé claro, preciso y profesional
5. Usa lenguaje accesible para el usuario"""
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
        elif classification == "general":
            user_prompt = f"""Pregunta del usuario: {query}

Proporciona una respuesta clara y precisa:"""
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
        # sources_seen evita citar el mismo documento varias veces cuando k recuperó
        # múltiples chunks del mismo archivo (comportamiento normal en RAG con chunking).
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
        
        print(f"   [OK] Respuesta generada ({len(answer)} caracteres)")
        print(f"   [OK] Fuentes: {len(sources_list)}")
        
        state["answer"] = answer
        state["metadata"]["generation_model"] = "groq-llama-3.3-70b-versatile"
        state["metadata"]["sources_count"] = len(sources_list)
        
    except Exception as e:
        print(f"   [WARN] Error en generacion: {str(e)[:100]}")
        # Fallback: crear respuesta simple basada en documentos
        # Si el LLM falla, se extrae el primer chunk directamente de ChromaDB como respuesta cruda.
        # Se mantiene la sección de fuentes para preservar trazabilidad incluso en el caso degradado.
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
    classification = state.get("classification", "general")

    print(f"\n✅ VERIFICANDO RESPUESTA (detallado)")

    verification = {
        "has_answer": len(answer) > 0,
        "answer_length": len(answer),
        "num_sources": len(documents),
        "quality_score": 0.0,
        "supported_by_context": None,
        "unsupported_claims": [],
        "coherence_score": None,
        "completeness_score": None,
        "recommended_action": "accept",
    }

    # Si es una consulta general, no verificamos contra contexto
    if classification == "general":
        verification["quality_score"] = 1.0
        verification["quality_level"] = "excellent"
        verification["supported_by_context"] = True
        verification["coherence_score"] = 100
        verification["completeness_score"] = 100
        verification["verification_method"] = "skipped_for_general"
        state["verification"] = verification
        print("   [OK] Verificacion omitida (consulta general)")
        return state

    max_retries = 1
    attempt = 0

    # Construir extracto de contexto (limitado) para verificación
    context_pieces = []
    for i, doc in enumerate(documents[:5], 1):
        content = doc.page_content
        if len(content) > 1500:
            content = content[:1500] + "..."
        doc_id = doc.metadata.get("id_documento", f"doc_{i}")
        context_pieces.append(f"Documento {i} ({doc_id}):\n{content}")
    context_excerpt = "\n\n---\n\n".join(context_pieces) if context_pieces else "(sin contexto)"

    def _normalize_support_score(value: Any) -> float:
        """Normaliza supported_by_context a un score 0-100."""
        if value is None:
            return 0.0
        if isinstance(value, bool):
            return 100.0 if value else 0.0
        if isinstance(value, (int, float)):
            return max(0.0, min(100.0, float(value)))
        if isinstance(value, str):
            raw = value.strip().lower()
            if raw in {"true", "yes", "si", "sí"}:
                return 100.0
            if raw in {"false", "no"}:
                return 0.0
            try:
                parsed = float(raw.replace(",", "."))
                return max(0.0, min(100.0, parsed))
            except ValueError:
                return 0.0
        return 0.0

    while True:
        attempt += 1
        try:
            llm = init_verification_llm(temperature=0)

            verify_prompt = f"""Eres un verificador moderado en derecho laboral. Tu objetivo es evaluar si la respuesta es correcta, útil y está bien soportada.

Devuelve UNICAMENTE un JSON estricto (sin explicaciones)
con las siguientes claves:
 - supported_by_context: 0-100 (¿la respuesta está respaldada por el contexto proporcionado?)
 - unsupported_claims: lista de frases (puede estar vacía)
 - coherence_score: entero 0-100
 - completeness_score: entero 0-100 (¿la respuesta cumple con lo pedido?)
 - recommended_action: una de ["accept","regenerate","ask_clarification"]

Contexto (fragmentos extraídos):
{context_excerpt}

Respuesta generada:
{answer}

Pregunta original:
{query}

Instrucciones para el verificador (NO incluir en la salida):
 • Sé JUSTO pero EXIGENTE: evalúa si la respuesta está bien soportada en el contexto.
 • Si hay afirmaciones no soportadas o contradictoras, inclúyelas en 'unsupported_claims'.
 • completeness_score: evalúa si la respuesta responde completamente la pregunta. 60+ es bueno, <50 puede requerir regeneración.
 • supported_by_context debe bajar claramente (por ejemplo por cada afirmación no respaldada restar 10 al valor de 100) si hay afirmaciones no respaldadas.
 • Si la respuesta es INCOMPLETA o tiene ERRORES LEGALES, recomienda 'regenerate'.
 • Regenera solo si la calidad es claramente mejorable.
 Responde SOLO con JSON válido."""

            response = llm.invoke(verify_prompt)
            content = response.content.strip()

            import json
            try:
                parsed = json.loads(content)
            except Exception:
                # Intentar limpiar texto y extraer JSON entre llaves
                import re
                m = re.search(r"(\{.*\})", content, re.DOTALL)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                    except Exception as e:
                        raise e
                else:
                    raise ValueError("No se pudo parsear JSON de la respuesta de verificación")

            # Normalizar y guardar en verification
            support_score_raw = _normalize_support_score(parsed.get("supported_by_context"))
            verification["support_score"] = int(round(support_score_raw))
            verification["supported_by_context"] = support_score_raw >= 70.0
            verification["unsupported_claims"] = parsed.get("unsupported_claims", []) or []
            verification["coherence_score"] = int(parsed.get("coherence_score") or 0)
            verification["completeness_score"] = int(parsed.get("completeness_score") or 0)
            verification["recommended_action"] = parsed.get("recommended_action", "accept")
            verification["verification_method"] = "groq_openai_gpt_oss_120b_json"

            # Derivar quality_score a partir de coherence/completeness y soporte
            # Ponderación: soporte contextual (35%) + coherencia (35%) + completitud (30%).
            # El soporte usa escala continua 0-100 para capturar gradientes de respaldo contextual.
            coherence = verification["coherence_score"] / 100.0
            completeness = verification["completeness_score"] / 100.0
            support = support_score_raw / 100.0

            # Ponderación balanceada: soporte y coherencia importantes, completitud también
            quality_score = (0.35 * support) + (0.35 * coherence) + (0.3 * completeness)
            quality_score = max(0.0, min(1.0, quality_score))
            verification["quality_score"] = quality_score

            # Clasificar (umbrales moderados)
            # Umbrales definidos empíricamente: ≥0.72 excellent, ≥0.55 good, ≥0.35 needs_improvement.
            # Calibrados para dominio legal donde la precisión es más importante que la fluidez.
            if quality_score >= 0.72:
                verification["quality_level"] = "excellent"
            elif quality_score >= 0.55:
                verification["quality_level"] = "good"
            elif quality_score >= 0.35:
                verification["quality_level"] = "needs_improvement"
            else:
                verification["quality_level"] = "poor"

            print(f"   [OK] Verificacion: nivel={verification['quality_level']} score={quality_score:.2%} (intento {attempt})")

            # Acción recomendada: regenerar si el verificador lo pide y aún quedan intentos.
            # Decide si regenerar: respetar la recomendación del verificador y permitir regeneración en casos claros.
            # La regeneración no depende solo de recommended_action=="regenerate":
            # también exige quality_score < 0.55 o afirmaciones no soportadas con score < 0.65.
            # Esto evita regeneraciones innecesarias cuando el verificador es conservador.
            # max_retries=1 limita a un solo ciclo para no degradar la latencia percibida en Streamlit.
            should_regenerate = False
            if verification["recommended_action"] == "regenerate":
                # Regenerar si la calidad es baja (<0.55) o hay afirmaciones no soportadas significativas
                unsupported_count = len(verification.get("unsupported_claims", []) or [])
                if quality_score < 0.55 or (unsupported_count > 0 and quality_score < 0.65):
                    should_regenerate = True

            if should_regenerate and attempt <= max_retries:
                print(f"   [RETRY] Regenerando (intento {attempt}/{max_retries})...")
                # Incrementar contador de regeneraciones en metadata
                reg_attempts = state.get("metadata", {}).get("regeneration_attempts", 0)
                state.setdefault("metadata", {})["regeneration_attempts"] = reg_attempts + 1

                # Llamar a generate_node para regenerar.
                # Se llama a generate_node directamente desde verify_node en vez de redirigir el grafo.
                # Esto simplifica la arquitectura (grafo lineal sin ciclos) a costa de acoplamiento entre nodos.
                state = generate_node(state)
                # Actualizar answer y documents desde estado regenerado
                answer = state.get("answer", "")
                documents = state.get("documents", [])
                # reconstruir context_excerpt con nuevos documentos
                context_pieces = []
                for i, doc in enumerate(documents[:5], 1):
                    content = doc.page_content
                    if len(content) > 1500:
                        content = content[:1500] + "..."
                    doc_id = doc.metadata.get("id_documento", f"doc_{i}")
                    context_pieces.append(f"Documento {i} ({doc_id}):\n{content}")
                context_excerpt = "\n\n---\n\n".join(context_pieces) if context_pieces else "(sin contexto)"
                # Volver a intentar verificación
                continue

            # Si no pide regenerar o no quedan intentos, guardar y salir
            break

        except Exception as e:
            print(f"   ⚠️ Error en verificación detallada: {e}")
            # Fallback ligero: marcar score medio y aceptar parcialmente.
            # Si el verificador falla durante la verificación, se asigna quality_score=0.5 y recommended_action="accept".
            # Se acepta la respuesta sin verificar para no bloquear el pipeline por un fallo del verificador.
            verification["verification_error"] = str(e)
            verification["quality_score"] = 0.5
            verification["quality_level"] = "needs_improvement"
            verification["recommended_action"] = "accept"
            break

    state["verification"] = verification

    # -----------------------------
    # Métricas de generación (LLM-as-a-judge + LangSmith)
    # -----------------------------
    # Se ejecutan al final para evaluar la respuesta ya generada y registrar
    # trazabilidad en LangSmith con @traceable dentro del módulo de métricas.
    try:
        generation_metrics_enabled = state.get("metadata", {}).get(
            "generation_metrics_enabled", True
        )
        if generation_metrics_enabled and answer.strip():
            contexts_for_eval = [doc.page_content for doc in documents[:10]]
            print("   [MTR] Calculando métricas de generación")

            generation_metrics = evaluate_generation_metrics(
                query=query,
                contexts=contexts_for_eval,
                answer=answer,
            )

            state.setdefault("metadata", {})["generation_metrics"] = generation_metrics
            state["verification"]["faithfulness_score"] = generation_metrics[
                "faithfulness"
            ]["score"]
            state["verification"]["answer_relevance_score"] = generation_metrics[
                "answer_relevance"
            ]["score"]
            state["verification"]["generation_average_score"] = generation_metrics[
                "average_score"
            ]

            print(
                "   [MTR] Generación -> "
                f"faithfulness={generation_metrics['faithfulness']['score']:.3f}, "
                f"relevance={generation_metrics['answer_relevance']['score']:.3f}"
            )
        else:
            state.setdefault("metadata", {})["generation_metrics_skipped_reason"] = (
                "deshabilitadas o respuesta vacía"
            )
    except Exception as generation_metric_error:
        state.setdefault("metadata", {})["generation_metrics_error"] = str(
            generation_metric_error
        )
        print(f"   [WARN] Error en métricas de generación: {generation_metric_error}")

    return state


def build_graph():
    """
    Construye y compila el grafo de LangGraph con 6 nodos y tools integradas.
    """
    graph = StateGraph(RAGState)

    # Agregar nodos
    graph.add_node("classify", classify_node)
    graph.add_node("query_transform", query_transform_node)
    graph.add_node("tool_calling", tool_calling_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("kg_retrieve", kg_retrieve_node)
    graph.add_node("generate", generate_node)
    graph.add_node("verify", verify_node)

    # Definir flujo con tools
    # El grafo es completamente lineal con un nodo de transformación de consultas.
    # Nota: query_transform se ejecuta para todas las consultas no-generales.
    graph.set_entry_point("classify")
    
    # Edge condicional desde classify
    # Si es general, saltamos query_transform, tool_calling y retrieve, y vamos directo a generate
    graph.add_conditional_edges(
        "classify",
        lambda state: "generate" if state.get("classification") == "general" else "query_transform",
        {
            "generate": "generate",
            "query_transform": "query_transform"
        }
    )
    
    graph.add_edge("query_transform", "tool_calling")
    graph.add_edge("tool_calling", "retrieve")
    graph.add_edge("retrieve", "kg_retrieve")
    graph.add_edge("kg_retrieve", "generate")
    graph.add_edge("generate", "verify")
    graph.add_edge("verify", END)

    return graph.compile()
