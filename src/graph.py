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
from src.tools import AVAILABLE_TOOLS
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
    """
    query = state["query"]
    
    print(f"\n🔍 CLASIFICANDO: {query}")
    
    llm = init_gemini_llm()
    
    classification_prompt = f"""Clasifica la siguiente consulta laboral en UNA de estas categorías:

1. legal_specific: Pregunta sobre una ley, decreto o sentencia específica
2. procedural: Pregunta sobre procedimientos, trámites o pasos a seguir
3. general: Pregunta general sobre derechos, obligaciones o conceptos laborales
4. calculation: Pregunta que requiere cálculos (liquidaciones, pagos, etc.)

Consulta: "{query}"

Responde SOLO con el nombre de la categoría (sin explicaciones):"""

    try:
        response = llm.invoke(classification_prompt)
        classification = response.content.strip().lower()
        
        # Validar clasificación
        valid_classifications = ["legal_specific", "procedural", "general", "calculation"]
        if classification not in valid_classifications:
            classification = "general"
        
        print(f"   ✓ Clasificación: {classification}")
        
    except Exception as e:
        print(f"   ⚠️ Error en clasificación (usando clasificación simple): {str(e)[:100]}")
        # Clasificación simple basada en palabras clave
        query_lower = query.lower()
        if any(word in query_lower for word in ["calcular", "liquidar", "cuánto", "pagar"]):
            classification = "calculation"
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
    """
    query = state["query"]
    classification = state.get("classification", "general")
    
    print(f"\n🔧 EVALUANDO HERRAMIENTAS ESPECIALIZADAS")
    
    tool_results = None
    
    # Si es una consulta de cálculo, usar la calculadora
    if classification == "calculation":
        # Detectar si es cálculo de prestaciones
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
                        "message": "Necesito más información para el cálculo"
                    }
                    print(f"      Salario detectado: ${salario:,.2f}")
                except:
                    pass
    
    # Si menciona un tipo específico de documento
    doc_type_match = re.search(r'(ley|decreto|sentencia)\s+(\d+)', query, re.IGNORECASE)
    if doc_type_match:
        doc_type = doc_type_match.group(1).upper()
        doc_num = doc_type_match.group(2)
        print(f"   ✓ Detectado: Búsqueda por documento específico - {doc_type} {doc_num}")
        tool_results = {
            "tool_used": "search_by_document_type",
            "doc_type": doc_type,
            "doc_number": doc_num
        }
    
    # Si menciona un rango de años
    year_match = re.findall(r'(19|20)\d{2}', query)
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
    """
    query = state["query"]
    classification = state.get("classification", "general")
    
    print(f"\n📚 RECUPERANDO DOCUMENTOS")
    
    try:
        # Cargar índice
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "normativa_laboral")
        
        embedding_fn = init_embeddings()
        vectorstore = load_chroma_index(persist_dir, embedding_fn, collection_name)
        
        # Determinar número de documentos a recuperar según clasificación
        k = 5 if classification == "legal_specific" else 3
        
        # Búsqueda por similitud
        docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)
        
        documents = [doc for doc, score in docs_with_scores]
        scores = [score for doc, score in docs_with_scores]
        
        print(f"   ✓ {len(documents)} documentos recuperados")
        for i, (doc, score) in enumerate(zip(documents, scores), 1):
            doc_id = doc.metadata.get("id_documento", "N/A")
            print(f"      {i}. {doc_id} (score: {score:.4f})")
        
        state["documents"] = documents
        state["metadata"]["retrieval_scores"] = scores
        state["metadata"]["num_retrieved"] = len(documents)
        
    except Exception as e:
        print(f"   ⚠️ Error en recuperación: {e}")
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
        
        # Construir contexto
        context = "\n\n---\n\n".join([
            f"Documento {i+1} ({doc.metadata.get('id_documento', 'N/A')}):\n{doc.page_content}"
            for i, doc in enumerate(documents)
        ])
        
        # Agregar resultados de tools si existen
        tool_info = ""
        tool_results = state.get("tool_results")
        if tool_results:
            tool_info = f"\n\nHerramienta utilizada: {tool_results.get('tool_used', 'N/A')}\nResultados: {tool_results}\n"
        
        # Prompt mejorado según clasificación
        system_prompt = """Eres un experto en derecho laboral colombiano. Tu trabajo es responder preguntas basándote ÚNICAMENTE en el contexto proporcionado.

Reglas:
1. Responde SOLO con información del contexto
2. Cita las leyes, decretos o sentencias específicas cuando sea relevante
3. Si la información no está en el contexto, di que no tienes esa información
4. Sé claro, preciso y profesional
5. Usa lenguaje accesible para el usuario"""

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
        
        print(f"   ✓ Respuesta generada ({len(answer)} caracteres)")
        
        state["answer"] = answer
        state["metadata"]["generation_model"] = "groq-llama-3.1-70b"
        
    except Exception as e:
        print(f"   ⚠️ Error en generación (usando respuesta basada en documentos): {str(e)[:100]}")
        # Fallback: crear respuesta simple basada en documentos
        if documents:
            answer = f"""Según los documentos de normativa laboral colombiana encontrados:

{documents[0].page_content[:500]}...

Fuente: {documents[0].metadata.get('id_documento', 'Documento legal colombiano')}

Nota: Esta es una extracción directa del documento. Para una respuesta más elaborada, se requiere configurar las API keys."""
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
