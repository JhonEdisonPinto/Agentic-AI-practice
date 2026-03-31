"""
Aplicación Streamlit para el RAG de Normativa Laboral Colombiana.
Sistema de consulta inteligente con LangGraph y ChromaDB.
"""
# Capa de presentación exclusivamente. Toda la lógica RAG se delega a
# build_graph() + graph.invoke(). Este archivo no accede directamente
# a ChromaDB, embeddings ni LLMs.
import streamlit as st
import time
import os
from typing import Dict, Any
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import load_settings
from src.graph import build_graph


def parse_csv_tokens(raw_value: str) -> list[str]:
    """Convierte una cadena CSV en lista de tokens limpios."""
    if not raw_value:
        return []
    return [item.strip() for item in raw_value.split(",") if item.strip()]


def parse_eval_ks(raw_value: str, default_ks: tuple[int, ...] = (1, 3, 5)) -> list[int]:
    """Parsea ks de evaluación desde CSV, con fallback seguro."""
    ks: list[int] = []
    for token in parse_csv_tokens(raw_value):
        try:
            k = int(token)
            if k > 0:
                ks.append(k)
        except ValueError:
            continue

    # Deduplicar preservando orden
    unique_ks = list(dict.fromkeys(ks))
    return unique_ks or list(default_ks)


# Configuración de la página
# Debe ser la primera llamada a st.* del script; cualquier otra llamada
# previa lanza StreamlitAPIException.
st.set_page_config(
    page_title="RAG Normativa Laboral Colombiana",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def check_database():
    """Verifica si existe la base de datos ChromaDB."""
    # Verificación superficial: comprueba existencia del directorio y que no esté vacío.
    # No valida integridad de la colección ni que haya chunks indexados.

    chroma_dir = REPO_ROOT / "data" / "chroma"
    return chroma_dir.exists() and any(chroma_dir.glob("*"))


def initialize_session_state():
    """Inicializa el estado de la sesión."""
    # Verificar base de datos
    # Secuencia de arranque: 1) verificar DB → 2) construir grafo → 3) inicializar historial.
    # Si la DB no existe, st.stop() bloquea la app completa antes de cualquier render.

    if not check_database():
        st.error("❌ Base de datos no encontrada")
        st.info("""
        **Para desplegar correctamente:**
        
        1. **Opción Simple**: Sube la carpeta `data/` al repositorio
           ```bash
           # Descomenta en .gitignore las líneas de data/
           git add data/
           git commit -m "Add database for deployment"
           git push
           ```
        
        2. **Opción Avanzada**: Los PDFs deben estar en `src/corpus/` 
           y se generará automáticamente
        """)
        st.stop()
    
    # El grafo se construye una sola vez por sesión de navegador y se reutiliza
    # en cada consulta. load_settings() debe ejecutarse antes para que las
    # API keys estén disponibles cuando build_graph() las necesite.
    if "graph" not in st.session_state:
        with st.spinner("🔨 Construyendo grafo RAG..."):
            st.session_state.graph = build_graph()
    
    # Historial en memoria: crece sin límite durante la sesión, no se persiste a disco.
    if "history" not in st.session_state:
        st.session_state.history = []


def display_sidebar():
    """Muestra la barra lateral con información."""
    with st.sidebar:
        st.title("⚖️ RAG Laboral")
        

        st.markdown("---")
        st.subheader("🔧 Herramientas disponibles")
        st.markdown("""
        El sistema usa **routing dirigido por LLM** 
        para seleccionar automáticamente la herramienta 
        más adecuada:
        
        1. **Cálculo de prestaciones sociales**
        2. **Búsqueda por tipo de documento**
        3. **Búsqueda por rango de años**
        4. **Extracción de artículos específicos**
        5. **Comparación de documentos**
        6. **Resumen de documentos**
        """)
        
        st.markdown("---")
        st.subheader("⚙️ Recuperación Vectorial")
        retrieval_strategy = st.selectbox(
            "Estrategia",
            options=["similarity", "mmr"],
            index=0,
            help="similarity: top-k por similitud. mmr: maximiza relevancia y diversidad.",
        )

        mmr_fetch_k = st.number_input(
            "MMR fetch_k",
            min_value=5,
            max_value=200,
            value=20,
            step=5,
            disabled=retrieval_strategy != "mmr",
            help="Cantidad de candidatos iniciales antes de aplicar diversidad MMR.",
        )

        mmr_lambda_mult = st.slider(
            "MMR lambda",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            disabled=retrieval_strategy != "mmr",
            help="0 prioriza diversidad, 1 prioriza similitud.",
        )

        # Sincronizar con el runtime para que graph.py/tools.py lean la misma estrategia
        # sin usar variables de entorno globales (evita fugas entre sesiones).
        metadata = st.session_state.get("metadata") or {}
        metadata["RETRIEVAL_STRATEGY"] = str(retrieval_strategy)
        metadata["MMR_FETCH_K"] = str(int(mmr_fetch_k))
        metadata["MMR_LAMBDA_MULT"] = str(float(mmr_lambda_mult))
        st.session_state["metadata"] = metadata

        st.caption(
            f"Estrategia activa: {retrieval_strategy.upper()}"
            + (f" | fetch_k={int(mmr_fetch_k)} lambda={float(mmr_lambda_mult):.2f}" if retrieval_strategy == "mmr" else "")
        )

        st.markdown("---")
        st.subheader("📋 Tipos de consulta")
        st.markdown("""
        - 🏛️ **Normativa específica**: Leyes, decretos, sentencias
        - 📋 **Procedimientos**: Trámites y pasos a seguir
        - 💼 **General laboral**: Derechos y conceptos laborales (usa corpus)
        - 💬 **General**: Saludos y preguntas no laborales (respuesta directa)
        - 🧮 **Cálculos**: Liquidaciones, prestaciones
        - 📄 **Resumen**: Resumen de documentos
        """)


def display_response(result: Dict[str, Any]):
    """Muestra la respuesta del RAG de manera estructurada."""
    # Renderiza el RAGState completo tras una ejecución exitosa.
    # Estructura: respuesta principal → columna proceso + columna documentos
    #             → metadatos de recuperación → JSON técnico expandible.

    # Respuesta principal
    st.subheader("💬 Respuesta")
    answer = result.get("answer", "No se pudo generar una respuesta.")
    st.markdown(answer)
    
    st.markdown("---")
    
    # Crear dos columnas para información adicional
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Información del Proceso")
        
        # Clasificación
        # Mapeo de clasificaciones internas del classify_node a etiquetas legibles.
        # Valores no mapeados se muestran en crudo sin romper el render.
        classification = result.get("classification", "N/A")
        classification_labels = {
            "legal_specific": "🏛️ Normativa específica",
            "procedural": "📋 Procedimientos",
            "general_laboral": "💼 Consulta general laboral",
            "general": "💬 Consulta general",
            "calculation": "🧮 Cálculos",
            "resume": "📄 Resumen"
        }
        st.write(f"**Tipo de consulta:** {classification_labels.get(classification, classification)}")
        
        # Herramientas ejecutadas
        tool_results = result.get("tool_results")
        if tool_results:
            tool_used = tool_results.get("tool_used", "N/A")
            st.write(f"**Herramienta usada:** ✅ `{tool_used}`")
        else:
            st.write("**Herramienta usada:** ➖ Búsqueda directa")
        
        # Documentos recuperados
        documents = result.get("documents", [])
        st.write(f"**Documentos recuperados:** {len(documents)}")
        
        # Verification
        verification = result.get("verification", {})
        quality_level = verification.get("quality_level", "unknown")
        quality_score = verification.get("quality_score", 0)
        
        quality_emojis = {
            "excellent": "🌟",
            "good": "✅",
            "needs_improvement": "⚠️",
            "poor": "❌"
        }
        
        emoji = quality_emojis.get(quality_level, "❓")
        st.write(f"**Calidad:** {emoji} {quality_level.replace('_', ' ').title()} ({quality_score:.0%})")
        # Detalles de verificación extendida
        supported = verification.get("supported_by_context")
        if supported is not None:
            st.write(f"**Soportada por contexto:** {'✅ Sí' if supported else '❌ No'}")

        # "or []" protege contra None: verify_node puede omitir la clave
        # en lugar de devolver lista vacía.
        unsupported_claims = verification.get("unsupported_claims", []) or []
        if unsupported_claims:
            st.write(f"**Afirmaciones no soportadas:** {len(unsupported_claims)}")
            with st.expander("Ver afirmaciones no soportadas"):
                for c in unsupported_claims:
                    st.write(f"- {c}")

        st.write(f"**Acción recomendada por verificador:** {verification.get('recommended_action', 'N/A')}")
        
    with col2:
        st.subheader("📚 Documentos Consultados")
        
        documents = result.get("documents", [])
        if documents:
            # Obtener documentos únicos
            # Deduplicación por id_documento: múltiples chunks del mismo documento
            # son habituales tras similarity_search. Se muestran máximo 5 documentos únicos.
            unique_docs = {}
            for doc in documents:
                doc_id = doc.metadata.get("id_documento", "N/A")
                if doc_id not in unique_docs:
                    unique_docs[doc_id] = {
                        "tipo": doc.metadata.get("tipo_documento", "N/A"),
                        "año": doc.metadata.get("año", "N/A")
                    }
            
            for doc_id, info in list(unique_docs.items())[:5]:
                st.write(f"• **{doc_id}**")
                st.write(f"  └─ {info['tipo']} - {info['año']}")
        else:
            st.info("No se recuperaron documentos específicos")
    
    # Mostrar metadatos de recuperación y regeneraciones
    # Trazabilidad del pipeline: k de recuperación, si se aplicó filtro de metadata
    # y número de reintentos de generación disparados por verify_node.
    with st.container():
        md = result.get("metadata", {})
        retrieval_k = md.get("retrieval_k")
        regen = md.get("regeneration_attempts", 0)
        used_filter = md.get("used_filter", False)
        st.markdown("---")
        st.subheader("⚙️ Meta")
        if retrieval_k is not None:
            st.write(f"**k de recuperación:** {retrieval_k}")
        st.write(f"**Filtro usado en recuperación:** {'✅' if used_filter else '➖'}")
        st.write(f"**Intentos de regeneración:** {regen}")

        retrieval_metrics = md.get("retrieval_metrics")
        retrieval_metrics_enabled = md.get("retrieval_metrics_enabled", False)
        if retrieval_metrics_enabled and retrieval_metrics:
            st.markdown("---")
            st.subheader("📈 Evaluación Retrieval @k")
            st.caption("Métricas calculadas cuando se provee ground truth en la consulta.")

            rows = []
            for raw_k, metric_values in retrieval_metrics.items():
                try:
                    k_value = int(raw_k)
                except (ValueError, TypeError):
                    k_value = raw_k

                rows.append({
                    "k": k_value,
                    "Recall@k": round(float(metric_values.get("recall", 0.0)), 3),
                    "Precision@k": round(float(metric_values.get("precision", 0.0)), 3),
                })

            rows = sorted(rows, key=lambda row: row["k"] if isinstance(row["k"], int) else 9999)
            st.dataframe(rows, use_container_width=True, hide_index=True)

            retrieved_doc_ids = md.get("retrieved_doc_ids", [])
            if retrieved_doc_ids:
                st.write(f"**IDs recuperados:** {', '.join(retrieved_doc_ids)}")
        elif md.get("retrieval_metrics_skipped_reason"):
            st.caption(f"Retrieval @k omitido: {md.get('retrieval_metrics_skipped_reason')}")
    
    # Expandible con detalles técnicos
    with st.expander("🔍 Ver detalles técnicos"):
        st.json({
            "classification": result.get("classification"),
            "num_documents": len(result.get("documents", [])),
            "tool_results": result.get("tool_results") is not None,
            "verification": result.get("verification"),
            "metadata": result.get("metadata", {})
        })


def main():
    """Función principal de la aplicación."""
    
    # Cargar configuración
    load_settings()
    
    # Inicializar estado
    initialize_session_state()
    
    # Mostrar sidebar
    display_sidebar()
    
    # Header principal
    st.title("⚖️ RAG Normativa Laboral Colombiana")
    st.markdown("""
    Sistema inteligente de consulta de normativa laboral colombiana. 
    Pregunta sobre leyes, decretos, sentencias y cálculos laborales.
    """)
    
    st.markdown("---")
    
    # Crear pestañas
    tab1, tab2 = st.tabs(["📖 Información del Sistema", "💬 Chat"])
    
    # Pestaña 1: Información del Sistema
    with tab1:
        st.header("📊 Información del RAG")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Documentos", "74")
            st.metric("Leyes", "22")
        
        with col2:
            st.metric("Decretos", "21")
            st.metric("Sentencias", "31")
        
        
        st.markdown("---")
        
        st.subheader("📚 Cobertura del Corpus")
        st.markdown("""
        El sistema contiene documentos sobre:
        
        - **Acoso Laboral**: Ley 1010 de 2006 y normativa relacionada
        - **Jornada Laboral**: Regulaciones sobre horarios y descansos
        - **Prestaciones Sociales**: Cálculos y normativa de prestaciones
        - **Contratación**: Tipos de contratos y condiciones
        - **Seguridad Social**: Obligaciones y afiliaciones
        - **Riesgos Laborales**: Prevención y compensación
        - **Teletrabajo**: Decreto 1072 y regulaciones
        """)
        
        st.markdown("---")
        
        st.subheader("🔧 Capacidades del Sistema")
        st.markdown("""
        1. **Consultas sobre documentos específicos**
        2. **Búsqueda por tipo de documento**
        3. **Búsqueda temporal (años)**
        4. **Extracción de artículos**
        5. **Comparación entre documentos**
        6. **Cálculo de prestaciones sociales**
        7. **Resúmenes de documentos**
        """)
    
    # Pestaña 2: Chat
    with tab2:
        st.header("💬 Consulta el RAG")

        with st.expander("🧪 Evaluación de retrieval (opcional)"):
            st.caption("Activa esta opción solo cuando quieras medir Recall@k y Precision@k en una consulta puntual.")
            eval_enabled = st.checkbox(
                "Activar evaluación @k para esta consulta",
                value=False,
                key="enable_retrieval_eval",
            )

            gt_doc_ids_raw = st.text_input(
                "Ground truth (IDs separados por coma)",
                value="",
                placeholder="Ej: LEY_1010_2006, DECRETO_1072_2015",
                key="ground_truth_doc_ids",
                disabled=not eval_enabled,
            )

            eval_ks_raw = st.text_input(
                "Valores de k (CSV)",
                value="1,3,5",
                placeholder="Ej: 1,3,5",
                key="retrieval_eval_ks",
                disabled=not eval_enabled,
            )
        
        # Input de consulta
        user_query = st.text_area(
            "✍️ Escribe tu consulta:",
            height=100,
            placeholder="Ejemplo: ¿Qué dice la ley 1010 sobre acoso laboral?",
            key="query_input"
        )
        
        col1, col2, col3 = st.columns([1, 1, 4])
        with col1:
            submit = st.button("🚀 Consultar", type="primary", use_container_width=True)
        with col2:
            clear = st.button("🗑️ Limpiar", use_container_width=True)
        
        # st.rerun() re-ejecuta el script completo pero NO destruye session_state;
        # el grafo y el historial se conservan, solo se limpia el área de resultados visible.
        if clear:
            st.rerun()
        
        # Procesar consulta
        if submit and user_query.strip():
            try:
                # Mostrar progreso
                with st.status("🔄 Procesando consulta...", expanded=True) as status:
                    st.write("🔍 Clasificando consulta...")
                    time.sleep(0.5)
                    
                    st.write("🔧 Evaluando herramientas...")
                    time.sleep(0.5)
                    
                    st.write("📚 Recuperando documentos...")
                    time.sleep(0.5)
                    
                    # Ejecutar el grafo
                    # Estructura inicial del RAGState. Cada campo vacío será populado por su nodo:
                    # classify → tool_calling → retrieve → generate → verify.
                    metadata = {}
                    if eval_enabled:
                        ground_truth_doc_ids = parse_csv_tokens(gt_doc_ids_raw)
                        if not ground_truth_doc_ids:
                            st.warning("⚠️ Activas evaluación @k, pero falta ground truth. Ingresa al menos un id_documento.")
                            st.stop()

                        metadata["ground_truth_doc_ids"] = ground_truth_doc_ids
                        metadata["retrieval_eval_ks"] = parse_eval_ks(eval_ks_raw)

                    initial_state = {
                        "query": user_query,
                        "classification": "",
                        "query_type": None,
                        "transformed_queries": None,
                        "documents": [],
                        "tool_results": None,
                        "kg_results": None,
                        "answer": "",
                        "verification": {},
                        "metadata": metadata
                    }
                    
                    result = st.session_state.graph.invoke(initial_state)
                    
                    st.write("✍️ Generando respuesta...")
                    time.sleep(0.5)
                    
                    st.write("✅ Verificando calidad...")
                    time.sleep(0.5)
                    
                    status.update(label="✅ Consulta procesada exitosamente", state="complete")
                
                st.markdown("---")
                
                # Mostrar resultados
                display_response(result)
                
                # Agregar al historial
                st.session_state.history.append({
                    "query": user_query,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "classification": result.get("classification"),
                    "quality": result.get("verification", {}).get("quality_level")
                })
                
            # st.exception() expone el traceback completo al usuario.
            # En producción, reemplazar por mensaje genérico para evitar filtrar
            # rutas internas o fragmentos de API keys.
            except Exception as e:
                st.error(f"❌ Error al procesar la consulta: {str(e)}")
                st.exception(e)
        
        elif submit and not user_query.strip():
            st.warning("⚠️ Por favor, escribe una consulta antes de enviar.")
        
        # Mostrar historial si existe
        # Solo se persiste metadata mínima (query, timestamp, clasificación, calidad).
        # La respuesta completa y los documentos no se almacenan en historial.
        # La lista crece sin límite en memoria; la UI muestra únicamente las últimas 5.
        if st.session_state.history:
            st.markdown("---")
            with st.expander("📜 Historial de consultas"):
                for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
                    st.write(f"**{i}.** {item['query']}")
                    st.write(f"   └─ {item['timestamp']} | {item['classification']} | {item['quality']}")


if __name__ == "__main__":
    main()
