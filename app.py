"""
Aplicación Streamlit para el RAG de Normativa Laboral Colombiana.
Sistema de consulta inteligente con LangGraph y ChromaDB.
"""
import streamlit as st
import time
import sys
import subprocess
from typing import Dict, Any
from pathlib import Path

from src.config import load_settings
from src.graph import build_graph


# Configuración de la página
st.set_page_config(
    page_title="RAG Normativa Laboral Colombiana",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def check_database():
    """Verifica si existe y funciona la base de datos ChromaDB."""
    chroma_dir = Path("./data/chroma")
    
    # Verificar si existe
    if not chroma_dir.exists() or not any(chroma_dir.glob("*")):
        return False
    
    # Verificar si funciona (no está corrupta)
    try:
        from src.config import init_embeddings
        from src.vectorstore import load_chroma_index
        import os
        
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "normativa_laboral")
        embedding_fn = init_embeddings()
        
        vectorstore = load_chroma_index(persist_dir, embedding_fn, collection_name)
        # Intentar hacer una consulta simple
        vectorstore.similarity_search("test", k=1)
        return True
    except Exception as e:
        print(f"⚠️ BD corrupta o inaccesible: {e}")
        return False


def initialize_database():
    """Genera la base de datos desde los PDFs si no existe o está corrupta."""
    corpus_dir = Path("./src/corpus")
    
    # Verificar si existen PDFs
    if not corpus_dir.exists() or not list(corpus_dir.glob("*.pdf")):
        st.error("❌ No se encontraron PDFs en `src/corpus/`")
        st.info("""
        **Para que funcione el sistema necesitas:**
        
        1. Subir los PDFs a la carpeta `src/corpus/`
        2. O subir la carpeta `data/` completa usando Git LFS
        
        **Contacta al administrador del sistema.**
        """)
        st.stop()
        return False
    
    # Generar la base de datos
    st.info("🔨 Generando base de datos desde PDFs. Esto puede tomar varios minutos...")
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    try:
        status_text.text("📚 Indexando documentos...")
        progress_bar.progress(10)
        
        # Ejecutar el script de indexación
        result = subprocess.run(
            [sys.executable, "index_pdfs.py"],
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos máximo
        )
        
        progress_bar.progress(90)
        
        if result.returncode == 0:
            progress_bar.progress(100)
            status_text.text("✅ Base de datos generada exitosamente")
            time.sleep(2)
            return True
        else:
            st.error(f"❌ Error al generar la BD:\n```\n{result.stderr}\n```")
            return False
            
    except subprocess.TimeoutExpired:
        st.error("⏱️ La generación de la BD excedió el tiempo límite")
        return False
    except Exception as e:
        st.error(f"❌ Error al generar la BD: {str(e)}")
        return False


def initialize_session_state():
    """Inicializa el estado de la sesión."""
    # Verificar y generar base de datos si es necesario
    if "db_initialized" not in st.session_state:
        if not check_database():
            st.warning("⚠️ Base de datos no disponible. Generando desde PDFs...")
            if not initialize_database():
                st.stop()
        st.session_state.db_initialized = True
    
    if "graph" not in st.session_state:
        with st.spinner("🔨 Construyendo grafo RAG..."):
            st.session_state.graph = build_graph()
    
    if "history" not in st.session_state:
        st.session_state.history = []


def display_sidebar():
    """Muestra la barra lateral con información."""
    with st.sidebar:
        st.title("⚖️ RAG Laboral")
        

        st.markdown("---")
        st.subheader("🔧 Herramientas disponibles")
        st.markdown("""
        1. **Cálculo de prestaciones sociales**
        2. **Búsqueda por tipo de documento**
        3. **Búsqueda por rango de años**
        4. **Extracción de artículos específicos**
        5. **Comparación de documentos**
        6. **Resumen de documentos**
        """)


def display_response(result: Dict[str, Any]):
    """Muestra la respuesta del RAG de manera estructurada."""
    
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
        classification = result.get("classification", "N/A")
        classification_labels = {
            "legal_specific": "🏛️ Normativa específica",
            "procedural": "📋 Procedimientos",
            "general": "💼 Consulta general",
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
        
    with col2:
        st.subheader("📚 Documentos Consultados")
        
        documents = result.get("documents", [])
        if documents:
            # Obtener documentos únicos
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
                    initial_state = {
                        "query": user_query,
                        "classification": "",
                        "documents": [],
                        "tool_results": None,
                        "answer": "",
                        "verification": {},
                        "metadata": {}
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
                
            except Exception as e:
                st.error(f"❌ Error al procesar la consulta: {str(e)}")
                st.exception(e)
        
        elif submit and not user_query.strip():
            st.warning("⚠️ Por favor, escribe una consulta antes de enviar.")
        
        # Mostrar historial si existe
        if st.session_state.history:
            st.markdown("---")
            with st.expander("📜 Historial de consultas"):
                for i, item in enumerate(reversed(st.session_state.history[-5:]), 1):
                    st.write(f"**{i}.** {item['query']}")
                    st.write(f"   └─ {item['timestamp']} | {item['classification']} | {item['quality']}")


if __name__ == "__main__":
    main()
