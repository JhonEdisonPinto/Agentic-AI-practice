"""
Aplicación Streamlit para el RAG de Normativa Laboral Colombiana.
Sistema de consulta inteligente con LangGraph y ChromaDB.
"""
import streamlit as st
import time
from typing import Dict, Any

from src.config import load_settings
from src.graph import build_graph


# Configuración de la página
st.set_page_config(
    page_title="RAG Normativa Laboral Colombiana",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)


def initialize_session_state():
    """Inicializa el estado de la sesión."""
    if "graph" not in st.session_state:
        with st.spinner("🔨 Construyendo grafo RAG..."):
            st.session_state.graph = build_graph()
    
    if "history" not in st.session_state:
        st.session_state.history = []


def display_sidebar():
    """Muestra la barra lateral con información y ejemplos."""
    with st.sidebar:
        st.title("⚖️ RAG Laboral")
        st.markdown("---")
        
        st.subheader("📚 Sobre este sistema")
        st.markdown("""
        Sistema RAG (Retrieval-Augmented Generation) especializado en 
        normativa laboral colombiana.
        
        **Corpus**: 74 documentos legales
        - 22 Leyes
        - 21 Decretos  
        - 31 Sentencias
        """)
        
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
        
        st.markdown("---")
        st.subheader("💡 Ejemplos de consultas")
        
        examples = {
            "Cálculo": "¿Cómo calculo las prestaciones sociales con un salario de $2,500,000?",
            "Documento específico": "Muéstrame información sobre la ley 1010 de 2006",
            "Rango de años": "¿Qué normativa sobre jornada laboral se publicó entre 2010 y 2020?",
            "Artículo específico": "¿Qué dice el artículo 5 de la ley 1010?",
            "Comparación": "¿Cuáles son las diferencias entre la ley 1010 y el decreto 1072?",
            "Resumen": "Dime un resumen del decreto 36 de 2016"
        }
        
        for category, example in examples.items():
            if st.button(f"📝 {category}", key=f"example_{category}", use_container_width=True):
                st.session_state.example_query = example


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
    
    # Check for example query from sidebar
    if "example_query" in st.session_state:
        query = st.session_state.example_query
        del st.session_state.example_query
    else:
        query = ""
    
    # Input de consulta
    user_query = st.text_area(
        "💬 Escribe tu consulta:",
        value=query,
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
