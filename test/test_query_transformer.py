"""
Demostración de las funciones de transformación de consultas.

Script de prueba para HyDE, Query Decomposition y MultiQueryRetriever.
Muestra cómo el sistema detecta automáticamente qué estrategia aplicar.

Ejecutar: python test_query_transformer.py
"""

from src.config import init_embeddings, init_groq_llm
from src.vectorstore import load_chroma_index
from src.query_transformer import (
    QueryTransformer,
    transform_query,
)
import os

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

def setup():
    """Inicializa componentes necesarios."""
    # Cargar variables de entorno (asegurar .env configurado)
    from src.config import load_settings
    load_settings()
    
    # Inicializar LLM y embeddings
    llm = init_groq_llm(temperature=0.1)
    embeddings = init_embeddings()
    
    # Cargar vectorstore
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "normativa_laboral")
    vectorstore = load_chroma_index(persist_dir, embeddings, collection_name)
    
    return llm, vectorstore


# ============================================================================
# EJEMPLO 1: Detección automática de tipo de consulta
# ============================================================================

def example_query_type_detection():
    """Demuestra cómo el sistema detecta si usar HyDE o Query Decomposition."""
    print("\n" + "="*70)
    print("EJEMPLO 1: Detección automática de tipo de consulta")
    print("="*70)
    
    llm, vectorstore = setup()
    transformer = QueryTransformer(llm, vectorstore)
    
    queries = [
        # Consulta corta → HyDE
        "¿Cuál es el salario mínimo?",
        
        # Consulta múltiple → Query Decomposition
        "¿Cuáles son los derechos de los trabajadores y cuáles son sus obligaciones?",
        
        # Consulta compleja con condicionales → Query Decomposition
        "¿Cómo se calcula la indemnización por despido injustificado? Y además, ¿qué derechos tengo?",
    ]
    
    for query in queries:
        query_type = transformer.detect_query_type(query)
        print(f"\n📝 Consulta: {query}")
        print(f"   → Tipo detectado: {query_type}")


# ============================================================================
# EJEMPLO 2: HyDE (Hypothetical Document Embeddings)
# ============================================================================

def example_hyde():
    """Demuestra cómo funciona HyDE para preguntas cortas."""
    print("\n" + "="*70)
    print("EJEMPLO 2: HyDE (Hypothetical Document Embeddings)")
    print("="*70)
    
    llm, vectorstore = setup()
    transformer = QueryTransformer(llm, vectorstore)
    
    question = "¿Qué son las prestaciones sociales?"
    
    print(f"\n📝 Pregunta: {question}")
    print("\n🔄 Proceso HyDE:")
    print("   1. Generando documento hipotético...")
    print("   2. Usando ese documento para buscar en la base vectorial...")
    
    hypo_doc, documents = transformer.hyde_search(question, k=4)
    
    print(f"\n📚 Documento hipotético generado ({len(hypo_doc)} caracteres):")
    print("-" * 70)
    print(hypo_doc[:300] + "...")
    print("-" * 70)
    
    print(f"\n✓ Documentos recuperados: {len(documents)}")
    for i, doc in enumerate(documents, 1):
        doc_id = doc.metadata.get('id_documento', 'N/A')
        tipo = doc.metadata.get('tipo_documento', 'N/A')
        print(f"   {i}. {doc_id} ({tipo})")


# ============================================================================
# EJEMPLO 3: Query Decomposition
# ============================================================================

def example_query_decomposition():
    """Demuestra cómo funciona Query Decomposition para consultas complejas."""
    print("\n" + "="*70)
    print("EJEMPLO 3: Query Decomposition")
    print("="*70)
    
    llm, vectorstore = setup()
    transformer = QueryTransformer(llm, vectorstore)
    
    question = "¿Cuál es la diferencia entre un contrato a término fijo y uno indefinido? Y además, ¿qué derechos tengo en cada caso?"
    
    print(f"\n📝 Pregunta compleja: {question}")
    print("\n🔄 Proceso de descomposición:")
    
    sub_queries, documents = transformer.decomposed_search(question, k=4)
    
    print(f"\n✓ Consulta descompuesta en {len(sub_queries)} sub-consultas")
    print(f"✓ Documentos recuperados: {len(documents)}")
    
    print(f"\nDetalles de documentos:")
    for i, doc in enumerate(documents[:3], 1):
        doc_id = doc.metadata.get('id_documento', 'N/A')
        tipo = doc.metadata.get('tipo_documento', 'N/A')
        content_preview = doc.page_content[:100].replace('\n', ' ')
        print(f"   {i}. {doc_id} ({tipo})")
        print(f"      Preview: {content_preview}...")


# ============================================================================
# EJEMPLO 4: Multi-Query Retrieval
# ============================================================================

def example_multi_query():
    """Demuestra cómo funciona Multi-Query Retrieval."""
    print("\n" + "="*70)
    print("EJEMPLO 4: Multi-Query Retrieval")
    print("="*70)
    
    llm, vectorstore = setup()
    transformer = QueryTransformer(llm, vectorstore)
    
    question = "¿Cómo se calcula el pago de horas extras?"
    
    print(f"\n📝 Pregunta: {question}")
    print("\n🔄 Proceso Multi-Query:")
    print("   1. Generando múltiples perspectivas de la pregunta...")
    print("   2. Buscando documentos para cada variación...")
    print("   3. Deduplicando y combinando resultados...")
    
    try:
        queries, documents = transformer.multi_query_retrieval(question, k=4)
        
        print(f"\n✓ Variaciones generadas: {len(queries)}")
        for i, q in enumerate(queries, 1):
            print(f"   {i}. {q}")
        
        print(f"\n✓ Documentos recuperados (deduplicados): {len(documents)}")
        for i, doc in enumerate(documents, 1):
            doc_id = doc.metadata.get('id_documento', 'N/A')
            tipo = doc.metadata.get('tipo_documento', 'N/A')
            print(f"   {i}. {doc_id} ({tipo})")
    
    except Exception as e:
        print(f"   ⚠️ Error en Multi-Query (fallback a HyDE): {str(e)[:100]}")


# ============================================================================
# EJEMPLO 5: Función de alto nivel transform_query
# ============================================================================

def example_transform_query_high_level():
    """Demuestra cómo usar la función de alto nivel transform_query."""
    print("\n" + "="*70)
    print("EJEMPLO 5: Función de alto nivel transform_query()")
    print("="*70)
    
    llm, vectorstore = setup()
    
    test_questions = [
        "¿Qué es el despido injustificado?",  # HyDE
        "¿Cuál es la diferencia entre cesantías y prima de servicios?",  # HyDE o Multi
        "¿Cómo se calcula la indemnización y cuáles son mis derechos?",  # Decomposition
    ]
    
    for question in test_questions:
        print(f"\n📝 Autoanálisis: {question}")
        
        result = transform_query(
            question=question,
            llm=llm,
            vectorstore=vectorstore,
            k=4
        )
        
        print(f"   Tipo: {result['query_type']}")
        print(f"   Consultas transformadas: {len(result['transformed_queries'])}")
        print(f"   Documentos recuperados: {len(result['documents'])}")
        print(f"   Metadata: {result['metadata']['num_transformed']} transformaciones, "
              f"{result['metadata']['num_documents_retrieved']} documentos")


# ============================================================================
# EJECUTABLE
# ============================================================================

def main():
    """Ejecuta todos los ejemplos."""
    try:
        print("\n" + "="*70)
        print("DEMOSTRACIÓN: Transformación de Consultas (HyDE + Decomposition)")
        print("="*70)
        
        # Ejecutar ejemplos
        example_query_type_detection()
        example_hyde()
        example_query_decomposition()
        example_multi_query()
        example_transform_query_high_level()
        
        print("\n" + "="*70)
        print("✅ Demostración completada")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error en demostración: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
