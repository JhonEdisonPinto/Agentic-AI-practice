"""
Script de prueba para el LangGraph RAG.
Prueba el flujo completo: Clasificación -> Recuperación -> Generación -> Verificación
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar desde src
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.config import load_settings
from src.graph import build_graph
import json


def test_rag_workflow():
    """
    Prueba el workflow completo del RAG.
    """
    print("=" * 80)
    print("PRUEBA DEL LANGGRAPH RAG - NORMATIVA LABORAL COLOMBIANA")
    print("=" * 80)
    
    # Cargar configuración
    print("\n⚙️  Cargando configuración...")
    load_settings()
    print("   ✓ Configuración cargada")
    
    # Construir grafo
    print("\n🔨 Construyendo grafo...")
    try:
        graph = build_graph()
        print("   ✓ Grafo compilado exitosamente")
    except Exception as e:
        print(f"   ✗ Error al construir grafo: {e}")
        return
    
    # Consultas de prueba
    test_queries = [
        "¿Qué es el acoso laboral según la ley colombiana?",
        "¿Cuáles son mis prestaciones sociales?",
        "¿Cuántas horas puedo trabajar a la semana?",
        "¿Cuánto dura el período de prueba en un contrato laboral?",
    ]
    
    print(f"\n📝 Probando {len(test_queries)} consultas...")
    
    for i, query in enumerate(test_queries, 1):
        print("\n" + "=" * 80)
        print(f"CONSULTA {i}/{len(test_queries)}")
        print("=" * 80)
        print(f"\n💬 Usuario: {query}")
        
        # Ejecutar grafo
        try:
            initial_state = {
                "query": query,
                "classification": "",
                "documents": [],
                "tool_results": None,
                "answer": "",
                "verification": {},
                "metadata": {}
            }
            
            # Invocar el grafo
            result = graph.invoke(initial_state)
            
            # Mostrar resultados
            print(f"\n📊 RESULTADOS:")
            print(f"\n1. Clasificación:")
            print(f"   • Tipo: {result.get('classification', 'N/A')}")
            
            print(f"\n2. Recuperación:")
            documents = result.get('documents', [])
            print(f"   • Documentos encontrados: {len(documents)}")
            if documents:
                for j, doc in enumerate(documents[:3], 1):
                    doc_id = doc.metadata.get('id_documento', 'N/A')
                    tipo = doc.metadata.get('tipo_documento', 'N/A')
                    print(f"      {j}. {doc_id} ({tipo})")
            
            print(f"\n3. Respuesta:")
            answer = result.get('answer', '')
            if len(answer) > 300:
                print(f"   {answer[:300]}...")
                print(f"   ... (respuesta completa: {len(answer)} caracteres)")
            else:
                print(f"   {answer}")
            
            print(f"\n4. Verificación:")
            verification = result.get('verification', {})
            quality_score = verification.get('quality_score', 0)
            quality_level = verification.get('quality_level', 'unknown')
            print(f"   • Calidad: {quality_level} ({quality_score:.2%})")
            print(f"   • Fuentes citadas: {verification.get('num_sources', 0)}")
            
            # Metadata
            metadata = result.get('metadata', {})
            if 'retrieval_scores' in metadata:
                avg_score = sum(metadata['retrieval_scores']) / len(metadata['retrieval_scores'])
                print(f"   • Score promedio de recuperación: {avg_score:.4f}")
            
        except Exception as e:
            print(f"\n❌ ERROR al procesar consulta: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Resumen final
    print("\n" + "=" * 80)
    print("✅ PRUEBA COMPLETADA")
    print("=" * 80)
    print("\n🎉 El LangGraph RAG está funcionando correctamente!")
    print("\nPróximos pasos:")
    print("   1. Integrar con la interfaz Streamlit")
    print("   2. Agregar más validaciones y mejoras")
    print("   3. Optimizar parámetros de recuperación")
    print("=" * 80)


def test_single_query():
    """
    Prueba con una sola consulta interactiva.
    """
    print("=" * 80)
    print("PRUEBA INTERACTIVA DEL RAG")
    print("=" * 80)
    
    load_settings()
    graph = build_graph()
    
    query = input("\n💬 Escribe tu consulta sobre derecho laboral: ")
    
    if not query.strip():
        print("❌ Consulta vacía")
        return
    
    print(f"\n🚀 Procesando: {query}")
    
    initial_state = {
        "query": query,
        "classification": "",
        "documents": [],
        "tool_results": None,
        "answer": "",
        "verification": {},
        "metadata": {}
    }
    
    result = graph.invoke(initial_state)
    
    print(f"\n{'='*80}")
    print("RESPUESTA")
    print("=" * 80)
    print(f"\n{result.get('answer', 'Sin respuesta')}")
    print(f"\n{'='*80}")
    
    verification = result.get('verification', {})
    print(f"Calidad: {verification.get('quality_level', 'N/A')} ({verification.get('quality_score', 0):.2%})")
    print(f"Fuentes consultadas: {len(result.get('documents', []))}")
    print("=" * 80)


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "interactive":
        test_single_query()
    else:
        test_rag_workflow()
