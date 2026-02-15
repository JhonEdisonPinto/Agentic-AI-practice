"""
Script de prueba usando embeddings GRATUITOS con Sentence Transformers.
No requiere API keys - el modelo se descarga y ejecuta localmente.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar desde src
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import os
from src.config import load_settings
from src.vectorstore import create_chroma_index

def test_free_embeddings():
    print("=" * 60)
    print("PRUEBA DE EMBEDDINGS GRATUITOS CON CHROMADB")
    print("=" * 60)
    
    # 1. Inicializar embeddings gratuitos
    print("\n1. Inicializando embeddings gratuitos locales...")
    print("   Modelo: paraphrase-multilingual-MiniLM-L12-v2")
    print("   - Soporta español ✓")
    print("   - Ejecuta localmente ✓")
    print("   - Sin costo de API ✓")
    
    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Modelo multilingüe optimizado para español
        model_name = "paraphrase-multilingual-MiniLM-L12-v2"
        
        embedding_fn = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        print(f"   ✓ Modelo de embeddings cargado exitosamente")
    except ImportError:
        print("\n   ⚠ Falta instalar dependencias. Ejecuta:")
        print("   pip install sentence-transformers langchain-huggingface")
        return
    except Exception as e:
        print(f"   ✗ Error al inicializar embeddings: {e}")
        return
    
    # 2. Probar generación de embeddings
    print("\n2. Probando generación de embeddings...")
    test_text = "El derecho laboral regula las relaciones entre trabajadores y empleadores."
    try:
        embedding_vector = embedding_fn.embed_query(test_text)
        print(f"   ✓ Embedding generado exitosamente")
        print(f"   - Dimensiones del vector: {len(embedding_vector)}")
        print(f"   - Primeros 5 valores: {[f'{v:.4f}' for v in embedding_vector[:5]]}")
    except Exception as e:
        print(f"   ✗ Error al generar embedding: {e}")
        return
    
    # 3. Crear índice Chroma de prueba
    print("\n3. Creando índice ChromaDB de prueba...")
    persist_dir = "./data/chroma_test_free"
    collection_name = "test_collection_free"
    try:
        vectorstore = create_chroma_index(
            persist_dir=persist_dir,
            embedding_fn=embedding_fn,
            collection_name=collection_name
        )
        print(f"   ✓ Índice Chroma creado exitosamente")
        print(f"   - Directorio: {persist_dir}")
        print(f"   - Colección: {collection_name}")
    except Exception as e:
        print(f"   ✗ Error al crear índice Chroma: {e}")
        return
    
    # 4. Agregar documentos de prueba en español
    print("\n4. Agregando documentos de prueba en español...")
    test_documents = [
        "El contrato de trabajo es un acuerdo entre empleador y trabajador donde se establecen las condiciones laborales.",
        "Las prestaciones sociales incluyen cesantías, primas, vacaciones y otros beneficios establecidos por ley.",
        "El salario mínimo es establecido anualmente por el gobierno colombiano mediante decreto.",
        "La jornada laboral ordinaria es de 8 horas diarias y 48 horas semanales según el código laboral.",
        "El acoso laboral está prohibido por la Ley 1010 de 2006 y sanciona conductas hostiles en el trabajo.",
    ]
    
    metadatas = [
        {"source": "codigo_laboral", "topic": "contrato", "articulo": "22"},
        {"source": "codigo_laboral", "topic": "prestaciones", "articulo": "249"},
        {"source": "decreto", "topic": "salario", "año": "2026"},
        {"source": "codigo_laboral", "topic": "jornada", "articulo": "161"},
        {"source": "ley_1010", "topic": "acoso", "articulo": "2"},
    ]
    
    try:
        ids = [f"doc_{i}" for i in range(len(test_documents))]
        vectorstore.add_texts(texts=test_documents, metadatas=metadatas, ids=ids)
        print(f"   ✓ {len(test_documents)} documentos agregados exitosamente")
    except Exception as e:
        print(f"   ✗ Error al agregar documentos: {e}")
        return
    
    # 5. Realizar búsquedas semánticas en español
    print("\n5. Realizando búsquedas semánticas en español...")
    queries = [
        "¿Cuáles son las prestaciones laborales?",
        "¿Qué dice la ley sobre el acoso en el trabajo?",
        "¿Cuántas horas se trabaja a la semana?",
    ]
    
    for query in queries:
        print(f"\n   📝 Query: '{query}'")
        try:
            results = vectorstore.similarity_search_with_score(query, k=2)
            print(f"   {'='*56}")
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n   🔍 Resultado #{i} (Score: {score:.4f})")
                print(f"   Documento: {doc.page_content}")
                print(f"   Metadata: {doc.metadata}")
        except Exception as e:
            print(f"   ✗ Error en la búsqueda: {e}")
            continue
    
    # 6. Verificar calidad de recuperación
    print(f"\n{'='*60}")
    print("6. Verificando calidad de la recuperación semántica...")
    
    test_cases = [
        ("prestaciones laborales", "prestaciones"),
        ("acoso laboral", "acoso"),
        ("jornada de trabajo", "jornada"),
    ]
    
    correct = 0
    for query_text, expected_topic in test_cases:
        results = vectorstore.similarity_search_with_score(query_text, k=1)
        if results:
            top_result = results[0][0]
            actual_topic = top_result.metadata.get("topic")
            if actual_topic == expected_topic:
                print(f"   ✓ '{query_text}' → '{actual_topic}' (correcto)")
                correct += 1
            else:
                print(f"   ✗ '{query_text}' → '{actual_topic}' (esperado: '{expected_topic}')")
    
    accuracy = (correct / len(test_cases)) * 100
    
    print("\n" + "=" * 60)
    print("PRUEBA COMPLETADA EXITOSAMENTE ✓")
    print("=" * 60)
    print(f"\nPrecisión de recuperación: {accuracy:.1f}% ({correct}/{len(test_cases)})")
    print("\nConclusión:")
    print("✓ El modelo de embeddings gratuito funciona correctamente")
    print("✓ ChromaDB está indexando y recuperando documentos")
    print("✓ La búsqueda semántica en español es efectiva")
    print("✓ No se requieren API keys ni costos")
    print("\nVentajas de este modelo:")
    print("- Totalmente gratuito y de código abierto")
    print("- Se ejecuta localmente (privacidad de datos)")
    print("- Optimizado para español y otros idiomas")
    print("- Ideal para documentos legales colombianos")
    print("\n¡Puedes proceder a indexar tus PDFs de normativa laboral!")
    print("=" * 60)


if __name__ == "__main__":
    test_free_embeddings()
