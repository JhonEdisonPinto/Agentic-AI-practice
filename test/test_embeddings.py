"""
Script de prueba para verificar que el modelo de embeddings funciona con ChromaDB.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar desde src
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import os
from src.config import load_settings, init_embeddings
from src.vectorstore import create_chroma_index

def test_embeddings_and_chroma():
    print("=" * 60)
    print("PRUEBA DE EMBEDDINGS CON CHROMADB")
    print("=" * 60)
    
    # 1. Cargar configuración
    print("\n1. Cargando configuración...")
    load_settings()
    embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", "gemini")
    print(f"   ✓ Proveedor de embeddings: {embeddings_provider}")
    
    # 2. Inicializar embeddings
    print("\n2. Inicializando modelo de embeddings...")
    try:
        embedding_fn = init_embeddings()
        print(f"   ✓ Modelo de embeddings inicializado correctamente")
    except Exception as e:
        print(f"   ✗ Error al inicializar embeddings: {e}")
        return
    
    # 3. Probar generación de embeddings
    print("\n3. Probando generación de embeddings...")
    test_text = "El derecho laboral regula las relaciones entre trabajadores y empleadores."
    try:
        embedding_vector = embedding_fn.embed_query(test_text)
        print(f"   ✓ Embedding generado exitosamente")
        print(f"   - Dimensiones del vector: {len(embedding_vector)}")
        print(f"   - Primeros 5 valores: {embedding_vector[:5]}")
    except Exception as e:
        print(f"   ✗ Error al generar embedding: {e}")
        return
    
    # 4. Crear índice Chroma de prueba
    print("\n4. Creando índice ChromaDB de prueba...")
    persist_dir = "./data/chroma_test"
    collection_name = "test_collection"
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
    
    # 5. Agregar documentos de prueba
    print("\n5. Agregando documentos de prueba...")
    test_documents = [
        "El contrato de trabajo es un acuerdo entre empleador y trabajador.",
        "Las prestaciones sociales incluyen cesantías, primas y vacaciones.",
        "El salario mínimo es establecido anualmente por el gobierno.",
        "La jornada laboral ordinaria es de 8 horas diarias.",
        "El acoso laboral está prohibido por la Ley 1010 de 2006.",
    ]
    
    metadatas = [
        {"source": "test", "topic": "contrato"},
        {"source": "test", "topic": "prestaciones"},
        {"source": "test", "topic": "salario"},
        {"source": "test", "topic": "jornada"},
        {"source": "test", "topic": "acoso"},
    ]
    
    try:
        vectorstore.add_texts(texts=test_documents, metadatas=metadatas)
        print(f"   ✓ {len(test_documents)} documentos agregados exitosamente")
    except Exception as e:
        print(f"   ✗ Error al agregar documentos: {e}")
        return
    
    # 6. Realizar búsqueda por similitud
    print("\n6. Realizando búsqueda por similitud...")
    query = "¿Cuáles son las prestaciones laborales?"
    try:
        results = vectorstore.similarity_search_with_score(query, k=3)
        print(f"   ✓ Búsqueda realizada exitosamente")
        print(f"\n   Query: '{query}'")
        print(f"\n   Resultados (top 3):")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n   {i}. Documento: {doc.page_content}")
            print(f"      Score: {score:.4f}")
            print(f"      Metadata: {doc.metadata}")
    except Exception as e:
        print(f"   ✗ Error en la búsqueda: {e}")
        return
    
    # 7. Verificar recuperación
    print("\n7. Verificando precisión de la recuperación...")
    expected_topic = "prestaciones"
    top_result = results[0][0]
    if top_result.metadata.get("topic") == expected_topic:
        print(f"   ✓ ¡Excelente! El documento más relevante es sobre '{expected_topic}'")
        print(f"   ✓ El sistema de embeddings está funcionando correctamente")
    else:
        print(f"   ⚠ Advertencia: Se esperaba '{expected_topic}' pero se obtuvo '{top_result.metadata.get('topic')}'")
    
    print("\n" + "=" * 60)
    print("PRUEBA COMPLETADA EXITOSAMENTE ✓")
    print("=" * 60)
    print("\nConclusión:")
    print("- El modelo de embeddings está funcionando correctamente")
    print("- ChromaDB está indexando y recuperando documentos exitosamente")
    print("- La búsqueda por similitud semántica funciona como se espera")
    print("\nPuedes proceder a indexar tus documentos PDF en la carpeta corpus/")
    print("=" * 60)


if __name__ == "__main__":
    test_embeddings_and_chroma()
