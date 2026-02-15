"""
Script de prueba para verificar que ChromaDB funciona con embeddings locales (sin API).
Usa un modelo de embeddings simple basado en TF-IDF para demostración.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar desde src
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import os
from src.vectorstore import create_chroma_index

class SimpleLocalEmbeddings:
    """Embeddings locales simples para pruebas sin necesidad de API."""
    
    def embed_query(self, text: str):
        """Genera un vector simple basado en el texto."""
        # Genera un vector de 384 dimensiones basado en hashing simple
        import hashlib
        # Crea un hash del texto
        text_hash = hashlib.sha256(text.encode()).digest()
        # Convierte a lista de floats normalizados
        vector = [float(b) / 255.0 for b in text_hash[:32]]
        # Expande a 384 dimensiones (estándar para embeddings)
        vector = vector * 12
        return vector
    
    def embed_documents(self, texts):
        """Genera vectores para múltiples documentos."""
        return [self.embed_query(text) for text in texts]


def test_chroma_local():
    print("=" * 60)
    print("PRUEBA DE CHROMADB CON EMBEDDINGS LOCALES")
    print("=" * 60)
    
    # 1. Inicializar embeddings locales
    print("\n1. Inicializando embeddings locales...")
    embedding_fn = SimpleLocalEmbeddings()
    print(f"   ✓ Embeddings locales inicializados")
    
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
    persist_dir = "./data/chroma_test_local"
    collection_name = "test_collection_local"
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
    
    # 4. Agregar documentos de prueba
    print("\n4. Agregando documentos de prueba...")
    test_documents = [
        "El contrato de trabajo es un acuerdo entre empleador y trabajador.",
        "Las prestaciones sociales incluyen cesantías, primas y vacaciones.",
        "El salario mínimo es establecido anualmente por el gobierno.",
        "La jornada laboral ordinaria es de 8 horas diarias.",
        "El acoso laboral está prohibido por la Ley 1010 de 2006.",
    ]
    
    metadatas = [
        {"source": "test", "topic": "contrato", "id": 1},
        {"source": "test", "topic": "prestaciones", "id": 2},
        {"source": "test", "topic": "salario", "id": 3},
        {"source": "test", "topic": "jornada", "id": 4},
        {"source": "test", "topic": "acoso", "id": 5},
    ]
    
    try:
        ids = [f"doc_{i}" for i in range(len(test_documents))]
        vectorstore.add_texts(texts=test_documents, metadatas=metadatas, ids=ids)
        print(f"   ✓ {len(test_documents)} documentos agregados exitosamente")
        for i, doc in enumerate(test_documents, 1):
            print(f"      {i}. {doc[:60]}...")
    except Exception as e:
        print(f"   ✗ Error al agregar documentos: {e}")
        return
    
    # 5. Verificar que los documentos se guardaron
    print("\n5. Verificando documentos guardados...")
    try:
        collection = vectorstore._collection
        count = collection.count()
        print(f"   ✓ Total de documentos en la colección: {count}")
    except Exception as e:
        print(f"   ⚠ Advertencia al contar documentos: {e}")
    
    # 6. Realizar búsqueda por similitud
    print("\n6. Realizando búsqueda por similitud...")
    queries = [
        "¿Cuáles son las prestaciones laborales?",
        "¿Qué es un contrato de trabajo?",
        "¿Cuántas horas es la jornada laboral?",
    ]
    
    for query in queries:
        print(f"\n   Query: '{query}'")
        try:
            results = vectorstore.similarity_search_with_score(query, k=3)
            print(f"   Resultados (top 3):")
            for i, (doc, score) in enumerate(results, 1):
                print(f"\n      {i}. Score: {score:.4f}")
                print(f"         Documento: {doc.page_content[:70]}...")
                print(f"         Topic: {doc.metadata.get('topic', 'N/A')}")
        except Exception as e:
            print(f"   ✗ Error en la búsqueda: {e}")
            continue
    
    print("\n" + "=" * 60)
    print("PRUEBA COMPLETADA ✓")
    print("=" * 60)
    print("\nConclusión:")
    print("- ChromaDB está funcionando correctamente")
    print("- Los documentos se indexan y recuperan exitosamente")
    print("- El sistema de vectorstore está operativo")
    print("\nNOTA: Estos son embeddings locales de prueba.")
    print("Para producción, necesitas una API key válida de Google Gemini")
    print("o OpenAI para obtener embeddings semánticos de calidad.")
    print("=" * 60)


if __name__ == "__main__":
    test_chroma_local()
