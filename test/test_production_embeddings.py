"""
Script de prueba usando la configuración de producción con embeddings gratuitos.
Este script usa las mismas funciones que usará tu aplicación real.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar desde src
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.config import load_settings, init_embeddings
from src.vectorstore import create_chroma_index
import os

def test_production_setup():
    print("=" * 70)
    print("PRUEBA DE CONFIGURACIÓN DE PRODUCCIÓN")
    print("=" * 70)
    
    # 1. Cargar configuración
    print("\n1. Cargando configuración desde .env...")
    load_settings()
    embeddings_provider = os.getenv("EMBEDDINGS_PROVIDER", "local")
    embeddings_model = os.getenv("EMBEDDINGS_MODEL", "default")
    print(f"   ✓ Proveedor configurado: {embeddings_provider}")
    print(f"   ✓ Modelo configurado: {embeddings_model}")
    
    # 2. Inicializar embeddings usando la función de config
    print("\n2. Inicializando embeddings usando init_embeddings()...")
    try:
        embedding_fn = init_embeddings()
        print(f"   ✓ Embeddings inicializados correctamente")
        print(f"   ✓ Tipo: {type(embedding_fn).__name__}")
    except Exception as e:
        print(f"   ✗ Error al inicializar embeddings: {e}")
        return
    
    # 3. Probar generación de embeddings
    print("\n3. Probando generación de embeddings...")
    test_texts = [
        "El derecho laboral colombiano protege a los trabajadores.",
        "Colombian labor law protects workers.",
        "Las prestaciones sociales son un derecho fundamental."
    ]
    
    try:
        for i, text in enumerate(test_texts, 1):
            embedding = embedding_fn.embed_query(text)
            print(f"   ✓ Texto {i}: {len(embedding)} dimensiones")
            if i == 1:
                print(f"      Primeros valores: {[f'{v:.4f}' for v in embedding[:5]]}")
    except Exception as e:
        print(f"   ✗ Error al generar embeddings: {e}")
        return
    
    # 4. Crear índice ChromaDB de producción
    print("\n4. Creando índice ChromaDB...")
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "normativa_laboral")
    
    try:
        vectorstore = create_chroma_index(
            persist_dir=persist_dir,
            embedding_fn=embedding_fn,
            collection_name=collection_name
        )
        print(f"   ✓ Índice creado exitosamente")
        print(f"   - Directorio: {persist_dir}")
        print(f"   - Colección: {collection_name}")
    except Exception as e:
        print(f"   ✗ Error al crear índice: {e}")
        return
    
    # 5. Agregar documentos de prueba
    print("\n5. Agregando documentos de prueba...")
    test_documents = [
        "El contrato de trabajo puede ser verbal o escrito según el Código Sustantivo del Trabajo.",
        "Las prestaciones sociales incluyen cesantías, intereses a las cesantías, prima de servicios y vacaciones.",
        "El salario mínimo legal mensual vigente (SMLMV) es fijado anualmente por decreto.",
        "La jornada laboral ordinaria es de ocho (8) horas diarias y cuarenta y ocho (48) horas semanales.",
        "El acoso laboral está definido y sancionado por la Ley 1010 de 2006.",
        "Los trabajadores tienen derecho a afiliación a seguridad social: salud, pensión y riesgos laborales.",
        "El período de prueba no puede exceder de dos (2) meses en contratos a término indefinido.",
        "Las horas extras deben ser pagadas con recargo del 25% o 75% según el caso.",
    ]
    
    metadatas = [
        {"fuente": "Código Sustantivo del Trabajo", "articulo": "37", "tema": "contrato"},
        {"fuente": "Código Sustantivo del Trabajo", "articulo": "249", "tema": "prestaciones"},
        {"fuente": "Decreto anual", "tema": "salario"},
        {"fuente": "Código Sustantivo del Trabajo", "articulo": "161", "tema": "jornada"},
        {"fuente": "Ley 1010 de 2006", "articulo": "2", "tema": "acoso"},
        {"fuente": "Ley 100 de 1993", "tema": "seguridad_social"},
        {"fuente": "Código Sustantivo del Trabajo", "articulo": "77", "tema": "periodo_prueba"},
        {"fuente": "Código Sustantivo del Trabajo", "articulo": "168", "tema": "horas_extras"},
    ]
    
    try:
        ids = [f"test_doc_{i}" for i in range(len(test_documents))]
        vectorstore.add_texts(texts=test_documents, metadatas=metadatas, ids=ids)
        print(f"   ✓ {len(test_documents)} documentos agregados")
    except Exception as e:
        print(f"   ✗ Error al agregar documentos: {e}")
        return
    
    # 6. Realizar búsquedas semánticas
    print("\n6. Realizando búsquedas semánticas en español...")
    queries = [
        ("¿Cuáles son mis prestaciones sociales?", "prestaciones"),
        ("¿Cuántas horas puedo trabajar a la semana?", "jornada"),
        ("¿Qué es el acoso laboral?", "acoso"),
        ("¿Cuánto dura el período de prueba?", "periodo_prueba"),
    ]
    
    correct = 0
    total = len(queries)
    
    for query, expected_tema in queries:
        try:
            results = vectorstore.similarity_search_with_score(query, k=1)
            if results:
                doc, score = results[0]
                actual_tema = doc.metadata.get("tema")
                match = "✓" if actual_tema == expected_tema else "✗"
                
                print(f"\n   {match} Query: {query}")
                print(f"      Documento encontrado: {doc.page_content[:80]}...")
                print(f"      Tema: {actual_tema} (esperado: {expected_tema})")
                print(f"      Score: {score:.4f}")
                print(f"      Fuente: {doc.metadata.get('fuente', 'N/A')}")
                
                if actual_tema == expected_tema:
                    correct += 1
        except Exception as e:
            print(f"   ✗ Error en búsqueda: {e}")
    
    accuracy = (correct / total) * 100
    
    # 7. Resumen final
    print("\n" + "=" * 70)
    print("RESULTADOS DE LA PRUEBA")
    print("=" * 70)
    print(f"\n✓ Embeddings funcionando: Sí")
    print(f"✓ ChromaDB operativo: Sí")
    print(f"✓ Documentos indexados: {len(test_documents)}")
    print(f"✓ Precisión de búsqueda: {accuracy:.1f}% ({correct}/{total} correctas)")
    
    if accuracy >= 75:
        print(f"\n🎉 ¡EXCELENTE! El sistema está listo para producción.")
    elif accuracy >= 50:
        print(f"\n✓ Sistema funcional. Considera ajustar parámetros para mejor precisión.")
    else:
        print(f"\n⚠ Sistema necesita ajustes en la configuración.")
    
    print("\n" + "=" * 70)
    print("CONFIGURACIÓN ACTUAL")
    print("=" * 70)
    print(f"Proveedor: {embeddings_provider}")
    print(f"Modelo: {embeddings_model}")
    print(f"ChromaDB: {persist_dir}")
    print(f"Colección: {collection_name}")
    print(f"\n💡 Para cambiar el proveedor, edita EMBEDDINGS_PROVIDER en .env")
    print(f"   Opciones: local (gratis), gemini (API key), openai (API key)")
    print("=" * 70)


if __name__ == "__main__":
    test_production_setup()
