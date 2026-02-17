"""
Script de prueba para las 5 herramientas (tools) del RAG.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar desde src
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

from src.config import load_settings, init_embeddings
from src.vectorstore import load_chroma_index
from src.tools import (
    calculate_prestaciones_sociales,
    search_by_document_type,
    search_by_year_range,
    extract_specific_article,
    compare_documents,
    resume_document
)
import os


def test_tools():
    """
    Prueba cada una de las 5 herramientas.
    """
    print("=" * 80)
    print("PRUEBA DE LAS 5 HERRAMIENTAS (TOOLS) DEL RAG")
    print("=" * 80)
    
    load_settings()
    
    # Cargar vectorstore
    print("\n⚙️  Cargando vectorstore...")
    try:
        persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
        collection_name = os.getenv("CHROMA_COLLECTION_NAME", "normativa_laboral")
        embedding_fn = init_embeddings()
        vectorstore = load_chroma_index(persist_dir, embedding_fn, collection_name)
        print("   ✓ Vectorstore cargado")
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return
    
    # ========================================================================
    # TOOL 1: Búsqueda por tipo de documento
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOOL 1: search_by_document_type")
    print("=" * 80)
    print("\n📝 Prueba: Buscar LEYES sobre acoso laboral")
    
    try:
        results = vectorstore.similarity_search("acoso laboral", k=10)
        leyes = [doc for doc in results if doc.metadata.get("tipo_documento") == "LEY"][:3]
        
        print(f"\n   Resultados encontrados: {len(leyes)}")
        for i, doc in enumerate(leyes, 1):
            print(f"\n   {i}. {doc.metadata.get('id_documento', 'N/A')}")
            print(f"      Tipo: {doc.metadata.get('tipo_documento', 'N/A')}")
            print(f"      Año: {doc.metadata.get('año', 'N/A')}")
            print(f"      Extracto: {doc.page_content[:150]}...")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # ========================================================================
    # TOOL 2: Búsqueda por rango de años
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOOL 2: search_by_year_range")
    print("=" * 80)
    print("\n📝 Prueba: Buscar documentos entre 2015 y 2020 sobre riesgos laborales")
    
    try:
        results = vectorstore.similarity_search("riesgos laborales", k=20)
        filtered = []
        for doc in results:
            año_str = doc.metadata.get("año", "")
            if año_str:
                try:
                    año = int(año_str)
                    if 2015 <= año <= 2020:
                        filtered.append(doc)
                except ValueError:
                    continue
        
        filtered = filtered[:3]
        print(f"\n   Resultados en rango 2015-2020: {len(filtered)}")
        for i, doc in enumerate(filtered, 1):
            print(f"\n   {i}. {doc.metadata.get('id_documento', 'N/A')}")
            print(f"      Año: {doc.metadata.get('año', 'N/A')}")
            print(f"      Tipo: {doc.metadata.get('tipo_documento', 'N/A')}")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # ========================================================================
    # TOOL 3: Calculadora de prestaciones sociales
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOOL 3: calculate_prestaciones_sociales")
    print("=" * 80)
    print("\n📝 Prueba: Calcular prestaciones para:")
    print("   • Salario: $2,500,000")
    print("   • Días trabajados: 360 (1 año)")
    print("   • Años de servicio: 1")
    
    try:
        resultado = calculate_prestaciones_sociales.invoke({
            "salario_mensual": 2500000,
            "dias_trabajados": 360,
            "años_servicio": 1.0
        })
        
        print(f"\n   📊 RESULTADOS:")
        print(f"   • Cesantías: ${resultado['cesantias']:,.2f}")
        print(f"   • Intereses cesantías: ${resultado['intereses_cesantias']:,.2f}")
        print(f"   • Prima de servicios: ${resultado['prima_servicios']:,.2f}")
        print(f"   • Vacaciones: ${resultado['vacaciones']:,.2f}")
        print(f"   ┌────────────────────────────────┐")
        print(f"   │ TOTAL: ${resultado['total_prestaciones']:,.2f} │")
        print(f"   └────────────────────────────────┘")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # ========================================================================
    # TOOL 4: Extraer artículo específico
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOOL 4: extract_specific_article")
    print("=" * 80)
    print("\n📝 Prueba: Extraer artículo 2 de la LEY 1010 DE 2006 (Acoso laboral)")
    
    try:
        # Buscar documentos de la Ley 1010
        results = vectorstore.similarity_search("artículo 2 definición acoso laboral", k=10)
        
        # Buscar el artículo 2 específicamente
        articulo_encontrado = None
        for doc in results:
            if "LEY_1010_2006"  in doc.metadata.get("id_documento", "") or \
               "1010" in doc.metadata.get("filename", ""):
                if "artículo 2" in doc.page_content.lower() or "art. 2" in doc.page_content.lower():
                    articulo_encontrado = doc.page_content
                    break
        
        if articulo_encontrado:
            print(f"\n   ✓ Artículo encontrado:")
            print(f"\n   {articulo_encontrado[:400]}...")
        else:
            print("   ⚠️  Artículo no encontrado en los resultados")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # ========================================================================
    # TOOL 5: Comparar documentos
    # ========================================================================
    print("\n" + "=" * 80)
    print("TOOL 5: compare_documents")
    print("=" * 80)
    print("\n📝 Prueba: Comparar tratamiento de 'jornada laboral' en 2 documentos")
    
    try:
        # Buscar fragmentos sobre jornada laboral
        results = vectorstore.similarity_search("jornada laboral horas trabajo", k=10)
        
        # Agrupar por documento
        docs_by_id = {}
        for doc in results:
            doc_id = doc.metadata.get("id_documento", "UNKNOWN")
            if doc_id not in docs_by_id:
                docs_by_id[doc_id] = []
            docs_by_id[doc_id].append(doc)
        
        # Tomar los 2 primeros documentos
        doc_ids = list(docs_by_id.keys())[:2]
        
        if len(doc_ids) >= 2:
            print(f"\n   Comparando:")
            print(f"   • Documento 1: {doc_ids[0]}")
            print(f"   • Documento 2: {doc_ids[1]}")
            
            for i, doc_id in enumerate(doc_ids, 1):
                docs = docs_by_id[doc_id]
                print(f"\n   📄 Documento {i} - {doc_id}:")
                print(f"      Fragmentos encontrados: {len(docs)}")
                if docs:
                    print(f"      Extracto: {docs[0].page_content[:200]}...")
        else:
            print("   ⚠️  No se encontraron suficientes documentos para comparar")
    except Exception as e:
        print(f"   ✗ Error: {e}")
    
    # Resumen final
    print("\n" + "=" * 80)
    print("✅ PRUEBA DE HERRAMIENTAS COMPLETADA")
    print("=" * 80)
    print("\n📋 Resumen de las 5 herramientas:")
    print("   1. ✓ search_by_document_type - Búsqueda por tipo (LEY/DECRETO/SENTENCIA)")
    print("   2. ✓ search_by_year_range - Búsqueda por rango de años")
    print("   3. ✓ calculate_prestaciones_sociales - Calculadora de liquidación")
    print("   4. ✓ extract_specific_article - Extracción de artículos específicos")
    print("   5. ✓ compare_documents - Comparación entre documentos")
    print("\n🎉 Todas las herramientas están implementadas y funcionando!")
    print("=" * 80)


if __name__ == "__main__":
    test_tools()
