"""
Script simple para verificar ChromaDB sin cargar embeddings.
"""
import chromadb
from pathlib import Path

persist_dir = "./data/chroma"
collection_name = "normativa_laboral"

print("=" * 80)
print("VERIFICACIÓN SIMPLE DE CHROMADB")
print("=" * 80)

# Verificar si existe el directorio
if not Path(persist_dir).exists():
    print(f"\n❌ El directorio {persist_dir} no existe")
    exit(1)

# Conectar a ChromaDB
client = chromadb.PersistentClient(path=persist_dir)

# Obtener la colección
try:
    collection = client.get_collection(name=collection_name)
    
    # Contar documentos
    count = collection.count()
    print(f"\n📊 ESTADÍSTICAS:")
    print(f"   • Total de chunks: {count}")
    
    if count > 0:
        # Obtener TODOS los metadatos (no solo 1000)
        print(f"\n⏳ Obteniendo todos los metadatos ({count} chunks)...")
        results = collection.get(
            limit=count,  # Obtener TODOS los chunks
            include=['metadatas']
        )
        
        # Extraer documentos únicos
        unique_docs = set()
        doc_types = {}
        
        for metadata in results['metadatas']:
            doc_id = metadata.get('id_documento', 'N/A')
            doc_type = metadata.get('tipo_documento', 'N/A')
            
            if doc_id != 'N/A':
                unique_docs.add(doc_id)
                
                if doc_type not in doc_types:
                    doc_types[doc_type] = set()
                doc_types[doc_type].add(doc_id)
        
        print(f"   • Documentos únicos: {len(unique_docs)}")
        print(f"\n📋 Por tipo:")
        for doc_type, docs in sorted(doc_types.items()):
            print(f"   • {doc_type}: {len(docs)} documentos")
        
        print(f"\n📄 Documentos indexados (primeros 20):")
        for i, doc_id in enumerate(sorted(list(unique_docs))[:20], 1):
            print(f"   {i}. {doc_id}")
        
        if len(unique_docs) > 20:
            print(f"   ... y {len(unique_docs) - 20} más")
            
        # Verificar si tenemos los 74
        corpus_count = 74
        if len(unique_docs) < corpus_count:
            print(f"\n⚠️  Faltan {corpus_count - len(unique_docs)} documentos")
        else:
            print(f"\n✅ Corpus completo!")
    else:
        print("\n⚠️  Base de datos vacía")
        
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nLa colección no existe. Ejecuta: python index_pdfs.py")

print("=" * 80)
