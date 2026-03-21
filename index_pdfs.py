"""
Script para indexar todos los PDFs de normativa laboral en ChromaDB.
Procesa los documentos, los divide en chunks y los almacena con embeddings.
Pipeline: PDF → páginas → chunks → embeddings → ChromaDB.
Ejecución manual única; re-ejecutar sin limpiar con clean_chroma.py puede duplicar datos.
"""
import os
import re
from pathlib import Path
from typing import List, Dict
from datetime import datetime

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src.config import load_settings, init_embeddings
from src.vectorstore import create_chroma_index


def extract_metadata_from_filename(filename: str) -> Dict[str, str]:
    """
    Extrae metadata del nombre del archivo.
    
    Ejemplos:
    - LEY 1010 DE 2006.pdf -> tipo: LEY, numero: 1010, año: 2006
    - DECRETO 1072 DE 2015.pdf -> tipo: DECRETO, numero: 1072, año: 2015
    - C-051 de 1995.pdf -> tipo: SENTENCIA, numero: C-051, año: 1995
    """
    # 3 patrones evaluados en orden; el primer match retorna inmediatamente.
    # Nombres no reconocidos caen en fallback como "DOCUMENTO" sin año ni número,
    # lo que los hace invisibles para tools que filtran por tipo_documento.
    metadata = {
        "filename": filename,
        "source": filename,
    }
    
    # Patrón para LEY/DECRETO: "LEY 1010 DE 2006"
    match = re.match(r"(LEY|DECRETO)\s+(\d+)\s+DE\s+(\d{4})", filename, re.IGNORECASE)
    if match:
        metadata["tipo_documento"] = match.group(1).upper()
        metadata["numero"] = match.group(2)
        metadata["año"] = match.group(3)
        metadata["id_documento"] = f"{match.group(1).upper()}_{match.group(2)}_{match.group(3)}"
        return metadata
    
    # Patrón para Sentencias: "C-051 de 1995"
    # Solo reconoce sentencias C-xxx y T-xxx; otros prefijos (SU, A) no se capturan.
    # El guión se omite en id_documento: "C-051" → "SENTENCIA_C051_1995".
    match = re.match(r"([CT])-(\d+)\s+de\s+(\d{4})", filename, re.IGNORECASE)
    if match:
        metadata["tipo_documento"] = "SENTENCIA"
        metadata["numero"] = f"{match.group(1).upper()}-{match.group(2)}"
        metadata["año"] = match.group(3)
        metadata["id_documento"] = f"SENTENCIA_{match.group(1).upper()}{match.group(2)}_{match.group(3)}"
        return metadata
    
    # Fallback
    metadata["tipo_documento"] = "DOCUMENTO"
    metadata["id_documento"] = filename.replace(".pdf", "").replace(" ", "_")
    
    return metadata


def load_pdf_documents(corpus_path: str) -> List[Document]:
    """
    Carga todos los PDFs del directorio corpus.
    """
    corpus_dir = Path(corpus_path)
    pdf_files = list(corpus_dir.glob("*.pdf"))
    
    print(f"\n📂 Encontrados {len(pdf_files)} archivos PDF en {corpus_path}")
    
    all_documents = []
    
    for i, pdf_path in enumerate(pdf_files, 1):
        print(f"\n[{i}/{len(pdf_files)}] Procesando: {pdf_path.name}")
        
        try:
            # Cargar PDF
            loader = PyPDFLoader(str(pdf_path))
            pages = loader.load()
            
            # Extraer metadata del nombre del archivo
            # La metadata extraída del nombre se propaga uniformemente
            # a todas las páginas del documento (no varía por página).
            file_metadata = extract_metadata_from_filename(pdf_path.name)
            
            # Agregar metadata a cada página
            for page_num, page in enumerate(pages, 1):
                page.metadata.update(file_metadata)
                page.metadata["page"] = page_num
                page.metadata["total_pages"] = len(pages)
            
            all_documents.extend(pages)
            print(f"   ✓ {len(pages)} páginas cargadas")
            
        # PDFs corruptos o protegidos se saltan sin detener el pipeline.
        except Exception as e:
            print(f"   ✗ Error al cargar {pdf_path.name}: {e}")
            continue
    
    print(f"\n✓ Total de páginas cargadas: {len(all_documents)}")
    return all_documents


def split_documents(documents: List[Document], embedding_fn=None) -> List[Document]:
    """
    Divide los documentos en chunks más pequeños para mejor recuperación.
    """
    print("\n✂️  Dividiendo documentos en chunks...")

    chunks = []

    # SemanticChunker mejora los cortes al detectar cambios semánticos reales
    # y reduce fragmentaciones artificiales por tamaño fijo.
    if embedding_fn is not None:
        try:
            from langchain_experimental.text_splitter import SemanticChunker

            semantic_chunker = SemanticChunker(embedding_fn)
            chunks = semantic_chunker.split_documents(documents)
            print("   ✓ Chunking semántico aplicado (SemanticChunker)")
        except Exception as e:
            print(f"   ⚠️ No se pudo usar SemanticChunker: {e}")
            print("   ⚠️ Usando fallback con RecursiveCharacterTextSplitter")

    if not chunks:
        # Fallback robusto: chunking por tamaño con overlap.
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,  # Tamaño del chunk en caracteres
            chunk_overlap=500,  # Overlap para mantener contexto
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = text_splitter.split_documents(documents)
        print("   ✓ Chunking por tamaño aplicado (fallback)")
    
    print(f"✓ {len(chunks)} chunks creados (promedio: {len(chunks)//len(documents)} chunks por página)")
    
    return chunks


def index_documents(chunks: List[Document], persist_dir: str, collection_name: str, embedding_fn=None):
    """
    Indexa los chunks en ChromaDB con embeddings.
    """
    print("\n🔨 Indexando documentos en ChromaDB...")
    print(f"   Directorio: {persist_dir}")
    print(f"   Colección: {collection_name}")
    
    # Inicializar embeddings (o reutilizar instancia ya creada)
    print("\n   Inicializando embeddings...")
    embedding_fn = embedding_fn or init_embeddings()
    print(f"   ✓ Embeddings listos: {type(embedding_fn).__name__}")
    
    # Crear índice
    print("\n   Creando vectorstore...")
    vectorstore = create_chroma_index(
        persist_dir=persist_dir,
        embedding_fn=embedding_fn,
        collection_name=collection_name
    )
    print("   ✓ Vectorstore creado")
    
    # Agregar documentos en batches
    # Batches de 50 chunks. Cada add_texts genera embeddings e inserta en ChromaDB.
    # IDs secuenciales globales ("chunk_0", "chunk_1"...); si el corpus no cambia
    # de tamaño, re-ejecutar sobreescribe por upsert implícito en lugar de duplicar.
    print("\n   Agregando documentos...")
    batch_size = 50
    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i:i + batch_size]
        batch_num = i // batch_size + 1
        
        try:
            texts = [doc.page_content for doc in batch]
            metadatas = [doc.metadata for doc in batch]
            ids = [f"chunk_{i+j}" for j in range(len(batch))]
            
            vectorstore.add_texts(texts=texts, metadatas=metadatas, ids=ids)
            print(f"   ✓ Batch {batch_num}/{total_batches}: {len(batch)} chunks indexados")
            
        except Exception as e:
            print(f"   ✗ Error en batch {batch_num}: {e}")
            continue
    
    print(f"\n✓ Indexación completada!")
    return vectorstore


def verify_index(vectorstore, num_samples: int = 3):
    """
    Verifica que el índice funcione correctamente con búsquedas de prueba.
    """
    print("\n🔍 Verificando índice con búsquedas de prueba...")
    
    test_queries = [
        "¿Qué es el acoso laboral?",
        "¿Cuáles son las prestaciones sociales?",
        "¿Cuántas horas es la jornada laboral?",
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n   Prueba {i}: {query}")
        try:
            results = vectorstore.similarity_search_with_score(query, k=2)
            for j, (doc, score) in enumerate(results, 1):
                print(f"      {j}. Score: {score:.4f}")
                print(f"         Documento: {doc.metadata.get('id_documento', 'N/A')}")
                print(f"         Tipo: {doc.metadata.get('tipo_documento', 'N/A')}")
                print(f"         Texto: {doc.page_content[:100]}...")
        except Exception as e:
            print(f"      ✗ Error: {e}")
    
    print("\n✓ Verificación completada")


def main():
    """
    Función principal de indexación.
    """
    print("=" * 70)
    print("INDEXACIÓN DE PDFs - NORMATIVA LABORAL COLOMBIANA")
    print("=" * 70)
    print(f"Inicio: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # 1. Cargar configuración
    print("\n1️⃣  CONFIGURACIÓN")
    load_settings()
    
    corpus_path = "src/corpus"
    # Lee rutas desde .env; los defaults coinciden con los del resto del sistema.
    persist_dir = os.getenv("CHROMA_PERSIST_DIR", "./data/chroma")
    collection_name = os.getenv("CHROMA_COLLECTION_NAME", "normativa_laboral")
    
    print(f"   Corpus: {corpus_path}")
    print(f"   ChromaDB: {persist_dir}")
    print(f"   Colección: {collection_name}")
    
    # 2. Cargar PDFs
    print("\n2️⃣  CARGA DE DOCUMENTOS")
    documents = load_pdf_documents(corpus_path)
    
    if not documents:
        print("\n❌ No se encontraron documentos para indexar.")
        return
    
    # Estadísticas
    # Conteo por tipo_documento a nivel de página, no de documento único.
    # Un PDF de 50 páginas suma 50 al conteo de su tipo.
    tipos = {}
    for doc in documents:
        tipo = doc.metadata.get("tipo_documento", "DESCONOCIDO")
        tipos[tipo] = tipos.get(tipo, 0) + 1
    
    print("\n   📊 Estadísticas por tipo:")
    for tipo, count in sorted(tipos.items()):
        print(f"      {tipo}: {count} páginas")
    
    # Inicializar embeddings una vez para reutilizarlos en chunking + indexación
    print("\n   Inicializando embeddings para chunking semántico...")
    embedding_fn = init_embeddings()

    # 3. Dividir en chunks
    print("\n3️⃣  DIVISIÓN EN CHUNKS")
    chunks = split_documents(documents, embedding_fn=embedding_fn)
    
    # 4. Indexar
    print("\n4️⃣  INDEXACIÓN")
    vectorstore = index_documents(chunks, persist_dir, collection_name, embedding_fn=embedding_fn)
    
    # 5. Verificar
    print("\n5️⃣  VERIFICACIÓN")
    verify_index(vectorstore)
    
    # Resumen final
    print("\n" + "=" * 70)
    print("✅ INDEXACIÓN COMPLETADA EXITOSAMENTE")
    print("=" * 70)
    print(f"\n📈 Resumen:")
    print(f"   • Documentos procesados: {len(set(d.metadata.get('filename') for d in documents))}")
    print(f"   • Páginas totales: {len(documents)}")
    print(f"   • Chunks indexados: {len(chunks)}")
    print(f"   • Base de datos: {persist_dir}")
    print(f"   • Colección: {collection_name}")
    print(f"\n⏰ Fin: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n🚀 ¡La base de conocimiento está lista para usar en tu RAG!")
    print("=" * 70)


if __name__ == "__main__":
    main()
