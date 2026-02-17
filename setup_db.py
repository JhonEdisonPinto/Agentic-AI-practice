"""
Script para inicializar la base de datos ChromaDB en el despliegue.
Se ejecuta automáticamente si no existe la BD.
"""
import os
from pathlib import Path

def setup_database():
    """Verifica si existe la BD, si no la crea."""
    chroma_dir = Path("./data/chroma")
    
    if not chroma_dir.exists() or not list(chroma_dir.glob("*")):
        print("🔨 Base de datos no encontrada. Inicializando...")
        print("⚠️ NOTA: Los PDFs deben estar en src/corpus/")
        
        # Verificar si existen PDFs
        corpus_dir = Path("./src/corpus")
        if not corpus_dir.exists() or not list(corpus_dir.glob("*.pdf")):
            print("❌ ERROR: No se encontraron PDFs en src/corpus/")
            print("💡 Opción 1: Sube la carpeta data/ al repositorio")
            print("💡 Opción 2: Agrega los PDFs a src/corpus/")
            return False
        
        # Ejecutar indexación
        try:
            from index_pdfs import main
            main()
            print("✅ Base de datos creada exitosamente")
            return True
        except Exception as e:
            print(f"❌ Error al crear la base de datos: {e}")
            return False
    else:
        print("✅ Base de datos encontrada")
        return True

if __name__ == "__main__":
    setup_database()
