"""
Script para limpiar la base de datos de ChromaDB y empezar de cero.
Útil cuando hay problemas con la indexación.
"""
import chromadb
from pathlib import Path
import shutil

persist_dir = "./data/chroma"
collection_name = "normativa_laboral"

print("=" * 80)
print("LIMPIEZA DE CHROMADB")
print("=" * 80)

# Verificar si existe el directorio
if not Path(persist_dir).exists():
    print(f"\n✓ El directorio {persist_dir} no existe (ya está limpio)")
    print("=" * 80)
    exit(0)

print(f"\n⚠️  ADVERTENCIA: Esto borrará toda la base de datos en:")
print(f"   {persist_dir}")
print(f"   Colección: {collection_name}")

respuesta = input("\n¿Continuar? (escribe 'SI' para confirmar): ")

if respuesta.strip().upper() != "SI":
    print("\n❌ Cancelado")
    print("=" * 80)
    exit(0)

try:
    # Opción 1: Borrar solo la colección
    print("\n🗑️  Eliminando colección...")
    client = chromadb.PersistentClient(path=persist_dir)
    
    try:
        client.delete_collection(name=collection_name)
        print(f"   ✓ Colección '{collection_name}' eliminada")
    except Exception as e:
        print(f"   • Colección no existía o error: {e}")
    
    # Opción 2: Borrar todo el directorio (más seguro)
    print(f"\n🗑️  Eliminando directorio completo...")
    shutil.rmtree(persist_dir)
    print(f"   ✓ Directorio {persist_dir} eliminado")
    
    print("\n✅ Limpieza completada!")
    print("\nAhora puedes ejecutar:")
    print("   python index_pdfs.py")
    
except Exception as e:
    print(f"\n❌ ERROR durante la limpieza: {e}")

print("=" * 80)
