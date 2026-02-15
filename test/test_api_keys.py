"""
Script para verificar que las API keys de Google Gemini y Groq funcionan correctamente.
"""
import sys
from pathlib import Path

# Agregar el directorio raíz al path para importar desde src
root_dir = Path(__file__).parent.parent
sys.path.insert(0, str(root_dir))

import os
from dotenv import load_dotenv

def test_api_keys():
    """
    Verifica que las API keys estén configuradas y funcionen.
    """
    print("=" * 70)
    print("VERIFICACIÓN DE API KEYS")
    print("=" * 70)
    
    # Limpiar variables de entorno existentes para forzar recarga desde .env
    if "GOOGLE_API_KEY" in os.environ:
        print("\n⚠️  Limpiando variable de entorno del sistema...")
        del os.environ["GOOGLE_API_KEY"]
    
    # Cargar variables de entorno desde .env (con override=True para forzar)
    load_dotenv(override=True)
    print("✓ Variables cargadas desde .env")
    
    # Verificar Google API Key
    print("\n1️⃣  GOOGLE GEMINI API KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    if google_key:
        print(f"   ✓ API Key encontrada")
        print(f"   • Longitud: {len(google_key)} caracteres")
        print(f"   • Primeros caracteres: {google_key[:15]}...")
        print(f"   • Últimos caracteres: ...{google_key[-10:]}")
        
        # Verificar que no tenga comillas
        if google_key.startswith('"') or google_key.startswith("'"):
            print("   ❌ ERROR: La API key tiene comillas al inicio!")
            print("      Solución: Quita las comillas del archivo .env")
        else:
            print("   ✓ Formato correcto (sin comillas)")
        
        # Probar conexión
        print("\n   Probando conexión con Gemini...")
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
            
            llm = ChatGoogleGenerativeAI(
                model="gemini-2.5-flash-lite",
                temperature=0,
            )
            
            response = llm.invoke("Di 'OK' si me recibes")
            print(f"   ✓ CONEXIÓN EXITOSA!")
            print(f"   • Respuesta de Gemini: {response.content}")
            
        except Exception as e:
            print(f"   ❌ ERROR en la conexión:")
            print(f"      {str(e)[:200]}")
    else:
        print("   ❌ API Key NO encontrada en las variables de entorno")
    
    # Verificar Groq API Key
    print("\n2️⃣  GROQ API KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    
    if groq_key:
        print(f"   ✓ API Key encontrada")
        print(f"   • Longitud: {len(groq_key)} caracteres")
        print(f"   • Primeros caracteres: {groq_key[:15]}...")
        
        # Verificar que no tenga comillas
        if groq_key.startswith('"') or groq_key.startswith("'"):
            print("   ❌ ERROR: La API key tiene comillas al inicio!")
        else:
            print("   ✓ Formato correcto (sin comillas)")
        
        # Probar conexión
        print("\n   Probando conexión con Groq...")
        try:
            from langchain_groq import ChatGroq
            
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                temperature=0,
            )
            
            response = llm.invoke("Di 'OK' si me recibes")
            print(f"   ✓ CONEXIÓN EXITOSA!")
            print(f"   • Respuesta de Groq: {response.content[:100]}")
            
        except Exception as e:
            print(f"   ❌ ERROR en la conexión:")
            print(f"      {str(e)[:200]}")
    else:
        print("   ❌ API Key NO encontrada en las variables de entorno")
    
    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)
    
    if google_key and groq_key:
        print("\n✅ Ambas API keys están configuradas correctamente")
        print("\n🚀 Tu sistema RAG puede usar:")
        print("   • Gemini para clasificación y verificación")
        print("   • Groq para generación de respuestas")
        print("   • Embeddings locales (sin costo)")
    else:
        print("\n⚠️  Algunas API keys faltan. Revisa tu archivo .env")
    
    print("=" * 70)


if __name__ == "__main__":
    test_api_keys()
