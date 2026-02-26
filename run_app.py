"""Script para iniciar la aplicación Streamlit de manera fácil."""
import subprocess
import sys

# Lanzador equivalente a "streamlit run app.py --server.port=8501 --server.headless=true".
# Toda la carga del RAG ocurre dentro de app.py.

if __name__ == "__main__":
    print("=" * 70)
    print("🚀 INICIANDO RAG NORMATIVA LABORAL COLOMBIANA")
    print("=" * 70)
    print()
    print("📍 La aplicación se abrirá en: http://localhost:8501")
    print("⌨️  Presiona Ctrl+C para detener el servidor")
    print()
    print("=" * 70)
    
    try:
        # sys.executable garantiza el mismo intérprete/entorno virtual activo.
        # Bloqueante: el control no retorna hasta que Streamlit termine.
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app.py",
            "--server.port=8501",
            "--server.headless=true"
        ])
    except KeyboardInterrupt:
        print("\n\n✅ Servidor detenido correctamente")
    except Exception as e:
        print(f"\n❌ Error al iniciar la aplicación: {e}")
