"""Script para iniciar la aplicación Streamlit de manera fácil."""
import subprocess
import sys
from pathlib import Path

# Lanzador equivalente a "streamlit run application/app.py --server.port=8501 --server.headless=true".
# Toda la carga del RAG ocurre dentro de application/app.py.

REPO_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = REPO_ROOT / "application" / "app.py"

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
            str(APP_PATH),
            "--server.port=8501",
            "--server.headless=true"
        ], cwd=str(REPO_ROOT))
    except KeyboardInterrupt:
        print("\n\n✅ Servidor detenido correctamente")
    except Exception as e:
        print(f"\n❌ Error al iniciar la aplicación: {e}")
