# Módulo de configuración centralizada del sistema RAG.
# Gestiona la inicialización de LLMs y embeddings usados en el grafo LangGraph.
# Debe cargarse antes que cualquier otro módulo que dependa de variables de entorno.
import os
from typing import Literal

from dotenv import load_dotenv


def load_settings() -> None:
    # Carga las variables de entorno desde el archivo .env al inicio de la aplicación.
    # override=True garantiza que los valores del .env sobreescriban variables del sistema operativo,
    # evitando conflictos en entornos donde ya existan keys definidas a nivel de SO.
    load_dotenv(override=True)


def init_gemini_llm():
    # Alias de compatibilidad: migrado a Groq para unificar proveedor y aumentar throughput.
    return init_groq_llm(temperature=0)



def init_groq_llm(temperature: float = 0.2, model: str | None = None):
    # Inicializa el LLM vía Groq.
    # Por defecto se usa llama-3.3-70b-versatile para clasificación, routing y generación.
    # El modelo también puede ajustarse por variable de entorno GROQ_MODEL.
    from langchain_groq import ChatGroq

    return ChatGroq(
        model=model or os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
        temperature=temperature,
    )


def init_verification_llm(temperature: float = 0):
    # LLM dedicado a verificación: por defecto usa openai/gpt-oss-120b.
    # Puede sobreescribirse con GROQ_VERIFICATION_MODEL.
    return init_groq_llm(
        temperature=temperature,
        model=os.getenv("GROQ_VERIFICATION_MODEL", "openai/gpt-oss-120b"),
    )

# Inicializa el modelo de embeddings según el provider indicado.
# Jerarquía de resolución: parámetro explícito > variable EMBEDDINGS_PROVIDER en .env > valor por defecto "local".
def init_embeddings(provider: Literal["gemini", "openai", "local", "huggingface"] | None = None):
    """
    Initialize embeddings provider.
    
    Providers:
    - gemini: Google Gemini embeddings (requires API key)
    - openai: OpenAI embeddings (requires API key)
    - local/huggingface: Free local embeddings using Sentence Transformers (no API key needed)
    """
    embeddings_provider = provider or os.getenv("EMBEDDINGS_PROVIDER", "local")
    
    # Embeddings de OpenAI. Requiere OPENAI_API_KEY definida en el entorno.
    if embeddings_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()
    
    # Embeddings de Google Gemini. Requiere GOOGLE_API_KEY definida en el entorno.
    if embeddings_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Embeddings locales sin costo ni API key. Opción por defecto del sistema.
    if embeddings_provider in ["local", "huggingface"]:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # paraphrase-multilingual-MiniLM-L12-v2: modelo multilingüe optimizado para español.
        model_name = os.getenv(
            "EMBEDDINGS_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        # El nombre del modelo puede sobreescribirse con la variable EMBEDDINGS_MODEL en el .env.
        # normalize_embeddings=True normaliza los vectores a magnitud unitaria,
        # haciendo similitud coseno y producto punto equivalentes en ChromaDB.
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    # Fallback silencioso: cualquier valor no reconocido en EMBEDDINGS_PROVIDER cae aquí.
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
