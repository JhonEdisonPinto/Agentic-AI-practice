import os
from typing import Literal

from dotenv import load_dotenv


def load_settings() -> None:
    # Carga todo del archivo .env si está presente.
    # override=True Asegura que no haya problemas del .env por valores dentro del sistema.
    load_dotenv(override=True)


def init_gemini_llm():
    # Gemini 2.5 Flash Para clasificar y validar.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )


def init_groq_llm():
    # Groq model for response generation.
    from langchain_groq import ChatGroq

    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.2,
    )


def init_embeddings(provider: Literal["gemini", "openai", "local", "huggingface"] | None = None):
    """
    Initialize embeddings provider.
    
    Providers:
    - gemini: Google Gemini embeddings (requires API key)
    - openai: OpenAI embeddings (requires API key)
    - local/huggingface: Free local embeddings using Sentence Transformers (no API key needed)
    """
    embeddings_provider = provider or os.getenv("EMBEDDINGS_PROVIDER", "local")

    if embeddings_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()
    
    if embeddings_provider == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    
    # Default: Free local embeddings (no API key required)
    if embeddings_provider in ["local", "huggingface"]:
        from langchain_huggingface import HuggingFaceEmbeddings
        
        # Modelo multilingüe optimizado para español
        model_name = os.getenv(
            "EMBEDDINGS_MODEL",
            "paraphrase-multilingual-MiniLM-L12-v2"
        )
        
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
    
    # Fallback to local embeddings
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(
        model_name="paraphrase-multilingual-MiniLM-L12-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )
