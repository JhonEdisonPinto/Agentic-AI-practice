import os
from typing import Literal

from dotenv import load_dotenv


def load_settings() -> None:
    # Load environment variables from .env if present.
    load_dotenv()


def init_gemini_llm():
    # Gemini 2.5 Flash for classification and verification.
    from langchain_google_genai import ChatGoogleGenerativeAI

    return ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=0,
    )


def init_groq_llm():
    # Groq model for response generation.
    from langchain_groq import ChatGroq

    return ChatGroq(
        model="llama-3.1-70b-versatile",
        temperature=0.2,
    )


def init_embeddings(provider: Literal["gemini", "openai"] | None = None):
    # Embeddings provider can be set via env var or argument.
    embeddings_provider = provider or os.getenv("EMBEDDINGS_PROVIDER", "gemini")

    if embeddings_provider == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings()

    from langchain_google_genai import GoogleGenerativeAIEmbeddings

    return GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
