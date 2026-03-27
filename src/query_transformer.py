"""
Módulo de transformación de consultas (Query Transformation).

Implementa dos estrategias de mejora semántica para retrieval:

1. HyDE (Hypothetical Document Embeddings):
   - Para preguntas cortas, ambiguas o de una sola línea
   - Genera un documento hipotético que respondería la pregunta
   - Usa ese documento para buscar en la base vectorial

2. Query Decomposition:
   - Para consultas complejas con múltiples preguntas o condicionales
   - Descompone la consulta en sub-consultas más simples
   - Resuelve cada sub-consulta por separado
   - Combina resultados

3. MultiQueryRetriever:
   - Genera múltiples variaciones de la consulta original
   - Útil para preguntas ambiguas o mal formuladas
   - Combina resultados de todas las variaciones
"""

from typing import List, Dict, Any, Tuple
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.retrievers import MultiQueryRetriever
from langchain_core.language_models import BaseLanguageModel
from langchain_community.vectorstores import Chroma

import re
import json


class QueryTransformer:
    """
    Analizador y transformador de consultas para RAG.
    Detecta el tipo de consulta y aplica la estrategia más adecuada.
    """

    def __init__(self, llm: BaseLanguageModel, vectorstore: Chroma):
        """
        Inicializa el transformador de consultas.
        
        Args:
            llm: Modelo de lenguaje para análisis y generación
            vectorstore: Base de datos vectorial (Chroma)
        """
        self.llm = llm
        self.vectorstore = vectorstore

    def detect_query_type(self, question: str) -> str:
        """
        Detecta si la consulta es corta (HyDE) o múltiple (Query Decomposition).
        
        Criterios:
        - Corta: < 20 palabras, una pregunta principal
        - Múltiple: Contiene conectores (y, o, además), múltiples "?", condicionales
        
        Args:
            question: Consulta del usuario
            
        Returns:
            "hyde" | "decomposition" | "multi_query"
        """
        word_count = len(question.split())
        question_count = question.count("?")
        
        # Patrones de conectores que indican multiple consultas
        multi_patterns = [
            r'\by\s+', r'\bo\s+', r'\bu\s+',  # conectores básicos
            r'además', r'asimismo', r'también',  # añadidura
            r'sin embargo', r'no obstante', r'pero',  # contraste
            r'si\s+', r'en\s+caso\s+de',  # condicionales
            r'comparar', r'diferencia', r'vs\.', r'versus',  # comparaciones
        ]
        
        has_multi_pattern = any(re.search(pattern, question.lower()) for pattern in multi_patterns)
        
        # Lógica de decisión
        if word_count > 20 and (question_count > 1 or has_multi_pattern):
            return "decomposition"
        elif question_count > 1:
            return "decomposition"
        elif any(term in question.lower() for term in ["cuál es la diferencia", "comparar", "vs", "versus"]):
            return "decomposition"
        else:
            return "hyde"

    def hyde_search(self, question: str, k: int = 4) -> Tuple[str, List[Document]]:
        """
        HyDE: Hypothetical Document Embeddings.
        
        Genera un documento hipotético que respondería la pregunta,
        luego usa ese documento para buscar en la base vectorial.
        
        Pasos:
        1. Generar documento hipotético
        2. Buscar por similitud usando ese documento
        
        Args:
            question: Pregunta del usuario
            k: Número de documentos a recuperar
            
        Returns:
            Tupla (documento_hipotético, documentos_recuperados)
        """
        print(f"\n📚 HyDE: Generando documento hipotético...")
        
        # Prompt para generar documento hipotético
        hyde_prompt = ChatPromptTemplate.from_messages([
            ("system", 
             "Escribe un documento corto (6-10 líneas) que respondería la pregunta. "
             "No uses viñetas. Texto informativo continuo. "
             "Sé específico y detallado, como si fuera una sección de un documento oficial."),
            ("human", "Pregunta: {question}\n\nDocumento hipotético:")
        ])
        
        try:
            # Generar el documento hipotético
            messages = hyde_prompt.format_messages(question=question)
            response = self.llm.invoke(messages)
            hypo_doc = response.content
            
            print(f"   ✓ Documento hipotético generado ({len(hypo_doc)} caracteres)")
            print(f"   Preview: {hypo_doc[:100]}...")
            
            # Buscar en la base vectorial usando el documento hipotético
            hits = self.vectorstore.similarity_search(hypo_doc, k=k)
            
            print(f"   ✓ Recuperados {len(hits)} documentos")
            
            return hypo_doc, hits
            
        except Exception as e:
            print(f"   ⚠️ Error en HyDE: {str(e)}")
            # Fallback: búsqueda simple por pregunta
            hits = self.vectorstore.similarity_search(question, k=k)
            return question, hits

    def decompose_query(self, question: str) -> List[str]:
        """
        Query Decomposition: Descompone una consulta compleja en sub-consultas.
        
        Detecta múltiples preguntas o condicionales y las separa
        para resolver secuencialmente.
        
        Args:
            question: Consulta compleja
            
        Returns:
            Lista de sub-consultas
        """
        print(f"\n🔗 Query Decomposition: Analizando consulta compleja...")
        
        decomposition_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Eres un experto en descomposición de consultas complejas. "
             "Tu tarea es dividir la pregunta en sub-preguntas más simples y específicas. "
             "Cada sub-pregunta debe ser independiente y respondible. "
             "Responde en formato JSON con clave 'sub_queries' y arreglo de strings."),
            ("human",
             "Descompón esta consulta en sus preguntas componentes:\n\n"
             "Consulta: {question}\n\n"
             "Responde SOLO con JSON válido (sin markdown):")
        ])
        
        try:
            messages = decomposition_prompt.format_messages(question=question)
            response = self.llm.invoke(messages)
            
            # Extraer JSON de la respuesta
            response_text = response.content.strip()
            
            # Intentar parsear JSON directamente
            if response_text.startswith("{"):
                parsed = json.loads(response_text)
            else:
                # Buscar JSON en la respuesta
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    parsed = json.loads(json_match.group())
                else:
                    raise ValueError("No se encontró JSON en la respuesta")
            
            sub_queries = parsed.get("sub_queries", [question])
            
            if not isinstance(sub_queries, list):
                sub_queries = [sub_queries]
            
            print(f"   ✓ Consulta descompuesta en {len(sub_queries)} sub-consultas:")
            for i, sq in enumerate(sub_queries, 1):
                print(f"      {i}. {sq}")
            
            return sub_queries
            
        except Exception as e:
            print(f"   ⚠️ Error en decomposición (fallback): {str(e)}")
            # Fallback: separar por patrones simples
            sub_queries = self._simple_decompose(question)
            return sub_queries

    def _simple_decompose(self, question: str) -> List[str]:
        """
        Fallback para descomposición simple basada en patrones.
        Separa por conectores y signos de puntuación.
        """
        # Reemplazar conectores con delimitadores
        text = question
        for connector in [' y ', ' o ', ' u ']:
            text = re.sub(f'({connector})', '|SPLIT|', text, flags=re.IGNORECASE)
        
        # Dividir por delimitador
        parts = text.split('|SPLIT|')
        sub_queries = [p.strip().rstrip('?').strip() for p in parts if p.strip()]
        
        return sub_queries if len(sub_queries) > 1 else [question]

    def decomposed_search(self, question: str, k: int = 4) -> Tuple[List[str], List[Document]]:
        """
        Ejecuta búsqueda con Query Decomposition.
        
        1. Descomponer consulta
        2. Buscar documentos para cada sub-consulta
        3. Combinar y deduplicar resultados
        
        Args:
            question: Consulta compleja
            k: Documentos por sub-consulta
            
        Returns:
            Tupla (sub_consultas, documentos_combinados)
        """
        sub_queries = self.decompose_query(question)
        
        print(f"\n🔎 Ejecutando búsquedas para cada sub-consulta...")
        
        all_docs = {}
        doc_scores = {}
        
        for i, sub_q in enumerate(sub_queries, 1):
            print(f"   [{i}/{len(sub_queries)}] Buscando: {sub_q}")
            
            try:
                docs = self.vectorstore.similarity_search_with_score(sub_q, k=k)
                
                for doc, score in docs:
                    # Usar ID del documento como clave
                    doc_id = id(doc)
                    if doc_id not in all_docs:
                        all_docs[doc_id] = doc
                        doc_scores[doc_id] = score
                    else:
                        # Promediar scores si el documento aparece múltiples veces
                        doc_scores[doc_id] = (doc_scores[doc_id] + score) / 2
                
                print(f"      ✓ {len(docs)} documentos recuperados")
                
            except Exception as e:
                print(f"      ⚠️ Error en búsqueda: {str(e)}")
        
        # Ordenar por score y retornar top k
        sorted_docs = sorted(
            [(doc, score) for doc_id, (doc, score) in 
             zip(all_docs.keys(), [(all_docs[did], doc_scores[did]) for did in all_docs])],
            key=lambda x: x[1],
            reverse=True
        )[:k]
        
        combined_docs = [doc for doc, _ in sorted_docs]
        
        print(f"   ✓ Total de documentos únicos recuperados: {len(combined_docs)}")
        
        return sub_queries, combined_docs

    def multi_query_retrieval(self, question: str, k: int = 4) -> Tuple[List[str], List[Document]]:
        """
        MultiQueryRetriever: Genera múltiples perspectivas de la consulta.
        
        Crea reformulaciones de la pregunta original desde diferentes ángulos,
        luego combina los resultados para mayor cobertura semántica.
        
        Usa langchain_classic.retrievers.MultiQueryRetriever para automatizar
        la generación de variaciones y retrieval.
        
        Args:
            question: Pregunta original
            k: Documentos por variación
            
        Returns:
            Tupla (variaciones_generadas, documentos_combinados)
        """
        print(f"\n🎯 Multi-Query Retrieval: Generando variaciones...")
        
        try:
            # Configurar logging para ver las preguntas generadas
            import logging
            logging.basicConfig()
            logging.getLogger("langchain_classic.retrievers.multi_query").setLevel(logging.INFO)
            
            # Crear el retriever multi-query
            # MultiQueryRetriever genera automáticamente las variaciones de la pregunta
            multi_retriever = MultiQueryRetriever.from_llm(
                retriever=self.vectorstore.as_retriever(search_kwargs={"k": k}),
                llm=self.llm
            )
            
            print(f"   ✓ MultiQueryRetriever configurado")
            
            # Ejecutar el retriever - esto genera automáticamente las variaciones
            # y devuelve documentos combinados y deduplicados
            documents = multi_retriever.invoke(question)
            
            print(f"   ✓ {len(documents)} documentos recuperados (deduplicados)")
            
            # Retornar la pregunta original como lista de "consultas transformadas"
            # (MultiQueryRetriever maneja las variaciones internamente)
            transformed_queries = [question]  # La pregunta original
            
            return transformed_queries, documents
            
        except Exception as e:
            print(f"   ⚠️ Error en MultiQueryRetriever: {str(e)[:100]}")
            import traceback
            traceback.print_exc()
            
            # Fallback: búsqueda simple
            documents = self.vectorstore.similarity_search(question, k=k)
            return [question], documents


def transform_query(
    question: str,
    llm: BaseLanguageModel,
    vectorstore: Chroma,
    k: int = 4
) -> Dict[str, Any]:
    """
    Función de alto nivel que detecta el tipo de consulta y aplica
    la estrategia más adecuada.
    
    Args:
        question: Consulta del usuario
        llm: Modelo de lenguaje
        vectorstore: Base de datos vectorial
        k: Número de documentos a recuperar
        
    Returns:
        Diccionario con:
        - query_type: "hyde" | "decomposition" | "multi_query"
        - transformed_queries: Consultas generadas/transformadas
        - documents: Documentos recuperados
        - metadata: Información adicional
    """
    transformer = QueryTransformer(llm, vectorstore)
    
    # Detectar tipo de consulta
    query_type = transformer.detect_query_type(question)
    
    print(f"\n✨ Transformación de consulta: {query_type.upper()}")
    print(f"   Consulta original: {question}")
    
    if query_type == "decomposition":
        sub_queries, documents = transformer.decomposed_search(question, k=k)
        transformed_queries = sub_queries
        
    elif query_type == "multi_query":
        try:
            queries, documents = transformer.multi_query_retrieval(question, k=k)
            transformed_queries = queries
        except Exception as e:
            print(f"   ⚠️ Fallback a HyDE: {str(e)}")
            hypo_doc, documents = transformer.hyde_search(question, k=k)
            transformed_queries = [hypo_doc]
            query_type = "hyde"
    
    else:  # hyde (por defecto)
        hypo_doc, documents = transformer.hyde_search(question, k=k)
        transformed_queries = [hypo_doc]
    
    return {
        "query_type": query_type,
        "transformed_queries": transformed_queries,
        "documents": documents,
        "metadata": {
            "original_question": question,
            "transformer_used": transformer.__class__.__name__,
            "num_transformed": len(transformed_queries),
            "num_documents_retrieved": len(documents),
        }
    }
