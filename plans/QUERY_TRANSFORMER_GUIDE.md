# 🔄 Guía de Transformación de Consultas (Query Transformer)

## 📋 Descripción General

El módulo `query_transformer.py` implementa estrategias avanzadas de mejora semántica para RAG (Retrieval Augmented Generation). El sistema **detecta automáticamente** qué estrategia aplicar a cada consulta:

### Estrategias Implementadas

1. **HyDE (Hypothetical Document Embeddings)**
   - Para preguntas cortas o ambiguas
   - Genera un documento hipotético que respondería la pregunta
   - Usa ese documento para búsqueda semántica mejorada

2. **Query Decomposition**
   - Para consultas complejas con múltiples preguntas
   - Descompone en sub-consultas más simples
   - Resuelve cada una por separado
   - Combina resultados

3. **MultiQueryRetriever**
   - Para preguntas mal formuladas o ambiguas
   - Genera múltiples reformulaciones
   - Combina resultados de todas las variaciones

---

## 🎯 Integración en el Grafo

El nodo `query_transform_node` se ejecuta en el siguiente flujo:

```
classify 
    ↓
query_transform ← NUEVO NODO
    ↓
tool_calling
    ↓
retrieve
    ↓
generate
    ↓
verify
```

### Lógica de Ejecución

- ✅ Se ejecuta para consultas: `legal_specific`, `procedural`, `general_laboral`, `calculation`, `resume`
- ⏭️ Se salta para: `general` (consultas sin contexto laboral)

---

## 📚 Uso en el Código

### Opción 1: Función de Alto Nivel (Recomendado)

```python
from src.query_transformer import transform_query
from src.config import init_groq_llm, init_embeddings
from src.vectorstore import load_chroma_index

# Inicializar componentes
llm = init_groq_llm(temperature=0.1)
embeddings = init_embeddings()
vectorstore = load_chroma_index("./data/chroma", embeddings, "normativa_laboral")

# Transformar consulta (detecta automáticamente estrategia)
result = transform_query(
    question="¿Cuál es la diferencia entre despido con causa y sin causa?",
    llm=llm,
    vectorstore=vectorstore,
    k=4  # documentos a recuperar
)

# Resultados
print(f"Tipo: {result['query_type']}")  # "hyde", "decomposition", o "multi_query"
print(f"Consultas: {result['transformed_queries']}")
print(f"Documentos: {result['documents']}")
print(f"Metadata: {result['metadata']}")
```

### Opción 2: Clase QueryTransformer (Más Control)

```python
from src.query_transformer import QueryTransformer

transformer = QueryTransformer(llm, vectorstore)

# Detectar tipo de consulta
query_type = transformer.detect_query_type(
    "¿Qué derechos tengo? ¿Y mis obligaciones?"
)

# Aplicar estrategia específica
if query_type == "hyde":
    hypo_doc, documents = transformer.hyde_search(question, k=4)
    
elif query_type == "decomposition":
    sub_queries, documents = transformer.decomposed_search(question, k=4)
    
else:  # multi_query
    queries, documents = transformer.multi_query_retrieval(question, k=4)
```

---

## 🔍 Detección de Tipo de Consulta

El sistema usa estos criterios:

### HyDE se aplica cuando:
- ✅ Menos de 20 palabras
- ✅ Una sola pregunta (un solo `?`)
- ✅ Sin conectores como "y", "o", "además"
- ✅ Sin patrones de comparación

**Ejemplos:**
- "¿Qué es una indemnización?"
- "¿Cuál es el salario mínimo?"
- "Explica sobre las prestaciones sociales"

### Query Decomposition se aplica cuando:
- ✅ Más de 20 palabras Y (múltiples preguntas O conectores)
- ✅ Múltiples signos de interrogación
- ✅ Contiene palabras como: "y", "o", "además", "comparar", "diferencia"
- ✅ Contiene condicionales: "si", "en caso de"

**Ejemplos:**
- "¿Cuáles son los derechos y cuáles son las obligaciones?"
- "¿Cuál es la diferencia entre A y B?"
- "¿Cómo se calcula? Y además, ¿qué derechos tengo?"

### Multi-Query se aplica cuando:
- ✅ Fallan las otras estrategias
- ✅ Preguntas ambiguas o mal formuladas

---

## 🧪 Ejecutar Pruebas

### Script de Demostración

```bash
python test_query_transformer.py
```

Ejecuta 5 ejemplos completos:
1. Detección automática de tipo de consulta
2. HyDE en acción
3. Query Decomposition en acción
4. Multi-Query Retrieval en acción
5. Función de alto nivel

### Resultados Esperados

```
EJEMPLO 1: Detección automática de tipo de consulta
📝 Consulta: ¿Cuál es el salario mínimo?
   → Tipo detectado: hyde

📝 Consulta: ¿Cuáles son mis derechos y obligaciones?
   → Tipo detectado: decomposition

EJEMPLO 2: HyDE (Hypothetical Document Embeddings)
📝 Pregunta: ¿Qué son las prestaciones sociales?
🔄 Proceso HyDE:
   1. Generando documento hipotético...
   2. Usando ese documento para buscar...

📚 Documento hipotético generado (547 caracteres):
   "Las prestaciones sociales son beneficios..."
✓ Documentos recuperados: 4
```

---

## 📊 Información en RAGState

Después de `query_transform_node`, el estado contiene:

```python
state = {
    "query": "Pregunta original",
    "query_type": "hyde" | "decomposition" | "multi_query" | "none" | "fallback",
    "transformed_queries": [
        "Documento hipotético o sub-consulta 1",
        "Sub-consulta 2",
        ...
    ],
    "metadata": {
        "query_transform": {
            "original_question": "...",
            "transformer_used": "QueryTransformer",
            "num_transformed": 3,
            "num_documents_retrieved": 4,
        },
        # ... otros campos
    },
    # ... resto del estado
}
```

---

## 🔧 Parámetros Personalizables

En `query_transform_node`:

```python
# Alterar LLM para transformación
llm = init_groq_llm(temperature=0.1)  # Aumentar para más creatividad

# Alterar número de documentos a recuperar
k = 8  # Por defecto es 4

# En transform_query()
result = transform_query(
    question="...",
    llm=llm,
    vectorstore=vectorstore,
    k=6  # Más documentos = más contexto = más latencia
)
```

---

## ⚙️ Comportamiento en Errores

### Si ocurre error en transformación:

```python
state["query_type"] = "fallback"
state["transformed_queries"] = [query]  # Usa consulta original
state["metadata"]["query_transform_error"] = "Error message"
```

El sistema continúa con la consulta original, evitando bloquear el pipeline.

### Si falla MultiQueryRetriever:

```
⚠️ Error en MultiQueryRetriever: [error details]
   Fallback a HyDE...
```

Se intenta automáticamente con HyDE.

---

## 📈 Casos de Uso

### Legal/Normativa (Recomendado para este sistema)
```python
# HYDE - Preguntas específicas
"¿Cuál es el artículo 25 del código laboral?"

# DECOMPOSITION - Comparaciones
"¿Cuál es la diferencia entre despido directo e indirecto?"

# DECOMPOSITION - Múltiples aspectos
"¿Cuáles son mis derechos, obligaciones y beneficios como empleado?"
```

### Optimización de Costos

```python
# HyDE es más barato (1 búsqueda)
# Query Decomposition es más caro (múltiples búsquedas)
# Multi-Query es medio (varía según reformulaciones)

# Para consultas muy simples
query_type = "hyde"  # Reducir llamadas a LLM
```

---

## 🚀 Mejores Prácticas

### ✅ HACER

1. **Use `transform_query()` para dejar que el sistema decida:**
   ```python
   result = transform_query(question, llm, vectorstore)
   ```

2. **Ajuste `k` según el tipo de consulta:**
   ```python
   # Consultas simples: k=3-4
   # Consultas complejas: k=6-8
   ```

3. **Monitoree metadata para entender qué estrategia se aplicó:**
   ```python
   strategy = result["metadata"]["query_type"]
   print(f"Estrategia usada: {strategy}")
   ```

### ❌ EVITAR

1. **No forzar una estrategia si no la necesita:**
   ```python
   # ❌ Malo
   hypo_doc, docs = transformer.hyde_search("¿A y B?", k=10)
   
   # ✅ Bueno
   result = transform_query("¿A y B?", llm, vectorstore)
   ```

2. **No ignorar los errores de transformación:**
   ```python
   # ❌ Malo
   if result["query_type"] == "fallback":
       pass  # Ignorar
   
   # ✅ Bueno
   if result["query_type"] == "fallback":
       log_warning(f"Transformación falló: {result['metadata']}")
   ```

---

## 📝 Logging y Debugging

Para ver el flujo completo:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Ahora verás:
# ✨ Transformación de consulta: HYDE
#    Consulta original: ¿Cuál es...?
#    ✓ Tipo de transformación: hyde
#    ✓ Consultas generadas: 1
```

---

## 📦 Dependencias

Verificar que estén instaladas:

```bash
pip install langchain-core
pip install langchain-community
pip install langchain-groq
pip install langchain-huggingface
```

Ver [requirements.txt](requirements.txt) para versiones.

---

## 🔗 Archivos Relacionados

- **[src/query_transformer.py](src/query_transformer.py)** - Módulo principal
- **[src/graph.py](src/graph.py)** - Integración en el grafo (nodo `query_transform_node`)
- **[test_query_transformer.py](test_query_transformer.py)** - Script de pruebas
- **[src/vectorstore.py](src/vectorstore.py)** - Acceso a ChromaDB
- **[src/config.py](src/config.py)** - Inicialización de LLM y embeddings

---

## ❓ Preguntas Frecuentes

### ¿Por qué mi consulta no se transforma?

**Posibles razones:**
1. Es clasificada como `general` → Se salta transformación (por diseño)
2. Ocurrió error y se usó fallback
3. El LLM no detectó patrón de descomposición

**Solución:** Revisar `state["metadata"]["query_transform"]`

### ¿Cómo reduzco la latencia?

1. Reducir `k` (documentos a recuperar): `k=2` en lugar de `k=4`
2. Usar HyDE preferentemente (menos llamadas a LLM)
3. Reducir `temperature` del LLM

### ¿Puedo usar otro LLM?

**Sí:** Solo pasar otro LLM compatible:
```python
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model="gpt-4")
result = transform_query(question, llm, vectorstore)
```

---

## 📞 Soporte

Para reportar issues o sugerencias, incluir:
- Consulta exacta que falló
- Error específico
- Contexto (clasificación, tipo de consulta)
- Resultado deseado vs actual
