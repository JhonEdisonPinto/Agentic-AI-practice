# 🏛️ RAG de Normativa Laboral Colombiana

Sistema de Recuperación Aumentada por Generación (RAG) especializado en normativa laboral colombiana, implementado con **LangGraph** y **ChromaDB**. El sistema utiliza embeddings locales gratuitos y proporciona respuestas precisas basadas en leyes, decretos y sentencias.

<img src="https://img.shields.io/badge/Python-3.11+-blue.svg" alt="Python">
<img src="https://img.shields.io/badge/LangChain-1.0+-green.svg" alt="LangChain">
<img src="https://img.shields.io/badge/ChromaDB-Vector%20Store-orange.svg" alt="ChromaDB">
<img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License">

## 📋 Descripción

Este proyecto implementa un sistema RAG avanzado que permite consultar la normativa laboral colombiana de forma conversacional. El sistema:

- 🔍 **Recupera documentos relevantes** de una base de más de 74 PDFs (leyes, decretos y sentencias)
- 🤖 **Clasifica consultas** automáticamente para optimizar la búsqueda
- 🛠️ **Ejecuta herramientas especializadas** para cálculos, búsquedas avanzadas y comparaciones
- 📝 **Genera respuestas** contextuales y precisas basadas en la normativa real
- ✅ **Verifica la calidad** de cada respuesta generada

## ✨ Características Principales

### 🔄 Pipeline LangGraph (5 Nodos)

1. **Classify**: Clasifica la consulta (legal_specific, procedural, general, calculation)
2. **Tool Calling**: Determina si ejecutar herramientas especializadas
3. **Retrieve**: Recupera documentos relevantes de ChromaDB
4. **Generate**: Genera respuesta usando LLMs (Gemini/Groq)
5. **Verify**: Verifica calidad y exactitud de la respuesta

### 🛠️ 5 Herramientas Especializadas

1. **`search_by_document_type`**: Busca por tipo específico (LEY, DECRETO, SENTENCIA)
2. **`search_by_year_range`**: Filtra documentos por rango de años
3. **`calculate_prestaciones_sociales`**: Calcula liquidaciones (cesantías, prima, vacaciones)
4. **`extract_specific_article`**: Extrae artículos específicos de documentos
5. **`compare_documents`**: Compara tratamiento de temas entre documentos

### 💾 Base de Conocimiento

- **74 documentos legales** indexados
- **Leyes laborales** (1010/2006, 1562/2012, 50/1990, etc.)
- **Decretos** (1072/2015, 2663/1950, etc.)
- **Sentencias** de la Corte Constitucional
- **Embeddings multilingües** optimizados para español

### 🆓 Modelos Gratuitos (Sin Costo)

- **Embeddings locales**: `paraphrase-multilingual-MiniLM-L12-v2`
- **Funciona offline** después de la primera descarga
- **Sin límites de uso** ni costos de API

## 🚀 Inicio Rápido

### 1. Requisitos Previos

- Python 3.11 o superior
- Git

### 2. Clonar el Repositorio

```bash
git clone https://github.com/tu-usuario/Agentic-AI-practice.git
cd Agentic-AI-practice
```

### 3. Crear Entorno Virtual

**Windows:**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instalar Dependencias

```bash
pip install -r requirements.txt
```

### 5. Configurar Variables de Entorno

```bash
# Copiar el archivo de ejemplo
cp .env.example .env

# Editar .env y agregar tus API keys
```

**Archivo `.env`:**
```env
# Obligatorias para funcionalidad completa
GOOGLE_API_KEY=tu-api-key-de-google-aqui
GROQ_API_KEY=tu-api-key-de-groq-aqui

# Configuración de embeddings (local = gratis)
EMBEDDINGS_PROVIDER=local
EMBEDDINGS_MODEL=paraphrase-multilingual-MiniLM-L12-v2

# ChromaDB
CHROMA_PERSIST_DIR=./data/chroma
CHROMA_COLLECTION_NAME=normativa_laboral
```

**Obtener API Keys:**
- **Google Gemini**: https://aistudio.google.com/app/apikey (Gratis)
- **Groq**: https://console.groq.com/keys (Gratis)

### 6. Indexar Documentos

```bash
# Indexación rápida (5 PDFs clave - ~2 minutos)
python quick_index.py

# O indexación completa (74 PDFs - ~15 minutos)
python index_pdfs.py
```

### 7. Probar el Sistema

```bash
# Verificar API keys
python test_api_keys.py

# Probar herramientas
python test_tools.py

# Probar RAG completo
python test_rag_graph.py

# Interfaz web con Streamlit
streamlit run app.py
```

## 📁 Estructura del Proyecto

```
Agentic-AI-practice/
├── src/
│   ├── config.py           # Configuración de LLMs y embeddings
│   ├── graph.py            # Pipeline LangGraph (5 nodos)
│   ├── vectorstore.py      # Funciones de ChromaDB
│   ├── tools.py            # 5 herramientas especializadas
│   └── corpus/             # 74 PDFs de normativa laboral
├── data/
│   └── chroma/             # Base de datos vectorial (generada)
├── app.py                  # Interfaz Streamlit
├── index_pdfs.py           # Script de indexación completa
├── quick_index.py          # Script de indexación rápida
├── test_*.py               # Scripts de prueba
├── PRUEBASCODIGO.ipynb     # 📓 Notebook interactivo con pruebas completas
├── requirements.txt        # Dependencias Python
├── .env.example            # Plantilla de configuración
└── README.md               # Este archivo
```

## 🔧 Uso del Sistema

### Consultas de Ejemplo

```python
from src.config import load_settings
from src.graph import build_graph

# Inicializar
load_settings()
graph = build_graph()

# Hacer consulta
result = graph.invoke({
    "query": "¿Qué es el acoso laboral según la ley colombiana?",
    "classification": "",
    "documents": [],
    "tool_results": None,
    "answer": "",
    "verification": {},
    "metadata": {}
})

print(result["answer"])
```

### Usando Herramientas Directamente

```python
from src.tools import calculate_prestaciones_sociales

# Calcular prestaciones
resultado = calculate_prestaciones_sociales.invoke({
    "salario_mensual": 2500000,
    "dias_trabajados": 360,
    "años_servicio": 1.0
})

print(f"Total prestaciones: ${resultado['total_prestaciones']:,.2f}")
```

### 🌐 Aplicación Web con Streamlit

#### Iniciar la aplicación

```bash
# Opción 1: Usando el script de inicio
python run_app.py

# Opción 2: Directamente con streamlit
streamlit run app.py

# Opción 3: Con configuración personalizada
streamlit run app.py --server.port 8501
```

La aplicación se abrirá automáticamente en http://localhost:8501

#### Características de la UI

- **📝 Área de consulta**: Entrada de texto para realizar preguntas
- **💡 Ejemplos predefinidos**: 6 consultas de ejemplo para cada tipo de herramienta
- **📊 Panel de resultados**: 
  - Respuesta principal con formato markdown
  - Información del proceso (clasificación, herramientas, calidad)
  - Documentos consultados con IDs y metadatos
  - Detalles técnicos expandibles
- **📜 Historial**: Registro de las últimas 5 consultas
- **⚡ Indicador de progreso**: Visualización del proceso en tiempo real

#### Tipos de consultas soportadas

1. **Cálculos**: Prestaciones sociales, liquidaciones
2. **Documentos específicos**: Leyes, decretos, sentencias por número
3. **Búsqueda por rango**: Normativa publicada entre años específicos
4. **Artículos específicos**: Extracción de artículos particulares
5. **Comparaciones**: Diferencias entre dos documentos legales
6. **Resúmenes**: Vista general de un documento completo

## 🏗️ Arquitectura

### Flujo del Sistema

```
Usuario → Classify → Tool Calling → Retrieve → Generate → Verify → Respuesta
              ↓           ↓            ↓          ↓          ↓
           Gemini      5 Tools    ChromaDB    Groq     Gemini
```

### Tecnologías Utilizadas

| Componente | Tecnología | Propósito |
|------------|------------|-----------|
| **Orquestación** | LangGraph | Pipeline de 5 nodos |
| **Vector DB** | ChromaDB | Almacenamiento y búsqueda |
| **Embeddings** | Sentence Transformers | Modelo multilingüe local |
| **LLM Clasificación** | Google Gemini 2.0 Flash | Clasificar consultas |
| **LLM Generación** | Groq (Llama 3.1 70B) | Generar respuestas |
| **LLM Verificación** | Google Gemini 2.0 Flash | Validar calidad |
| **Framework** | LangChain 1.0+ | Integración de componentes |
| **UI** | Streamlit | Interfaz web |

## 📊 Rendimiento

- **Tiempo de respuesta**: 3-5 segundos
- **Precisión de recuperación**: ~90% (top-3)
- **Documentos indexados**: 74 PDFs
- **Chunks en base de datos**: ~2,500
- **Tamaño modelo embeddings**: 471 MB (primera descarga)

## 🧪 Testing

> 📓 **Las pruebas interactivas completas están en `PRUEBASCODIGO.ipynb`** - Jupyter Notebook con ejemplos detallados, visualización de resultados y debugging paso a paso.

### Scripts de prueba CLI

```bash
# Test de embeddings
python test_embeddings_free.py

# Test de producción
python test_production_embeddings.py

# Test de herramientas
python test_tools.py

# Test del grafo completo
python test_rag_graph.py

# Test interactivo
python test_rag_graph.py interactive
```

### Ejecutar PRUEBASCODIGO.ipynb

```bash
# Con Jupyter
jupyter notebook PRUEBASCODIGO.ipynb

# O con VS Code
# Abre el archivo y ejecuta las celdas manualmente
```

## 📚 Documentación Adicional

### Agregar Nuevos Documentos

1. Coloca PDFs en `src/corpus/`
2. Ejecuta: `python index_pdfs.py`
3. Los documentos se indexarán automáticamente

### Cambiar Proveedor de Embeddings

```env
# En .env
EMBEDDINGS_PROVIDER=gemini  # o openai o local
```

### Modificar el Pipeline

Edita `src/graph.py` para agregar nodos o cambiar el flujo:

```python
graph.add_node("nuevo_nodo", nuevo_node_function)
graph.add_edge("classify", "nuevo_nodo")
```

## 🤝 Contribuir

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/nueva-funcionalidad`)
3. Commit tus cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Abre un Pull Request

## ⚠️ Limitaciones Conocidas

- Los embeddings locales son buenos pero no perfectos para español legal
- Requiere ~500MB de espacio para el modelo de embeddings
- La primera ejecución descarga el modelo (puede tardar)
- API keys de Gemini/Groq tienen límites de rate (gratis)

## 🔮 Mejoras Futuras

- [ ] Agregar más herramientas especializadas
- [ ] Implementar caché de respuestas frecuentes
- [ ] Soporte para consultas multi-documento
- [ ] Exportar respuestas en PDF
- [ ] Sistema de feedback de usuarios
- [ ] Fine-tuning del modelo de embeddings

## 📝 Licencia

Este proyecto está bajo la Licencia MIT. Ver archivo `LICENSE` para más detalles.

## 👤 Autor

- Jhon Edison Pinto - [@JhonEdisonPinto](https://github.com/JhonEdisonPinto)
- Juan Sebastian Hoyos Castillo - [@SebastianHoyoss](https://github.com/SebastianHoyoss)
- Jhogert David Bita Aldana - [@JhogertBita](https://github.com/JhogertBita)

**⭐ Si este proyecto te fue útil, considera darle una estrella en GitHub ⭐**
