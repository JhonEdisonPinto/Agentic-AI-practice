# Guía de Pruebas de Tools - test_tools_cases.ipynb

## 📋 Descripción General

El notebook `test_tools_cases.ipynb` contiene pruebas exhaustivas del sistema RAG de normativa laboral colombiana. Incluye **11+ casos de uso** que verifican el funcionamiento de todas las **tools** disponibles.

---

## 🚀 Cómo Usar el Notebook

### 1. Requisitos Previos
- Python 3.8+
- ChromaDB instalado
- Dependencias de `requirements.txt`
- Variables de entorno en `.env`

### 2. Estructura del Notebook

El notebook está organizado en las siguientes secciones:

#### **Sección 1: Inicialización (Celdas 1-5)**
- Importación de librerías
- Verificación de disponibilidad de ChromaDB
- Limpieza de la base de datos (opcional)
- Indexación de PDFs (comentado, activar si es necesario)
- Inicialización del vectorstore

#### **Sección 2: Casos de Uso (Celdas 6-16)**
11 casos de prueba completos que verifican:
1. Búsqueda básica por similitud semántica
2. Búsqueda por tipo de documento
3. Búsqueda por rango de años
4. Cálculo de prestaciones sociales
5. Extracción de artículos específicos
6. Comparación de documentos
7. Resumen de documento completo
8. Búsqueda conceptualmente similar
9. Búsqueda con diferentes profundidades (k)
10. Pruebas de rendimiento/benchmarking
11. Manejo de errores y casos límite

#### **Sección 3: Conclusiones (Celda 17)**
- Resumen de resultados
- Estadísticas de ejecución
- Próximos pasos recomendados

---

## 🔧 Gestión de Base de Datos

### Verificar Disponibilidad (Celda 2)
```python
# Esta celda comprueba si ChromaDB está disponible
# Muestra el número de chunks indexados y estadísticas
```

**Expectedoutput:**
- ✅ ChromaDB DISPONIBLE
- 📊 Total de chunks: [número]
- 📋 Por tipo: LEY: [#], DECRETO: [#], etc.

---

### Limpiar Base de Datos (Celda 3)
```python
# DESCOMENTA para ejecutar
# Elimina toda la base de datos y reinicia
```

**Cuidado:** Esta acción es irreversible. Solo ejecutar si deseas reinicializar la BD.

---

### Indexar PDFs (Celda 4)
```python
# DESCOMENTA para ejecutar
# Indexa todos los PDFs del corpus en ChromaDB
```

**Proceso:**
1. Carga los embeddings
2. Crea el índice ChromaDB
3. Procesa cada PDF:
   - Extrae metadatos del nombre
   - Carga el contenido
   - Divide en chunks
   - Agrega al vectorstore

---

## 📊 Casos de Uso Detallados

### Caso 1: Búsqueda Básica
**Objetivo:** Verificar búsqueda por similitud semántica
**Consultas probadas:** 4 diferentes
**Métricas:** Número de resultados, tiempo de respuesta

---

### Caso 2: Búsqueda por Tipo
**Objetivo:** Filtrar por tipo de documento (LEY, DECRETO, SENTENCIA)
**Función usada:** `search_by_document_type()`
**Ejemplo:**
```python
# Buscar LEYs sobre "derechos del trabajador"
results = search_by_document_type("derechos del trabajador", "LEY", vectorstore)
```

---

### Caso 3: Rango de Años
**Objetivo:** Búsqueda temporal limitada
**Función usada:** `search_by_year_range()`
**Rangos probados:** 2000-2010, 2010-2020, 2020-2025
**Ejemplo:**
```python
# Buscar entre 2015 y 2020
results = search_by_year_range("contrato", 2015, 2020, vectorstore)
```

---

### Caso 4: Cálculo de Prestaciones
**Objetivo:** Calcular beneficios sociales de trabajadores
**Función usada:** `calculate_prestaciones_sociales()`
**Parámetros:** Salario, días trabajados, años de servicio
**Ejemplo:**
```python
# Calcular para 1 año de servicio
result = calculate_prestaciones_sociales(
    salario_mensual=1200000,  # COP
    dias_trabajados=360,
    años_servicio=1
)
# Retorna: cesantías, intereses, prima, vacaciones, total
```

---

### Caso 5: Artículos Específicos
**Objetivo:** Extraer artículos concretos de documentos
**Función usada:** `extract_specific_article()`
**Ejemplo:**
```python
# Extraer Artículo 3 de la LEY 1010
result = extract_specific_article("LEY_1010_2006", "3", vectorstore)
```

---

### Caso 6: Comparar Documentos
**Objetivo:** Comparar contenido entre dos leyes
**Función usada:** `compare_documents()`
**Ejemplo:**
```python
# Comparar LEY 1010 vs DECRETO 1072 sobre acoso laboral
result = compare_documents("LEY_1010_2006", "DECRETO_1072_2015", "acoso laboral", vectorstore)
```

---

### Caso 7: Resumen de Documento
**Objetivo:** Obtener contenido completo para resumir
**Función usada:** `resume_document()`
**Ejemplo:**
```python
# Obtener contenido completo de una ley
result = resume_document("LEY_1010_2006", vectorstore)
# Contiene el documento completo para que el LLM lo resuma
```

---

### Caso 8-11: Búsquedas Avanzadas
- **Caso 8:** Variaciones semánticas de la misma consulta
- **Caso 9:** Análisis de profundidad (k=1, 5, 10, 20)
- **Caso 10:** Benchmarking y rendimiento
- **Caso 11:** Manejo de errores y casos límite

---

## 📈 Interpretación de Resultados

### Métricas Clave
| Métrica | Esperado | Rango Aceptable |
|---------|----------|-----------------|
| Tiempo de búsqueda | <100ms | <500ms |
| Documentos encontrados | >0 | por consulta |
| Precisión | >80% | >60% |
| Chunks indexados | >1000 | >100 |

---

## ⚙️ Configuración de Embeddings

El notebook usa la configuración de embeddings del archivo `.env`:

```env
# Opciones: "openai", "gemini", "local", "huggingface"
EMBEDDINGS_PROVIDER=local
```

**Recomendaciones:**
- `local`: Gratis, sin API keys (recomendado para desarrollo)
- `openai`: Mejor calidad, requiere API key
- `gemini`: Equilibrio, requiere API key
- `huggingface`: Gratis, modelos de SentenceTransformers

---

## 🐛 Solución de Problemas

### Error: "Base de datos vacía"
**Solución:** Ejecuta la Celda 4 para indexar PDFs

```python
# Celda 4: Descomenta y ejecuta
# python index_pdfs.py
```

---

### Error: "ChromaDB no encontrado"
**Solución:** Verifica que el directorio existe:

```bash
ls -la ./data/chroma
# Debe existir y contener SQLite database
```

---

### Error: "Embeddings no disponibles"
**Solución:** Verifica la configuración en `.env` y las API keys:

```bash
# Para local embeddings, no hay requisitos adicionales
# Para OpenAI/Gemini, verifica OPENAI_API_KEY o GOOGLE_API_KEY
```

---

## 🎯 Próximos Pasos

1. **Expandir Corpus:** Agregar más documentos al directorio `src/corpus/`
2. **Calibrar Embeddings:** Experimentar con diferentes proveedores
3. **Optimizar Chunks:** Ajustar tamaño de chunk (1000 caracteres)
4. **Filtros Avanzados:** Implementar búsquedas con metadatos más complejos
5. **Evaluación:** Crear métricas personalizadas de precisión
6. **Producción:** Implementar caché y optimizaciones

---

## 📞 Recursos Adicionales

- **ChromaDB Docs:** https://docs.trychroma.com/
- **LangChain Docs:** https://python.langchain.com/
- **RAG Fundamentals:** https://arxiv.org/abs/2312.10997

---

**Última actualización:** 17 de febrero de 2026  
**Versión:** 1.0  
**Estado:** ✅ Funcional
