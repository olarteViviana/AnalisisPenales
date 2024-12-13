# Análisis de Sentencias Penales

Este proyecto es una herramienta de análisis de sentencias penales que utiliza técnicas de procesamiento de lenguaje natural y búsqueda vectorial para extraer, procesar y analizar sentencias de la Corte Constitucional de Colombia.

## Tecnologías Utilizadas

### Principales Frameworks y Bibliotecas
- **LangChain**: Framework para desarrollo de aplicaciones con modelos de lenguaje
- **MongoDB Atlas**: Base de datos en la nube con capacidades de búsqueda vectorial
- **OpenAI**: Modelos de lenguaje y embeddings
- **Groq**: Modelo de lenguaje alternativo
- **Ollama**: Integración con modelos de lenguaje locales

### Bibliotecas de Procesamiento
- **BeautifulSoup4**: Para web scraping y parsing de HTML
- **Pandas**: Para manipulación y análisis de datos
- **Requests**: Para realizar peticiones HTTP

## Estructura del Proyecto

```
AnalisisPenales/
├── src/
│   ├── analisisPenal.py
│   ├── main.py
│   └── .env
└── prompts/
    └── template_T.txt
```

## Funcionalidades Principales

### 1. Web Scraping de Sentencias
- Búsqueda automática en el sitio web de la Corte Constitucional
- Extracción de enlaces y contenido de sentencias
- Procesamiento y limpieza de texto extraído

### 2. Procesamiento de Datos
- Conversión de texto a formato estructurado
- Segmentación de documentos largos en chunks manejables
- Almacenamiento en MongoDB Atlas

### 3. Análisis Vectorial
- Generación de embeddings usando OpenAI
- Búsqueda vectorial en MongoDB Atlas
- Procesamiento de consultas mediante modelos de lenguaje

## Configuración y Uso

### Requisitos Previos
1. Python 3.8 o superior
2. MongoDB Atlas cuenta y cluster configurado
3. Claves API necesarias:
   - OpenAI API Key
   - LangChain API Key
   - Groq API Key

### Variables de Entorno
Crear un archivo `.env` con las siguientes variables:
```
OPENAI_API_KEY=your_openai_key
LANGCHAIN_API_KEY=your_langchain_key
GROQ_API_KEY=your_groq_key
MONGODB_URI=your_mongodb_connection_string
MONGODB_DB=your_database_name
MONGODB_COLLECTION=your_collection_name
```

### Flujo de Trabajo

1. **Búsqueda de Sentencias**
   ```python
   scraping_sentencias("término de búsqueda")
   ```
   - Realiza web scraping de sentencias
   - Procesa y almacena los resultados en MongoDB

2. **Configuración del Modelo**
   ```python
   configurar_modelo()
   ```
   - Configura la conexión con MongoDB Atlas
   - Inicializa los embeddings y la búsqueda vectorial

3. **Inicialización de la Cadena**
   ```python
   initialize_chain()
   ```
   - Prepara el pipeline de procesamiento
   - Configura los modelos de lenguaje

## Características Técnicas Detalladas

### Procesamiento de Documentos
- División de textos largos en chunks de 1000 caracteres
- Preservación de metadatos durante el procesamiento
- Manejo de múltiples formatos de documentos HTML

### Búsqueda Vectorial
- Utiliza embeddings de OpenAI (modelo text-embedding-ada-002)
- Implementación de búsqueda semántica en MongoDB Atlas
- Capacidad de procesamiento de consultas en lenguaje natural

### Manejo de Errores
- Validación de documentos en la base de datos
- Manejo de excepciones en solicitudes HTTP
- Verificación de configuración de variables de entorno

## Contribución y Desarrollo

Para contribuir al proyecto:
1. Fork del repositorio
2. Crear una rama para nuevas características
3. Realizar cambios y documentar
4. Enviar pull request

## Notas Importantes
- Asegurarse de tener todas las claves API necesarias
- Mantener actualizadas las dependencias
- Revisar la documentación de MongoDB Atlas para configuración óptima
