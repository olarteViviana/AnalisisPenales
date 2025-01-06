# Pandas para manipulación y análisis de datos
import pandas as pd
# MongoDB Atlas Vector Search para búsquedas vectoriales en MongoDB
from langchain_mongodb import MongoDBAtlasVectorSearch
# Cargador de documentos de MongoDB para Langchain
from langchain_community.document_loaders.mongodb import MongodbLoader
# Document class for creating document objects
from langchain.schema import Document
# Permite la ejecución de código asíncrono en notebooks
import nest_asyncio
# Cliente principal de MongoDB para conexión a la base de datos
from pymongo.mongo_client import MongoClient
# Manejo de variables de entorno y rutas del sistema
import os
# Carga de variables de entorno desde archivo .env
from dotenv import load_dotenv
# Embeddings de Nomic para procesamiento de texto
from langchain_nomic import NomicEmbeddings
# Driver de MongoDB para Python
import pymongo
# Manejo de datos en formato JSON
import json
# BeautifulSoup para parsing de HTML
from bs4 import BeautifulSoup
# Requests para realizar peticiones HTTP
import requests
# Embeddings de OpenAI para procesamiento de texto
from langchain_openai import OpenAIEmbeddings
# Plantillas de chat para Langchain
from langchain_core.prompts import ChatPromptTemplate
# Modelo de chat de OpenAI
from langchain_openai import ChatOpenAI
# Parser de salida para cadenas de texto
from langchain_core.output_parsers import StrOutputParser
# Componentes ejecutables de Langchain
from langchain_core.runnables import RunnablePassthrough
# Integración con Ollama para modelos de lenguaje
from langchain_ollama import ChatOllama
# Modelo de Groq para LLM
from langchain_groq import ChatGroq
# Integración con Groq para chat
from langchain_groq import ChatGroq
from litellm import completion
from litellm import get_supported_openai_params
from litellm import supports_response_schema

load_dotenv()
# Establece el User-Agent para identificar nuestra aplicación en las solicitudes HTTP
# Esto ayuda a los servidores web a identificar qué software está haciendo las peticiones
os.environ["USER_AGENT"] = "MyApp/1.0"
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', '')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY', '')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY', '')
client = pymongo.MongoClient(os.environ['MONGODB_URI'])
try:
    # Send a ping to confirm a successful connection
    client.admin.command('ping')
    print("Successfully connected to MongoDB.")
except Exception as e:
    print(f"Error connecting to MongoDB: {e}")
    raise
collections = client.get_database(os.environ['MONGODB_DB']).get_collection(os.environ['MONGODB_COLLECTION'])
chain = None

# Función para dividir texto en chunks más pequeños
def custom_split(text, chunk_size=500, overlap=50):
    """
    Divide un texto en chunks más pequeños con un tamaño y solapamiento específicos.
    
    Args:
        text (str): Texto a dividir
        chunk_size (int): Tamaño máximo de cada chunk en caracteres
        overlap (int): Número de caracteres que se solapan entre chunks
        
    Returns:
        list: Lista de chunks de texto
    """
    try:
        # Validar entrada
        if not isinstance(text, str):
            raise ValueError("El texto debe ser una cadena de caracteres")
        if not text.strip():
            return []
            
        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            # Tomar el chunk actual
            end = min(start + chunk_size, text_length)
            
            # Si no es el último chunk, buscar el último espacio en blanco
            if end < text_length:
                # Buscar el último espacio en blanco en el chunk
                last_space = text.rfind(' ', start, end)
                if last_space != -1:
                    end = last_space
            
            # Añadir el chunk a la lista
            chunk = text[start:end].strip()
            if chunk:  # Solo añadir si el chunk no está vacío
                chunks.append(chunk)

            # Mover el inicio al siguiente chunk, considerando el solapamiento
            start = end - overlap if end < text_length else text_length

        return chunks
    except Exception as e:
        print(f"Error al dividir el texto en chunks: {str(e)}")
        raise ValueError(f"Error en custom_split: {str(e)}")

# Función para realizar scraping y guardar resultados
def scraping_sentencias(termino_de_busqueda):
    try:
        # Construir la URL para la búsqueda
        termino_de_busqueda = termino_de_busqueda.replace(' ', '+')
        URL = 'https://www.corteconstitucional.gov.co/relatoria/buscador_new/?searchOption=texto&fini=1992-01-01&ffin=2024-10-29&buscar_por='+ termino_de_busqueda +'&accion=search&verform=si&slop=1&volver_a=relatoria&qu=625&maxprov=100&OrderbyOption=des__score'

        # Realizar la solicitud GET a la página
        print(f"Buscando en URL: {URL}")
        response = requests.get(URL, timeout=30)  # Added timeout
        response.raise_for_status()  # Verificar si la solicitud fue exitosa
        
        # Parsear el contenido HTML con BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Encontrar todas las etiquetas 'a' con atributo 'href'
        enlaces = []
        for a in soup.find_all('a', href=True):
            href = a['href']
            if 'relatoria' in href and len(href) > 49:
                if not href.startswith('http'):
                    href = 'https://www.corteconstitucional.gov.co' + href
                enlaces.append(href)

        if not enlaces:
            print(f"No se encontraron enlaces para el término de búsqueda: {termino_de_busqueda}")
            # Intentar buscar directamente la sentencia si el formato es correcto
            if termino_de_busqueda.startswith('T-') and termino_de_busqueda.endswith('-24'):
                sentencia_url = f'https://www.corteconstitucional.gov.co/relatoria/2024/{termino_de_busqueda}.htm'
                print(f"Intentando acceder directamente a: {sentencia_url}")
                try:
                    nota = requests.get(sentencia_url, timeout=30)
                    nota.raise_for_status()
                    enlaces = [sentencia_url]
                except:
                    raise ValueError(f"No se encontró la sentencia {termino_de_busqueda}")
            else:
                raise ValueError("No se encontraron enlaces de sentencias")

        # diccionario_relatorias
        diccionario_relatorias = {}

        for enlace in enlaces:
            try:
                print(f"Procesando enlace: {enlace}")
                nota = requests.get(enlace, timeout=30)
                nota.raise_for_status()
                s_nota = BeautifulSoup(nota.text, 'html.parser')
                
                # Buscar el contenido en diferentes clases
                content = None
                for class_name in ['WordSection1', 'Section1']:
                    section = s_nota.find('div', attrs={'class': class_name})
                    if section:
                        content = section.text.strip()
                        break
                
                if content:
                    diccionario_relatorias[enlace] = content
                    print(f"Contenido extraído exitosamente de {enlace}")
                else:
                    print(f"No se encontró el contenido esperado en el enlace {enlace}")
                
            except requests.exceptions.RequestException as e:
                print(f"Error al solicitar el enlace {enlace}: {e}")
                continue

        if not diccionario_relatorias:
            raise ValueError("No se pudo extraer contenido de ninguna sentencia")

        try:
            print("Preparando datos para MongoDB...")
            # Convertir el diccionario en un df de pandas
            data = list(diccionario_relatorias.items())
            df = pd.DataFrame(data)
            df.columns = ['Enlace', 'Texto']
            df['Sentencia'] = df['Enlace'].str.split('/relatoria/').str[-1].str.split('.htm').str[0]
            df = df[['Sentencia', 'Texto']]

            # Exportar el df en formato JSON Lines
            nombre_json = ('sentencias_' + termino_de_busqueda).replace('+', '_') + '.jsonl'
            print(f"Guardando datos en {nombre_json}...")
            df.to_json(nombre_json, orient='records', lines=True)

            # Procesar y cargar documentos
            print("Procesando documentos para MongoDB...")
            docs_to_insert = []
            with open(nombre_json, 'r', encoding='utf-8') as f:
                for line in f:
                    doc = json.loads(line)
                    docs_to_insert.append(Document(
                        page_content=doc['Texto'],
                        metadata={'sentencia': doc['Sentencia']}
                    ))

            # Dividir documentos en chunks más pequeños
            print("Dividiendo documentos en chunks...")
            docs_splits = []
            for doc in docs_to_insert:
                chunks = custom_split(doc.page_content)
                for chunk in chunks:
                    docs_splits.append(Document(page_content=chunk, metadata=doc.metadata))

            # Insertar en la base de datos
            print(f"Insertando {len(docs_splits)} chunks en MongoDB...")
            documents_to_insert = [{
                'page_content': doc.page_content,
                'metadata': doc.metadata
            } for doc in docs_splits]
            collections.insert_many(documents_to_insert)
            print("Documentos insertados exitosamente en la base de datos")
            
            return diccionario_relatorias

        except Exception as e:
            print(f"Error en el procesamiento de datos: {str(e)}")
            raise ValueError(f"Error al procesar los datos: {str(e)}")

    except Exception as e:
        print(f"Error en el proceso de scraping: {str(e)}")
        raise

# Función para cargar plantilla de chat
def cargar_template():
    """Carga todos los templates desde archivos externos"""
    template = """Eres un asistente legal especializado en análisis jurídico exhaustivo. Tu tarea es proporcionar un análisis detallado y estructurado de sentencias judiciales, siguiendo este formato específico:

1. METADATOS DEL CASO
- Número de Expediente
- Fecha de la Decisión
- Accionante y Accionado
- Magistrado Ponente
- Tema Central del Caso
- Derechos Fundamentales Involucrados
- Tipo de Acción Legal

2. CRONOLOGÍA DETALLADA
- Línea de tiempo completa de los hechos relevantes
- Fechas específicas de cada evento significativo
- Desarrollo procesal del caso
- Decisiones previas y sus fundamentos
- Antecedentes procesales relevantes

3. MARCO NORMATIVO APLICABLE
- Artículos constitucionales relevantes
- Leyes y decretos aplicables
- Tratados internacionales pertinentes
- Códigos y estatutos relacionados
- Jurisprudencia relacionada

4. ARGUMENTOS PRINCIPALES
- Argumentos de la parte accionante
- Argumentos de la parte accionada
- Consideraciones de instancias previas
- Intervenciones relevantes
- Problemas jurídicos planteados

5. ANÁLISIS DE PRUEBAS
- Pruebas documentales presentadas
- Testimonios y declaraciones
- Peritajes y conceptos técnicos
- Valoración probatoria realizada
- Elementos probatoris determinantes

6. DEFECTOS IDENTIFICADOS
- Defectos procedimentales
- Defectos sustantivos
- Defectos fácticos
- Violaciones constitucionales
- Irregularidades procesales

7. CONCLUSIONES Y DECISIÓN
- Ratio decidendi
- Órdenes específicas
- Salvamentos de voto
- Efectos de la decisión
- Impacto jurisprudencial

8. SENTENCIAS RELEVANTES
- Sentencias tipo T citadas
- Sentencias tipo C citadas
- Sentencias tipo SU citadas
- Sentencias tipo A citadas
- Línea jurisprudencial relacionada

Utiliza la siguiente información para generar tu respuesta: {context}

Pregunta: {question}

Instrucciones específicas:
1. Proporciona MÍNIMO tres párrafos detallados para cada sección
2. Incluye todas las fechas, números de expediente y referencias específicas
3. Cita textualmente las partes más relevantes de la sentencia
4. Explica el razonamiento detrás de cada argumento y decisión
5. Relaciona cada punto con la normativa y jurisprudencia aplicable
6. NO omitas detalles relevantes ni hagas resúmenes breves
7. Mantén un lenguaje técnico-jurídico apropiado para jueces
8. Incluye análisis crítico y conexiones con otros casos similares
9. Identifica claramente la línea jurisprudencial y su evolución
10. Destaca el impacto de la decisión en el ordenamiento jurídico
11. No incluyas información irrelevante o redundante
12. Asegúrate de listar todas las sentencias referenciadas"""
    
    return template

# Configuración de la conexión con MongoDB Atlas
def configurar_modelo():
    try:
        nest_asyncio.apply()
        load_dotenv()

        if collections.count_documents({}) == 0:
            print("Base de datos vacía. Iniciando scraping...")
            return None

        loader = MongodbLoader(
            connection_string=os.environ['MONGODB_URI'],
            db_name=os.environ['MONGODB_DB'],
            collection_name=os.environ['MONGODB_COLLECTION'],
            filter_criteria={},
            field_names=["metadata", "page_content"],
        )
        docs = loader.load()

        if not docs:
            print("No se encontraron documentos en la base de datos.")
            return None

        embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
        
        vectorStore = MongoDBAtlasVectorSearch.from_documents(
            documents=docs,
            embedding=embeddings,
            collection=collections,
            index_name=os.environ['MONGODB_VECTOR_INDEX']
        )

        # Configurar el retriever para mantener la riqueza del contexto
        retriever = vectorStore.as_retriever(
            search_kwargs={
                "k": 500,  # Mantener más documentos relevantes para un análisis completo
                "similarity_threshold": 0.2,  # Balance entre precisión y cobertura
                "fetch_k": 700  # Búsqueda inicial amplia
            }
        )

        template = cargar_template()
        if not template:
            raise ValueError("No se pudo cargar la plantilla del prompt")
            
        prompt = ChatPromptTemplate.from_template(template)

        # Usar GPT-4 Turbo con ventana de contexto grande
        model = ChatOpenAI(
            temperature=1,
            model="gpt-4-1106-preview",
            max_tokens=4096,
            presence_penalty=0.7,  # Añadido para fomentar elaboración
            frequency_penalty=0.7  # Añadido para evitar repeticiones
        )

        chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        return chain
        
    except Exception as e:
        print(f"Error al configurar el modelo: {str(e)}")
        raise

def initialize_chain():
    global chain
    try:
        chain = configurar_modelo()
        if chain is None:
            # Si no hay chain porque la base de datos está vacía, retornamos None sin error
            return None
        return chain
    except Exception as e:
        raise Exception(f"Error al iniciar el chain: {e}")

def clear_collection():
    collections.delete_many({})

if __name__ == "__main__":
    cargar_template()
    initialize_chain()