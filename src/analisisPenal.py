# Pandas para manipulación y análisis de datos
import pandas as pd
# MongoDB Atlas Vector Search para búsquedas vectoriales en MongoDB
from langchain_mongodb import MongoDBAtlasVectorSearch
# Cargador de documentos de MongoDB para Langchain
from langchain_community.document_loaders.mongodb import MongodbLoader
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
# Integración con Groq para chat
from langchain_groq import ChatGroq

load_dotenv()
# Establece el User-Agent para identificar nuestra aplicación en las solicitudes HTTP
# Esto ayuda a los servidores web a identificar qué software está haciendo las peticiones
os.environ["USER_AGENT"] = "MyApp/1.0"
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', '')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY', '')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY', '')
client = pymongo.MongoClient(os.environ['MONGODB_URI'])
collections = client.get_database(os.environ['MONGODB_DB']).get_collection(os.environ['MONGODB_COLLECTION'])
chain = None

# Función para realizar scraping y guardar resultados
def scraping_sentencias(termino_de_busqueda):
    # Construir la URL para la búsqueda
    termino_de_busqueda = termino_de_busqueda.replace(' ', '+')
    URL = 'https://www.corteconstitucional.gov.co/relatoria/buscador_new/?searchOption=texto&fini=1992-01-01&ffin=2024-10-29&buscar_por='+ termino_de_busqueda +'&accion=search&verform=si&slop=1&volver_a=relatoria&qu=625&maxprov=100&OrderbyOption=des__score'

    # Inicializar soup como None
    soup = None

    # Realizar la solicitud GET a la página
    response = requests.get(URL)
    
    # Verificar si la solicitud fue exitosa
    if response.status_code == 200:
        # Parsear el contenido HTML con BeautifulSoup
        soup = BeautifulSoup(response.content, 'html.parser')

        # Encontrar todas las etiquetas 'a' con atributo 'href'
        enlaces = [a['href'] for a in soup.find_all('a', href=True)]
    else:
        print(f"Error al acceder a la página: {response.status_code}")
        return

    # crear una lista con los enlaces obtenidos anteriormente

    lista_enlaces = []

    # Encontrar todas las etiquetas 'a' con atributo 'href'
    for a in soup.find_all('a', href=True):
        lista_enlaces.append(a['href'])


    # filtrar lista_enlaces y coger solo los enlaces contengan la palabra relatoria

    enlaces_relatoria = [enlace for enlace in lista_enlaces if 'relatoria' in enlace]
    enlaces_relatoria = [enlace for enlace in enlaces_relatoria if len(enlace) > 49]

    # Imprimir la lista de enlaces filtrados
    enlaces_relatoria

    # diccionario_relatorias
    diccionario_relatorias = {}

    for enlace in enlaces_relatoria:
        try:
            nota = requests.get(enlace)
            nota.raise_for_status()  # Verifica si hubo algún problema con la respuesta HTTP
            s_nota = BeautifulSoup(nota.text, 'html.parser')
            # First try WordSection1
            word_section = s_nota.find('div', attrs={'class': 'WordSection1'})
            if word_section is not None:
                texto = word_section.text.strip()
                diccionario_relatorias[enlace] = texto
            else:
                # If WordSection1 not found, try Section1
                section1 = s_nota.find('div', attrs={'class': 'Section1'})
                if section1 is not None:
                    texto = section1.text.strip()
                    diccionario_relatorias[enlace] = texto
                else:
                    print(f"No se encontró el contenido esperado en el enlace {enlace}")
        except requests.exceptions.RequestException as e:
            print(f"Error al solicitar el enlace {enlace}: {e}")

    diccionario_relatorias

    # convertir el diccionario en un df de pandas
    items = list(diccionario_relatorias.items())
    df = pd.DataFrame(items)
    df.columns = ['Enlace', 'Texto']

    df['Sentencia'] = df['Enlace'].str.split('/relatoria/').str[-1].str.split('.htm').str[0]

    # Reorganizar el DataFrame
    df = df[['Sentencia', 'Texto']]  # Selecciona las columnas en el orden deseado

    # exportar el df en formato JSON Lines

    nombre_json = ('sentencias_' + termino_de_busqueda).replace('+', '_') + '.jsonl'
    # Assuming df is your DataFrame
    df.to_json(nombre_json, orient='records', lines=True)

    # Leer el archivo JSON Lines y cargar los documentos
    with open(nombre_json, 'r') as f:
        docs_to_insert = [json.loads(line) for line in f]

    # Splitting 

    from langchain.text_splitter import CharacterTextSplitter
    from langchain.docstore.document import Document

    docs_to_insert = [
        Document(page_content=doc['Texto'], metadata={'sentencia': doc['Sentencia']})
        for doc in docs_to_insert
    ]

    def custom_split(text, max_size=1000):
        chunks = []
        while len(text) > max_size:
            chunk = text[:max_size]
            chunks.append(chunk)
            text = text[max_size:]
        if text:
            chunks.append(text)
        return chunks

    docs_splits = []
    for doc in docs_to_insert:
        chunks = custom_split(doc.page_content)
        for chunk in chunks:
            docs_splits.append(Document(page_content=chunk, metadata=doc.metadata))
    docs_splits_dict = [doc.dict() for doc in docs_splits]

    collections.insert_many(docs_splits_dict)

def cargar_template():
    """Carga todos los templates desde archivos externos"""
    template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'template_T.txt')
    try:
        with open(template_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise ValueError("No se encontró el archivo template_T.txt en la carpeta prompts")

# Configuración de la conexión con MongoDB Atlas
def configurar_modelo():

    nest_asyncio.apply()
    load_dotenv()

    if collections.count_documents({}) == 0:
        raise ValueError("No hay documentos en la base de datos. Por favor, realice una búsqueda primero.")

    loader = MongodbLoader(
        connection_string=os.environ['MONGODB_URI'],
        db_name=os.environ['MONGODB_DB'],
        collection_name=os.environ['MONGODB_COLLECTION'],
        filter_criteria={},
        field_names=["metadata", "page_content"],
    )
    docs = loader.load()

    if not docs:
        raise ValueError("No se encontraron documentos en la base de datos.")

    # Insertar documentos en la base de datos
    
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

    vectorStore = MongoDBAtlasVectorSearch.from_documents( 
        documents= docs,\
        embedding= embeddings, 
        collection= collections,
        index_name=os.environ['MONGODB_VECTOR_INDEX']
    )

    retriever = vectorStore.as_retriever(search_kwargs={"similarity_threshold": 0.1})


    template = cargar_template()
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(temperature=0, model="gpt-4o")
    #model = ChatGroq(temperature=0, model="llama-3.1-70b-versatile")
    # Local LLM
    ollama_llm = "phi3.5"
    model_local = ChatOllama(model=ollama_llm)

    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain
    
def initialize_chain():
    global chain
    try:
        chain = configurar_modelo()
        return chain
    except Exception as e:
        raise Exception(f"Error al iniciar el chain: {e}")


def clear_collection():
    collections.delete_many({})

if __name__ == "__main__":
    cargar_template()
    initialize_chain()