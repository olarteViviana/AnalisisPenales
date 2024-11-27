import pandas as pd
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders.mongodb import MongodbLoader
import nest_asyncio
from pymongo.mongo_client import MongoClient
import os
from dotenv import load_dotenv
from langchain_nomic import NomicEmbeddings
import pymongo
import json
from bs4 import BeautifulSoup
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

load_dotenv()
os.environ["USER_AGENT"] = "MyApp/1.0"
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')
client = pymongo.MongoClient(os.environ['MONGODB_URI'])
collections = client.get_database(os.environ['MONGODB_DB']).get_collection(os.environ['MONGODB_COLLECTION'])

# Función para realizar scraping y guardar resultados
def scraping_sentencias(termino_de_busqueda):
    # Construir la URL para la búsqueda
    termino_de_busqueda = termino_de_busqueda.replace(' ', '+')
    URL = 'https://www.corteconstitucional.gov.co/relatoria/buscador_new/?searchOption=texto&fini=1992-01-01&ffin=2024-10-29&buscar_por='+ termino_de_busqueda +'&accion=search&verform=si&slop=1&volver_a=relatoria&qu=625&maxprov=100&OrderbyOption=des__score'

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
            texto = (s_nota.find('div', attrs={'class': 'WordSection1'}).text).strip()
            diccionario_relatorias[enlace] = texto
        except requests.exceptions.RequestException as e:
            print(f"Error al solicitar el enlace {enlace}: {e}")
        except AttributeError:
            try:
                # Si no encuentra 'WordSection1', intenta con 'Section1'
                texto = (s_nota.find('div', attrs={'class': 'Section1'}).text).strip()
                diccionario_relatorias[enlace] = texto
            except AttributeError as e:
                print(f"Error procesando el contenido del enlace {enlace}: {e}")

    diccionario_relatorias

    # convertir el diccionario en un df de pandas

    df = pd.DataFrame(list(diccionario_relatorias.items()), columns=['Enlace', 'Texto'])

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

# Conficuración de la conexión con MongoDB Atlas
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
        documents= docs,
        embedding= embeddings, 
        collection= collections,
        index_name=os.environ['MONGODB_VECTOR_INDEX']
    )

    retriever = vectorStore.as_retriever(search_kwargs={"similarity_threshold": 0.1})


    template = """Quiero un análisis jurídico profesional sobre los derechos fundamentales amenazados o dañados que se debaten en la Corte Constitucional de Colombia. Es importante conocer los hechos de acuerdo a las circunstancias de modo, el tiempo con las fechas y hora, el lugar
    donde ocurrieron y las personas naturales o jurídicas que tienen conflicto entre ellas. También es importante conocer cuál fue el daño o peligro que afecta los derechos fundamentales dentro de las consideraciones tenidas en cuenta por la Corte Constitucional. Finalmente, requiero saber lo que resuelve la Corte Constitucional. Con base en las anteriores instrucciones, proporciona un resumen de:
    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = ChatOpenAI(temperature=0, model="gpt-4o")

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
chain = None

def initialize_chain():
    try:
        chain = configurar_modelo()
        return chain
    except Exception as e:
        raise Exception(f"Error al iniciar el chain: {e}")


def clear_collection():
    collections.delete_many({})