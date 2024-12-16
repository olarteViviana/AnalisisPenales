# Pandas para manipulación y análisis de datos
import pandas as pd
from langchain.schema import Document
import nest_asyncio
import os
from dotenv import load_dotenv
import json
from bs4 import BeautifulSoup
import requests
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Cargar variables de entorno
load_dotenv()

# Configurar variables de entorno
os.environ["USER_AGENT"] = "MyApp/1.0"
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY', '')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY', '')
os.environ["GROQ_API_KEY"] = os.getenv('GROQ_API_KEY', '')

# Inicializar ChromaDB
embeddings = OpenAIEmbeddings()
CHROMA_DB_DIR = "./chroma_db"
vectorstore = None

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
    if not text:
        return []
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap
    )
    
    return text_splitter.split_text(text)

def scraping_sentencias(termino_de_busqueda):
    """
    Realiza el scraping de sentencias y las almacena en ChromaDB
    
    Args:
        termino_de_busqueda (str): Término para buscar sentencias
    """
    global vectorstore
    
    # Configurar ChromaDB si no está inicializado
    if vectorstore is None:
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name="sentencias")
    
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
            print("Preparando datos para ChromaDB...")
            # Convertir el diccionario en un df de pandas
            data = list(diccionario_relatorias.items())
            df = pd.DataFrame(data)
            df.columns = ['Enlace', 'Texto']
            df['Sentencia'] = df['Enlace'].str.split('/relatoria/').str[-1].str.split('.htm').str[0]
            df = df[['Sentencia', 'Texto']]

            # Procesar y cargar documentos
            print("Procesando documentos para ChromaDB...")
            docs_to_insert = []
            for _, row in df.iterrows():
                chunks = custom_split(row['Texto'])
                for chunk in chunks:
                    docs_to_insert.append(Document(
                        page_content=chunk,
                        metadata={'sentencia': row['Sentencia']}
                    ))

            # Insertar en ChromaDB
            print(f"Insertando {len(docs_to_insert)} documentos en ChromaDB...")
            vectorstore.add_documents(docs_to_insert)
            print("Documentos insertados exitosamente en ChromaDB")
            
            return True

        except Exception as e:
            print(f"Error al procesar los documentos: {e}")
            return False

    except Exception as e:
        print(f"Error durante el scraping: {str(e)}")
        return False

def cargar_template():
    """Carga todos los templates desde archivos externos"""
    template_path = os.path.join(os.path.dirname(__file__), '..', 'prompts', 'template_T.txt')
    try:
        with open(template_path, 'r', encoding='utf-8') as file:
            return file.read()
    except FileNotFoundError:
        raise ValueError("No se encontró el archivo template_T.txt en la carpeta prompts")

def configurar_modelo():
    """
    Configura el modelo y la base de datos ChromaDB
    """
    global vectorstore
    
    try:
        nest_asyncio.apply()
        
        # Inicializar ChromaDB si no existe
        if vectorstore is None:
            vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name="sentencias")
        
        # Verificar si hay documentos en la base de datos
        if len(vectorstore.get()['ids']) == 0:
            print("Base de datos vacía. Iniciando scraping...")
            return None
        
        # Configurar el retriever para mantener la riqueza del contexto
        retriever = vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": 5,  # Mantener más documentos relevantes para un análisis completo
                "fetch_k": 10,  # Búsqueda inicial amplia
                "score_threshold": 0.7  # Balance entre precisión y cobertura
            }
        )

        template = cargar_template()
        if not template:
            raise ValueError("No se pudo cargar la plantilla del prompt")
            
        prompt = ChatPromptTemplate.from_template(template)

        # Usar GPT-4 Turbo con ventana de contexto grande
        model = ChatOpenAI(
            temperature=0,
            model="gpt-4-1106-preview",
            max_tokens=4096
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

def clear_collection():
    """
    Limpia la colección de ChromaDB
    """
    global vectorstore
    if vectorstore is not None:
        vectorstore.delete_collection()
        vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=embeddings, collection_name="sentencias")

if __name__ == "__main__":
    configurar_modelo()
