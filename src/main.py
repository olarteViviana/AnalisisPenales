# Archivo streamlit_app.py

import streamlit as st
import os
from pymongo import MongoClient
from dotenv import load_dotenv
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_community.document_loaders.mongodb import MongodbLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Cargar variables de entorno
load_dotenv()

# Configuración de la página
st.set_page_config(page_title="Consulta de Sentencias", layout="centered")
st.title("Consulta de Sentencias de la Corte Constitucional")

# Input del término de búsqueda y botones para iniciar scraping y consulta
with st.sidebar:
    termino_de_busqueda = st.text_input("Ingrese el término de búsqueda para scraping")
    if st.button("Buscar y almacenar sentencias"):
        # Nota: Aquí no estamos ejecutando el scraping en Streamlit para evitar duplicar el código
        st.warning("Ejecute el código de scraping antes de iniciar Streamlit")

# Conexión a MongoDB para cargar datos
@st.cache_resource
def load_data():
    client = MongoClient(os.environ['MONGODB_URI'])
    loader = MongodbLoader(
        connection_string=os.environ['MONGODB_URI'],
        db_name=os.environ['MONGODB_DB'],
        collection_name=os.environ['MONGODB_COLLECTION']
    )
    return loader.load()

# Función para realizar la consulta
def consulta_sentencias(query):
    # Configuración de embeddings y carga de documentos
    embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
    docs = load_data()
    client = MongoClient(os.environ['MONGODB_URI'])
    collection = client.get_database(os.environ['MONGODB_DB']).get_collection(os.environ['MONGODB_COLLECTION'])

    # Crear el índice vectorial para búsqueda
    vectorStore = MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        collection=collection,
        index_name=os.environ['MONGODB_VECTOR_INDEX']
    )
    
    retriever = vectorStore.as_retriever(search_kwargs={"similarity_threshold": 0.1})
    
    # Plantilla de prompt
    prompt = ChatPromptTemplate.from_template("""
    En la información proporcionada, actúe como magistrado de la Corte Constitucional y responda a la pregunta.
    {context}
    Question: {question}
    """)
    
    model = ChatOpenAI(temperature=0, model="gpt-4o")
    chain = {"context": retriever, "question": RunnablePassthrough()} | prompt | model | StrOutputParser()
    
    # Ejecutar la consulta
    result = chain.invoke(query)
    return result

# Input para consulta de sentencias
query = st.text_input("Ingrese su pregunta sobre una sentencia")

# Botón para realizar la consulta y mostrar resultado
if st.button("Consultar sentencias"):
    if query:
        with st.spinner("Consultando..."):
            resultado = consulta_sentencias(query)
        st.write("### Resultado de la consulta")
        st.write(resultado)
    else:
        st.warning("Ingrese una pregunta para realizar la consulta")