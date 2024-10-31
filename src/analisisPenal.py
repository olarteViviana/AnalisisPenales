from langchain_mongodb import MongoDBAtlasVectorSearch
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_nomic import NomicEmbeddings
from langchain_nomic.embeddings import NomicEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
import pymongo

load_dotenv()
os.environ["USER_AGENT"] = "MyApp/1.0"
os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')
os.environ["LANGCHAIN_API_KEY"] = os.getenv('LANGCHAIN_API_KEY')

print("OPENAI_API_KEY:", os.getenv('OPENAI_API_KEY'))
print("LANGCHAIN_API_KEY:", os.getenv('LANGCHAIN_API_KEY'))

# Documen loading

from langchain_community.document_loaders import WebBaseLoader

urls = [
    "https://www.corteconstitucional.gov.co/relatoria/2024/T-435-24.htm",
    "https://www.corteconstitucional.gov.co/relatoria/2024/T-440-24.htm",
    "https://www.corteconstitucional.gov.co/relatoria/2024/T-378-24.htm",
    "https://www.corteconstitucional.gov.co/relatoria/2024/SU241-24.htm",
]

docs = [WebBaseLoader(url).load() for url in urls]
docs_list = [item for sublist in docs for item in sublist]

# Splitting 

from langchain_text_splitters import CharacterTextSplitter

text_splitter = CharacterTextSplitter.from_tiktoken_encoder(
    chunk_size=7500, chunk_overlap=100
)
doc_splits = text_splitter.split_documents(docs_list)

import tiktoken

encoding = tiktoken.get_encoding("cl100k_base")
encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")
for d in doc_splits:
    print("The document is %s tokens" % len(encoding.encode(d.page_content)))

# Conficuración de la conexión con MongoDB Atlas

mongo_uri = "mongodb+srv://olartevivianaa:S0q0d8rQ7jcutEVR@solucioneslegales.pkbwc.mongodb.net/?retryWrites=true&w=majority&appName=SolucionesLegales"
client = pymongo.MongoClient(mongo_uri)
database = client.solucionesLegales
collection_name = "sentencias"
collections = database[collection_name]

try:
    client.admin.command('ping')
    print('Conexión exitosa')
except pymongo.errors.ConnectionFailure as e:
    print('No se pudo conectar a MongoDB: %s' % e)

# Insertar documentos en la base de datos
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

docs_to_insert = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in doc_splits]
collections.insert_many(docs_to_insert)

vectorstore = MongoDBAtlasVectorSearch(
    embedding=embeddings,
    collection=collections
)
retriever = vectorstore.as_retriever()

# Prompt
template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# LLM API
model = ChatOpenAI(temperature=0, model="gpt-4-1106-preview")

# Chain
chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt
    | model
    | StrOutputParser()
)

# Question
chain.invoke("What are the types of agent memory?")