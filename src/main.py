import streamlit as st
from analisisPenal import scraping_sentencias, initialize_chain, clear_collection
import logging
import os

st.title("Análisis de Sentencias - Corte Constitucional")

if 'chain' not in st.session_state:
    st.session_state.chain = None
    clear_collection()

st.sidebar.header("Buscar Sentencias")
termino_de_busqueda = st.sidebar.text_input("Ingrese el término de búsqueda", "")

if st.sidebar.button("Buscar Sentencias"):
    if termino_de_busqueda.strip():
        st.sidebar.success(f"Buscando sentencias con el término: {termino_de_busqueda}...")
        st.sidebar.warning("El análisis puede tomar unos minutos")
        with st.spinner("Realizando la búsqueda..."):
            try:
                diccionario_relatorias = scraping_sentencias(termino_de_busqueda)
                st.session_state.chain = initialize_chain()
                st.sidebar.success("Búsqueda completada")
            except Exception as e:
                st.sidebar.error(f"Error al buscar sentencias: {e}")
    else:
        st.sidebar.error("Por favor, ingrese un término de búsqueda válido")

if 'termino' in st.session_state:
    termino_de_busqueda = st.session_state.termino
    st.sidebar.info("Resultados encontrados:")

    if diccionario_relatorias:
        for enlace, texto in diccionario_relatorias.items():
            st.success(f"Enlace: {enlace}, Total enlaces: {len(diccionario_relatorias)}")
    else:
        st.error("No se encontraron resultados para este término")

st.subheader("Chat para Ánalisis de Sentencias")
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

user_input = st.text_input("Haz tu pregunta sobre las sentencias disponibles: ")

if st.button("Enviar"):
    if user_input.strip():
        st.session_state.chat_history.append({"role": "user", "content": user_input})

        with st.spinner("Analizando la respuesta..."):
            try:
                if st.session_state.chain is None:
                    st.error("Por favor, primero realice una búsqueda de sentencias.")
                else:
                    result = st.session_state.chain.invoke(user_input)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result}
                    )
            except Exception as e:
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": f"Error al procesar la pregunta: {e}"}
                )

    else:
        st.error("Por favor, ingrese una pregunta válida")

# Mostrar el historial del chat
for message in st.session_state.chat_history:
    if message["role"] == "user":
        st.markdown(f"**Usuario:** {message['content']}")
    else:
        st.markdown(f"**Asistente:** {message['content']}")

st.sidebar.info(
    """
    Esta herramienta utiliza modelos de lenguaje para analizar y extraer información
    clave de sentencias de la Corte Constitucional. Puedes buscar sentencias específicas
    y hacer preguntas relacionadas.
    """
)