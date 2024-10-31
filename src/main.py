import streamlit as st
from analisisPenal import chain

st.title("Análisis de Sentencias Penales")
st.write("Este sistema permite realizar consultas basadas en el análisis de sentencias de la Corte Constitucional de Colombia.")

query = st.text_input("Escribe tu pregunta sobre las sentencias:")

if st.button("Consultar"):
    if query:
        with st.spinner("Consultando..."):
            response = chain.invoke(query)
            st.success("Consulta completada.")
            st.write("**Respuesta:**")
            st.write(response)
    else:
        st.warning("Por favor, escribe una pregunta para consultar.")

# Ejecución en Streamlit
if __name__ == "__main__":
    st.write("¡Bienvenido! Ingresa una consulta para comenzar.")
