import streamlit as st
from analisisPenal import scraping_sentencias, initialize_chain, clear_collection
import logging
import os
from record_audio import record_audio
from record_audio import AudioRecorder
from transcribe_audio import transcribe_audio
from voice_mode import detect_topic
from datetime import datetime
import tempfile
import time

def initialize_session_state():
    """Inicializa el estado de la sesión"""
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'recording_status' not in st.session_state:
        st.session_state.recording_status = None
    if 'output_file' not in st.session_state:
        st.session_state.output_file = None
    if 'text_input' not in st.session_state:
        st.session_state.text_input = ""
    if 'recorder' not in st.session_state:
        st.session_state.recorder = AudioRecorder()

def handle_recording():
    """Maneja la lógica de grabación"""
    try:
        # Manejar el inicio/detención de la grabación
        if not st.session_state.is_recording:
            # Iniciar grabación
            temp_dir = tempfile.gettempdir()
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            st.session_state.output_file = os.path.join(temp_dir, f"record-{timestamp}.wav")

            # Iniciar grabación
            if st.session_state.recorder.start_recording(st.session_state.output_file, verbose=True):
                st.session_state.is_recording = True
                st.session_state.recording_status = "recording"
                st.rerun()
        else:
            # Detener grabación
            st.write("Deteniendo grabación...")
            if st.session_state.recorder.stop_recording():
                st.session_state.recording_status = "processing"

                # Verificar que el archivo existe
                if st.session_state.output_file and os.path.exists(st.session_state.output_file):
                    file_size = os.path.getsize(st.session_state.output_file)
                    if file_size < 1024:
                        raise Exception("La grabación es demasiado corta")

                    # Transcribir el audio
                    st.write("Iniciando transcripción...")
                    transcript = transcribe_audio(st.session_state.output_file, verbose=True, use_local=False)

                    if not transcript or transcript.strip() == "." or len(transcript.strip()) < 2:
                        raise Exception("La transcripción está vacía o no es válida")

                    st.session_state.text_input = transcript.strip()
                    st.session_state.recording_status = "success"
                else:
                    raise Exception("No se encontró el archivo de audio")
            
            # Limpiar                    
            try:
                if st.session_state.output_file and os.path.exists(st.session_state.output_file):
                    os.remove(st.session_state.output_file)
            except Exception as e:
                st.error(f"Error al eliminar el archivo temporal: {str(e)}")
            
            st.session_state.output_file = None
            st.session_state.is_recording = False
            st.session_state.recorder.cleanup()
            st.rerun()
                        
    except Exception as e:
        st.write(f"Error: {str(e)}")
        st.session_state.recording_status = f"error: {str(e)}"
        st.session_state.is_recording = False
        st.session_state.recorder.cleanup()
        st.rerun()

def main():
    # Inicializar el estado de la sesión
    initialize_session_state()

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

    # Chat input
    with st.container():
        col1, col2 = st.columns([4, 1])
        
        with col1:
            # Input de texto
            if 'text_input' not in st.session_state:
                st.session_state.text_input = ""
                
            user_input = st.text_input(
                "Haz tu pregunta sobre las sentencias disponibles:",
                value=st.session_state.text_input,
                key="input_field",
                label_visibility="collapsed"
            )
        with col2:
            # Botón de grabación
            button_text = "⏹️" if st.session_state.is_recording else "🎤"
            button_help = "Detener la grabación" if st.session_state.is_recording else "Iniciar grabación"

            # Botón de grabación con callback
            st.button(button_text, help=button_help, key="record_button", on_click=handle_recording)
            
        # Mostrar mensajes de estado
        if st.session_state.recording_status == "recording":
            st.info("Grabando... Presiona ⏹️ para detener", icon="🎤")
        elif st.session_state.recording_status == "processing":
            st.info("Procesando grabación...", icon="⌛")
        elif st.session_state.recording_status == "success":
            st.success("Grabación transcrita exitosamente!", icon="✅")
            # Limpiar el estado después de mostrar el mensaje
            st.session_state.recording_status = None
        elif st.session_state.recording_status and st.session_state.recording_status.startswith("error"):
            st.error(f"Error: {st.session_state.recording_status[6:]}", icon="❌")
            st.session_state.recording_status = None
                

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

if __name__ == "__main__":
    main()