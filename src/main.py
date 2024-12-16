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
    default_states = {
        'is_recording': False,
        'recording_status': None,
        'output_file': None,
        'text_input': "",
        'recorder': AudioRecorder(),
        'chat_history': [],
        'chain': None,
        'diccionario_relatorias': {},
        'search_in_progress': False,
        'last_error': None,
        'search_results': None
    }
    
    for key, default_value in default_states.items():
        if key not in st.session_state:
            st.session_state[key] = default_value

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
    try:
        # Inicializar el estado de la sesión
        initialize_session_state()

        st.title("Análisis de Sentencias - Corte Constitucional")

        st.sidebar.header("Buscar Sentencias")
        termino_de_busqueda = st.sidebar.text_input("Ingrese el término de búsqueda", "")

        if st.sidebar.button("Buscar Sentencias"):
            if not termino_de_busqueda.strip():
                st.sidebar.error("Por favor, ingrese un término de búsqueda válido")
                return

            st.session_state.search_in_progress = True
            st.sidebar.success(f"Buscando sentencias con el término: {termino_de_busqueda}...")
            st.sidebar.warning("El análisis puede tomar unos minutos")
            
            try:
                with st.spinner("Realizando la búsqueda..."):
                    # Limpiar el historial y la cadena anterior
                    st.session_state.chat_history = []
                    clear_collection()
                    st.session_state.diccionario_relatorias = scraping_sentencias(termino_de_busqueda)
                    
                    # Intentar inicializar el chain
                    st.session_state.chain = initialize_chain()
                    if st.session_state.chain is None:
                        # Si el chain es None, significa que la base de datos estaba vacía
                        # Intentamos inicializar de nuevo después del scraping
                        st.session_state.chain = initialize_chain()
                    
                    st.session_state.search_results = len(st.session_state.diccionario_relatorias)
                    st.sidebar.success(f"Búsqueda completada. Se encontraron {st.session_state.search_results} resultados.")
            except Exception as e:
                st.session_state.last_error = str(e)
                st.sidebar.error(f"Error al buscar sentencias: {e}")
            finally:
                st.session_state.search_in_progress = False

        # Mostrar resultados de búsqueda
        if st.session_state.diccionario_relatorias:
            with st.expander("Ver resultados de la búsqueda"):
                for enlace in st.session_state.diccionario_relatorias.keys():
                    st.write(f"- {enlace}")

        st.subheader("Chat para Análisis de Sentencias")

        # Agregar un botón para limpiar el historial
        if st.button("Limpiar historial"):
            st.session_state.chat_history = []
            st.session_state.text_input = ""
            st.rerun()

        # Chat input
        with st.container():
            col1, col2 = st.columns([4, 1])
            
            with col1:
                user_input = st.text_input(
                    "Haz tu pregunta sobre las sentencias disponibles:",
                    value=st.session_state.text_input,
                    key="input_field",
                    label_visibility="collapsed"
                )
            
            with col2:
                button_text = "⏹️" if st.session_state.is_recording else "🎤"
                button_help = "Detener la grabación" if st.session_state.is_recording else "Iniciar grabación"
                st.button(button_text, help=button_help, key="record_button", on_click=handle_recording)

        # Mostrar estado de grabación
        if st.session_state.recording_status == "recording":
            st.info("Grabando... Presiona ⏹️ para detener", icon="🎤")
        elif st.session_state.recording_status == "processing":
            st.info("Procesando grabación...", icon="⌛")
        elif st.session_state.recording_status == "success":
            st.success("Grabación transcrita exitosamente!", icon="✅")
            st.session_state.recording_status = None
        elif st.session_state.recording_status and st.session_state.recording_status.startswith("error"):
            st.error(f"Error: {st.session_state.recording_status[6:]}", icon="❌")
            st.session_state.recording_status = None

        if st.button("Enviar"):
            if user_input is None or not user_input.strip():
                st.error("Por favor, ingrese una pregunta válida")
                return

            if not st.session_state.chain:
                st.error("Por favor, primero realice una búsqueda de sentencias.")
                return

            st.session_state.chat_history.append({"role": "user", "content": user_input})

            try:
                with st.spinner("Analizando la respuesta..."):
                    result = st.session_state.chain.invoke(user_input)
                    st.session_state.chat_history.append(
                        {"role": "assistant", "content": result}
                    )
                    st.session_state.text_input = ""
                    st.rerun()
            except Exception as e:
                error_msg = f"Error al procesar la pregunta: {str(e)}"
                st.error(error_msg)
                st.session_state.chat_history.append(
                    {"role": "assistant", "content": error_msg}
                )

        # Mostrar el historial del chat
        for message in st.session_state.chat_history:
            role_icon = "👤" if message["role"] == "user" else "🤖"
            st.markdown(f"**{role_icon} {message['role'].title()}:** {message['content']}")

    except Exception as e:
        st.error(f"Error en la aplicación: {str(e)}")

if __name__ == "__main__":
    main()