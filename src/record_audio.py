import pyaudio
import wave
import os
from rich.console import Console

console = Console()

# Configuración de audio
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024

class AudioRecorder:
    def __init__(self):
        self.is_recording = False
        self.frames = []
        self.output_file = None
        self.stream = None
        self.pyaudio = None

    def start_recording(self, output_file, verbose=False):
        # Limpiar estado anterior
        self.cleanup()
        self.frames = []
        self.output_file = output_file

        if not output_file:
            raise ValueError("No se especificó un archivo de salida")

        if verbose:
            console.print("[yellow]Iniciando grabación...[/yellow]")

        try:
            self.pyaudio = pyaudio.PyAudio()

            # Obtener información del dispositivo de entrada predeterminado
            default_input = self.pyaudio.get_default_input_device_info()
            if verbose:
                console.print(f"Dispositivo de entrada: {default_input['name']}")

            self.stream = self.pyaudio.open(
                format=FORMAT, 
                channels=CHANNELS, 
                rate=RATE,
                input=True, 
                frames_per_buffer=CHUNK,
                stream_callback=self._audio_callback
            )
            
            # Verificar que el stream está activo
            if not self.stream.is_active():
                raise Exception("No se pudo iniciar la grabación")

            self.is_recording = True

            if verbose:
                console.print("*Grabando... Presiona el botón nuevamente para detener")
            
            return True
        
        except Exception as e:
            if verbose:
                console.print(f"[red]Error al grabar el audio: {str(e)}[/red]")
            self.cleanup()
            raise Exception(f"Error en la grabación de audio: {str(e)}")

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """Callback para la grabación de audio"""
        if self.is_recording:
            self.frames.append(in_data)
        return (in_data, pyaudio.paContinue)
            
    def _save_audio_file(self, verbose=False):
        """Guarda el archivo de audio"""
        try:
            if not self.frames:
                raise Exception("No se grabaron datos de audio")
            
            if verbose:
                console.print("Guardando archivo...")
                console.print(f"Número de frames a guardar: {len(self.frames)}")

            # Asegurarse de que el directorio existe
            os.makedirs(os.path.dirname(self.output_file), exist_ok=True)

            # Cerrar stream y PyAudio
            if self.stream:
                self.stream.stop_stream()
                self.stream.close()
                self.stream = None

            if self.pyaudio:
                self.pyaudio.terminate()
                self.pyaudio = None

            # Preparar los datos de audio
            audio_data = b"".join(self.frames)
            if len(audio_data) == 0:
                raise Exception("No se grabaron datos de audio")

            # Guardar el archivo
            wf = wave.open(self.output_file, "wb")
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(pyaudio.get_sample_size(FORMAT))
            wf.setframerate(RATE)
            wf.writeframes(b"".join(self.frames))
            wf.close()

            # Verificar que el archivo se guardó correctamente
            if not os.path.exists(self.output_file):
                raise Exception(f"El archivo de audio no se guardó correctamente: {self.output_file}")

            file_size = os.path.getsize(self.output_file)
            if file_size == 0:
                raise Exception("El archivo de audio está vacío:")

            if verbose:
                console.print(
                    f"[green]Archivo guardado exitosamente en: {self.output_file}[/green]"
                )
                console.print(
                    f"[blue]Tamaño del archivo: {os.path.getsize(self.output_file)} bytes[/blue]"
                )
            return self.output_file

        except Exception as e:
            if verbose:
                console.print(f"[red]Error al grabar el audio: {str(e)}[/red]")
            raise e

    def stop_recording(self):
        """Detiene la grabación"""
        self.is_recording = False
        if self.frames:
            self._save_audio_file(verbose=True)
            return True
        return False

    def is_currently_recording(self):
        """Retorna el estado de la grabación"""
        return self.is_recording

    def cleanup(self):
        """
        Limpia los recursos de audio
        """
        self.is_recording = False
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except:
                pass
            self.pyaudio = None
        self.frames = []

def record_audio(output_file, verbose=False):
    """
    Graba audio usando PyAudio y lo guarda en un archivo WAV
    """
    if not hasattr(record_audio, 'recorder'):
        record_audio.recorder = AudioRecorder()
    
    recorder = record_audio.recorder
    if not recorder.is_recording:
        # Iniciar nueva grabación
        try:
            recorder.start_recording(output_file, verbose)
            return True
        except Exception as e:
            if verbose:
                console.print(f"[red]Error al iniciar la grabación: {str(e)}[/red]")
            recorder.cleanup()
            raise e
    else:
        # Detener grabación en curso
        try:
            output_file = recorder.stop_recording()
            return True
        except Exception as e:
            if verbose:
                console.print(f"[red]Error al detener la grabación: {str(e)}[/red]")
            recorder.cleanup()
            raise e

