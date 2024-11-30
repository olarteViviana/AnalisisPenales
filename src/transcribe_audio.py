from litellm import transcription
from rich.console import Console

console = Console()


def transcribe_audio(audio_file_path, verbose, use_local):
    if verbose:
        console.print("[yellow]Attempting to run transcription[/yellow]")

    if use_local:
        pass
    else:
        response = transcribe_hosted(audio_file_path)

    # for debugging
    # print(f"Transcription: {response}")
    return response


def transcribe_hosted(audio_file_path):
    with open(audio_file_path, "rb") as audio_file:
        response = transcription(
            model="groq/whisper-large-v3-turbo",
            file=audio_file,
            prompt="Specify context or spelling",
            temperature=0,
            response_format="json",
        )
    return response["text"]
