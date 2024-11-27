import os
import select
import sys
import wave

import pyaudio
from rich.console import Console

console = Console()

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100


def record_audio(output_file, verbose=False):
    p = pyaudio.PyAudio()

    stream = p.open(
        format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK
    )

    console.print("[yellow]Recording... Press Enter to stop.[/yellow]")

    frames = []

    try:
        while True:
            if select.select([sys.stdin], [], [], 0.0)[0]:
                if sys.stdin.readline().strip() == "":
                    break
            data = stream.read(CHUNK)
            frames.append(data)
    except KeyboardInterrupt:
        pass

    console.print("[green]Finished recording.[/green]")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_file, "wb")
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b"".join(frames))
    wf.close()

    if verbose:
        console.print(
            f"[yellow]Audio file size: {os.path.getsize(output_file)} bytes[/yellow]"
        )
