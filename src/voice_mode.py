import argparse
import json
from datetime import datetime

import litellm
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from record_audio import record_audio
from transcribe_audio import transcribe_audio

console = Console()


ASCII_ART = r"""
                   _                       _____   _   
 ___   ___  _ __  | |_  ___  _ __    ___   \_   \ /_\  
/ __| / _ \| '_ \ | __|/ _ \| '_ \  / __|   / /\///_\\ 
\__ \|  __/| | | || |_|  __/| | | || (__ /\/ /_ /  _  \
|___/ \___||_| |_| \__|\___||_| |_| \___|\____/ \_/ \_/
by UdeMedellín (entrada de voz)
"""
console.print(Text(ASCII_ART, style="bold blue"))

# information about the model
params = {"model": "groq/llama-3.1-70b-versatile", "temperature": 0.3, "stream": True}


def detect_topic(text, params) -> str:
    prompt = f"""
    Please identify the main topic the user want to know about in the following text. Respond in JSON format using this schema: {{"topics": ["topic1", "topic2", ...]}} with no preamble or additional text.
    
    **User Query:** {text}
    """
    params["messages"] = [{"role": "user", "content": prompt}]
    response = litellm.completion(**params)
    assistant_message = ""
    for chunk in response:
        content = chunk.choices[0].delta.content
        if content:
            assistant_message += content
    assistant_message.strip()
    # Parse the string into a dictionary
    data_dict = json.loads(assistant_message)
    # Pretty-print the JSON object
    pretty_json = json.dumps(data_dict, indent=4, ensure_ascii=False)
    return pretty_json


def display_rich_output(transcript, topics) -> None:
    # Print the ASCII art directly without a border

    # Create panels using Panel with expand=True
    transcript_panel = Panel(
        transcript.strip(),
        title="Transcript",
        border_style="cyan",
        padding=(1, 2),
        expand=True,
    )

    topics_panel = Panel(
        topics, title="Topics", border_style="cyan", padding=(1, 2), expand=True
    )

    # Display the panels
    console.print(transcript_panel)
    console.print(topics_panel)
    # Add similar panels for summary, sentiment, intent, and topics if needed


def main() -> None:
    parser = argparse.ArgumentParser(description="Voice Assistant")
    parser.add_argument(
        "--local", action="store_true", help="Use local transcription model"
    )
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_file = f"record-{timestamp}.wav"
    record_audio(output_file, verbose=True)
    transcript = transcribe_audio(output_file, verbose=True, use_local=args.local)
    topics = detect_topic(transcript, params)

    display_rich_output(transcript, topics)


if __name__ == "__main__":
    main()

