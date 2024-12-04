import os
import json
from tqdm import tqdm

# Define file paths
file_path = "data/jsonl/sentencias_clasificadas.jsonl"  # Update this to the location of your JSONL file
base_directory = "./data/markdown_no_references"  # Update this to the base directory containing the folders

# Count total lines first
total_lines = sum(1 for _ in open(file_path, "r", encoding="utf-8"))

# Process the JSONL file
with open(file_path, "r", encoding="utf-8") as file:
    # Add progress bar
    for line in tqdm(file, total=total_lines, desc="Processing sentences"):
        # Parse each JSON line
        data = json.loads(line.strip())

        # Extract properties
        tipo_sentencia = data.get("tipo_sentencia", "Unknown")
        sentencia = data.get("Sentencia", "").replace("/", "-")  # Replace "/" with "-"
        texto = data.get("Texto", "").replace(
            "\r\n", " "
        )  # Replace "\r\n" with a space

        # Define folder path based on 'tipo_sentencia'
        folder_path = os.path.join(base_directory, tipo_sentencia)

        # Ensure the folder exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Define markdown file path
        file_name = f"{sentencia}.md"
        file_path = os.path.join(folder_path, file_name)

        # Write the 'Texto' content to the markdown file
        with open(file_path, "w", encoding="utf-8") as md_file:
            md_file.write(texto)

print(
    f"Markdown files created successfully in their respective folders under {base_directory}."
)
