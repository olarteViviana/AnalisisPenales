import os
import tiktoken
import csv
from tqdm import tqdm


def count_tokens(text):
    """
    Counts the number of tokens in the given text using the tiktoken library.
    """
    try:
        tokenizer = tiktoken.get_encoding("cl100k_base")
        tokens = tokenizer.encode(text)
        return len(tokens)
    except Exception as e:
        print(f"Error in tokenization: {e}")
        return 0


def process_files(directory, csv_filepath):
    """
    Processes markdown files in the specified directory:
    - Counts tokens in each file.
    - Writes the results to a CSV file.
    """
    try:
        with open(csv_filepath, mode="w", newline="", encoding="utf-8") as csv_file:
            fieldnames = ["sentencia", "year", "file_path", "tokens"]
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()

            files = [f for f in os.listdir(directory) if f.endswith(".md")]
            for filename in tqdm(files, desc="Processing files"):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, "r", encoding="utf-8") as file:
                        content = file.read()
                    token_count = count_tokens(content)
                    year, sentencia = (
                        filename.split("-")[0],
                        "-".join(filename.split("-")[1:-1]),
                    )
                    writer.writerow(
                        {
                            "sentencia": sentencia,
                            "year": year,
                            "file_path": filepath,
                            "tokens": token_count,
                        }
                    )
                except Exception as e:
                    print(f"Error processing file {filename}: {e}")
    except FileNotFoundError:
        print(f"Directory not found: {directory}")
    except Exception as e:
        print(f"Error reading directory {directory}: {e}")


if __name__ == "__main__":
    directory = "./data/markdown_no_references/Test"
    csv_filepath = "./data/sentencias_t.csv"
    process_files(directory, csv_filepath)
