import csv
import datetime
import json
import os

# from litellm import completion
import ollama

date = datetime.datetime.now().strftime("%Y-%m-%d")

# Constants
TOKEN_LIMIT = 5000
CSV_FILEPATH = "./data/01-sentencias.csv"
OUTPUT_CSV_FILEPATH = "./data/02-sentencias[facts].csv"

# create folder for backup files if it does not exist
os.makedirs(f"./data/bk/{date}", exist_ok=True)
BK_BASE_PATH = f"./data/bk/{date}/02-sentencias[facts]"
ROWS_PER_BACKUP = 5


#############################
# Load data from a CSV file #
#############################
def load_csv_data(filepath):
    """Load data from a CSV file."""
    with open(filepath, mode="r", newline="", encoding="utf-8") as csv_file:
        reader = csv.DictReader(csv_file)
        return list(reader)


############################
# Read content from a file #
############################
def read_file_content(file_path):
    """Read content from a file."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            return file.read()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return None


###############################
# Create a prompt for the LLM #
###############################
def create_prompt(sentencia, content):
    """Create a prompt for the LLM."""
    schema = """
    {
        "facts": "string",
        "facts_in_simple_words": "string"
    }
    """
    prompt = f"""
        You are an expert legal assistant tasked with understanding the full context of a legal document ("sentencia") and extracting full-detailed information about the background and factual circumstances. Follow these steps carefully:\n
        1. **Understand the Sentencia:** Take a deep breath and carefully read the provided legal document to identify the full **context and/or facts that led to the legal case**. \n
        2. **Extract Information:**\n
            - **Facts**: Provide a detailed and comprehensive description of the background and factual circumstances **as described in the sentencia**. Ensure this includes all relevant events, actions, participants, and key dates that resulted in the filing of the action. Be exhaustive and ensure no significant detail is left out.\n
            - **Facts in Simple Words**: rewrite the facts in simple, everyday language. Use clear, common words to explain the situation in a way that is easy for anyone to understand, while still capturing the essential details.\n
        3. **Output Format:** Present the extracted information in JSON format, strictly adhering to the following schema:\n
        {schema}\n\n
        **Important:** \n
            - For "facts," include all the details provided in the sentencia without omitting any important context.\n
            - For "facts_in_simple_words," ensure clarity and simplicity, avoiding legal jargon.\n
            - Output must strictly follow the JSON structure without any additional text.\n
            - Write all values in Spanish.\n\n
        **Sentencia:** {sentencia}\n
        {content}
    """

    return prompt


#####################################
# Process a single row from the CSV #
#####################################
def process_row(row, writer):
    """Process a single row from the CSV."""
    if int(row["tokens"]) < TOKEN_LIMIT:
        sentencia = row["sentencia"]
        file_path = row["file_path"]
        content = read_file_content(file_path)

        if content is None:
            fact_extraction = False
            parsed_json = {
                "sentencia_fact": sentencia,
                "fact_extraction": fact_extraction,
                "facts": "",
                "facts_in_simple_words": "",
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "total_tokens": 0,
            }
        else:
            prompt = create_prompt(sentencia, content)
            message = [{"role": "user", "content": prompt}]

            params = {
                "model": "gemma2-simpo-8k",
                # "temperature": 0.5,
            }

            params["messages"] = message
            params["format"] = "json"

            try:
                # response = completion(**params)
                response = ollama.chat(**params)
                # llm_response = response.choices[0].message.content
                print(response["message"]["content"])
                # parsed_json = extract_json_from_response(llm_response)

                # if parsed_json is None:
                #     fact_extraction = False
                #     parsed_json = {
                #         "sentencia_fact": sentencia,
                #         "fact_extraction": fact_extraction,
                #         "facts": "",
                #         "facts_in_simple_words": "",
                #         "prompt_tokens": 0,
                #         "completion_tokens": 0,
                #         "total_tokens": 0,
                #     }
                # else:
                #     fact_extraction = True
                #     parsed_json = {
                #         "sentencia_fact": sentencia,
                #         "fact_extraction": fact_extraction,
                #         **parsed_json,
                #         "prompt_tokens": response.usage.prompt_tokens,
                #         "completion_tokens": response.usage.completion_tokens,
                #         "total_tokens": response.usage.total_tokens,
                #     }

                #     # print the parsed json well formatted to the console
                #     print(json.dumps(parsed_json, indent=4, ensure_ascii=False))

            except Exception as e:
                print(f"Error processing LLM response: {e}")
    #             fact_extraction = False
    #             parsed_json = {
    #                 "sentencia_fact": sentencia,
    #                 "fact_extraction": fact_extraction,
    #                 "facts": "",
    #                 "facts_in_simple_words": "",
    #                 "prompt_tokens": 0,
    #                 "completion_tokens": 0,
    #                 "total_tokens": 0,
    #             }

    #     # Write the parsed JSON to the CSV
    #     writer.writerow({**row, **parsed_json})
    # else:
    #     print(f"Skipping {row['sentencia']} as it has more than {TOKEN_LIMIT} tokens.")
    #     parsed_json = {
    #         "sentencia_fact": "",
    #         "fact_extraction": "",
    #         "facts": "",
    #         "facts_in_simple_words": "",
    #         "prompt_tokens": "",
    #         "completion_tokens": "",
    #         "total_tokens": "",
    #     }
    #     writer.writerow({**row, **parsed_json})


################################
# Parse JSON from LLM response #
################################
def extract_json_from_response(llm_response):
    """Extract JSON from the LLM response."""
    try:
        start = llm_response.find("```json") + len("```json")
        end = llm_response.rfind("```")
        json_llm_response = llm_response[start:end].strip()
        return json.loads(json_llm_response)
    except Exception as e:
        print(f"Error extracting JSON from LLM response: {e}")
        return None


def main():
    """Main function to process CSV data and write the output to a new CSV file."""
    rows = load_csv_data(CSV_FILEPATH)

    # Open the output CSV file for writing
    with open(
        OUTPUT_CSV_FILEPATH, mode="w", newline="", encoding="utf-8"
    ) as output_csv_file:
        # Get the fieldnames from the original CSV
        original_fieldnames = rows[0].keys() if rows else []
        new_fieldnames = [
            "sentencia_fact",
            "fact_extraction",
            "facts",
            "facts_in_simple_words",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        ]
        fieldnames = list(original_fieldnames) + new_fieldnames

        writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)
        writer.writeheader()

        i = 0
        for row in rows:
            i += 1
            process_row(row, writer)
            # Backup OUTPUT_CSV_FILEPATH file for debugging purposes rewrite it each ROWS_PER_BACKUP
            if i % ROWS_PER_BACKUP == 0:
                print(f"\nGenerating backup file {BK_BASE_PATH}_bk[{i}].csv\n")
                output_csv_file.flush()
                with open(
                    OUTPUT_CSV_FILEPATH, mode="r", newline="", encoding="utf-8"
                ) as output_csv_file_read:
                    with open(
                        f"{BK_BASE_PATH}_bk[{i}].csv",
                        mode="w",
                        newline="",
                        encoding="utf-8",
                    ) as backup_csv_file:
                        backup_csv_file.write(output_csv_file_read.read())
                    writer = csv.DictWriter(output_csv_file, fieldnames=fieldnames)


if __name__ == "__main__":
    main()
