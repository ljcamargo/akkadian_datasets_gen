import os
import csv
import io
import argparse
import time
import re
import ollama
from corpus_utils import get_akkadian_context_lines, remove_nul

CONTEXT_LINES = 2
BATCH_SIZE = 2

# CSV dialect expected in LLM responses
CSV_DELIMITER = "|"
CSV_QUOTECHAR = '"'


def strip_markdown_codeblock(text: str) -> str:
    """Remove markdown code fences if the LLM wrapped its response in them."""
    # Match ```csv ... ```, ``` ... ```, or any variant
    text = text.strip()
    match = re.match(r"^```[a-zA-Z]*\n?(.*?)```$", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Also strip a single leading/trailing fence line if only one side is present
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def parse_csv_response(response_text: str) -> list[dict]:
    """
    Parse the LLM's CSV response into a list of {'akkadian': ..., 'english': ...} dicts.
    Tolerates markdown code block wrapping and minor whitespace issues.
    Returns only rows where at least one column is non-empty.
    """
    cleaned = strip_markdown_codeblock(response_text)

    reader = csv.DictReader(
        io.StringIO(cleaned),
        delimiter=CSV_DELIMITER,
        quotechar=CSV_QUOTECHAR,
    )

    rows = []
    for row in reader:
        # Normalize keys: strip whitespace, lowercase
        normalized = {k.strip().lower(): v.strip() if v else "" for k, v in row.items()}
        akk = normalized.get("akkadian", "")
        eng = normalized.get("english", "")
        # Skip completely empty rows and rows where both cols are identical
        # (guards against the LLM copying source into both columns)
        if not akk and not eng:
            continue
        if akk and akk == eng:
            continue
        rows.append({"akkadian": akk, "english": eng})

    return rows


def append_rows_to_csv(output_csv: str, rows: list[dict], write_header: bool):
    """Append parsed rows to the global output CSV."""
    with open(output_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["akkadian", "english"],
            delimiter=",",
            quotechar='"',
            quoting=csv.QUOTE_MINIMAL,
        )
        if write_header:
            writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5:1.5b", help="Ollama model to use")
    parser.add_argument("--limit", type=int, default=0, help="Process only N entries")
    parser.add_argument("--start", type=int, default=1, help="Start processing from specific record number (1-indexed)")
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt for debugging")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama server host")
    args = parser.parse_args()

    input_file = "workspace/publications.csv"
    output_csv = "workspace/outputs/publications/publication_translations_ollama.csv"
    error_log = "workspace/outputs/publications/error_log_ollama.txt"

    records_to_process = []

    print("Loading CSV...")
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(remove_nul(f))
        for row in reader:
            has_akk = row.get("has_akkadian", "").strip().lower()
            if has_akk == "true":
                records_to_process.append(row)

    print(f"Total {len(records_to_process)} records with Akkadian text found in CSV.")

    start_index = max(0, args.start - 1)
    records_to_process = records_to_process[start_index:]

    if args.limit > 0:
        records_to_process = records_to_process[:args.limit]

    print(f"Starting execution at record {args.start}. {len(records_to_process)} records queued for processing.")

    # -------------------------------------------------------------------------
    # Chat messages — split into system + user for instruct models.
    # client.chat() applies the model's chat template automatically, which is
    # essential for instruct-tuned models (qwen2.5-instruct, mistral-instruct,
    # llama3-instruct, etc.).  Never use client.generate() with a raw prompt
    # for these models — the chat template is what makes them follow instructions.
    # -------------------------------------------------------------------------
    system_message = """You are an expert Assyriologist. Your task is to scan the following OCR text from academic publications and extract pairs of Akkadian text together with their English translation.

STRICT RULES — READ CAREFULLY:
- DO NOT invent, guess, or fabricate any translation. Only extract pairs that are explicitly present in the text.
- If there is no akkadian text, do not repeat put english text on the akkadian column, its totally fine and desirable, to leave blank cells, even if everything is blank,
- Do not use placeholders for missing translations, leave it in blank
- The akkadian column must not contain english texts under any circumstance, leave blank if no akkadian text, english of course can contain unstranslatable words of akkadian
- It is COMPLETELY FINE — and very common — to leave the english column blank when no translation is visible in the text. An empty english cell is always better than a made-up one.
- If a passage is in German, French, Turkish, or another modern language instead of English, translate it into English ONLY if you are confident. If you are unsure, leave the english column blank.
- DO NOT copy the same text into both columns. If you cannot find a pairing, put the Akkadian text in the akkadian column and leave english blank.
- DO NOT include any non-Akkadian text in the akkadian column.
- NO explanations, NO comments, NO extra text, NO markdown — ONLY the CSV rows with the exact headers "akkadian" and "english".
- DON'T wrap the CSV in markdown code blocks or any other formatting, just raw CSV text.
- CSV headers are mandatory, no wrapping in quotes
- If no Akkadian text can be identified at all, respond with ONLY the header row and nothing else.

Respond ONLY with a pipe-delimited (|), no quotes, Unix line endings (\\n) CSV. No explanations, no markdown, no extra text — just the CSV.
The CSV must have exactly two columns with these exact header names (case-sensitive):
akkadian|english

Example of a valid response (note: empty english is fine):
akkadian|english
ana bēlīya qibīma GIN|Say to my lord GIN
itti PN ana āli illiku|
awīlum šū damqam ipu|That man did something good


Do not wrap your response in markdown code blocks or backticks.
"""

    user_template = (
        "Extract Akkadian–English pairs from the following OCR pages. "
        "Follow all rules from the system prompt exactly.\n\n"
        "{batched_pages}\n\n"
        "Reply with the CSV and nothing else."
    )

    print(f"Connecting to Ollama at {args.host}...")
    client = ollama.Client(host=args.host)

    try:
        client.list()
        print(f"Connected to Ollama. Using model: {args.model}")
    except Exception as e:
        print(f"Failed to connect to Ollama at {args.host}: {e}")
        return

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Determine whether we need to write the CSV header.
    # If the file already exists and has content, skip header to avoid duplicates.
    output_csv_exists = os.path.isfile(output_csv) and os.path.getsize(output_csv) > 0

    print("Starting LLM extraction...")
    for i in range(0, len(records_to_process), BATCH_SIZE):
        batch = records_to_process[i:i + BATCH_SIZE]
        batched_texts = []

        absolute_start_record = args.start + i
        absolute_end_record = absolute_start_record + len(batch) - 1

        total_batches = (len(records_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
        print(
            f"Processing batch {i // BATCH_SIZE + 1}/{total_batches} "
            f"(absolute records {absolute_start_record}–{absolute_end_record})"
        )

        for row in batch:
            pdf_name = row.get("pdf_name", "")
            page = row.get("page", "")
            page_text = row.get("page_text", "")
            page_text = get_akkadian_context_lines(page_text, lines_margin=CONTEXT_LINES)
            if page_text:
                batched_texts.append(f"--- Page {page} from {pdf_name} ---\n{page_text}")

        if not batched_texts:
            print(f"Skipping batch {i // BATCH_SIZE + 1}: no usable Akkadian text found.")
            continue

        batched_pages = "\n----\n".join(batched_texts)
        user_message = user_template.format(batched_pages=batched_pages)

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user",   "content": user_message},
        ]

        print(f"  User message length: {len(user_message)} chars")

        if args.show_prompt:
            print(f"--- System message (Batch {i // BATCH_SIZE + 1}) ---")
            print(system_message)
            print(f"--- User message (Batch {i // BATCH_SIZE + 1}) ---")
            print(user_message)
            print("-" * 40)

        max_retries = 3
        success = False
        response_text = None

        for attempt in range(max_retries):
            try:
                response = client.chat(
                    model=args.model,
                    messages=messages,
                    stream=False,
                    options={},
                )

                # chat() returns a ChatResponse object; message content is nested
                response_text = response.message.content.strip()
                print(f"  Got response ({len(response_text)} chars), parsing CSV...")

                rows = parse_csv_response(response_text)
                print(f"  Parsed {len(rows)} row(s).")

                write_header = not output_csv_exists
                append_rows_to_csv(output_csv, rows, write_header=write_header)
                output_csv_exists = True  # header written on first successful write

                success = True
                break

            except Exception as e:
                print(f"  Attempt {attempt + 1} failed: {e}")
                with open(error_log, "a", encoding="utf-8") as erf:
                    erf.write(
                        f"--- FAILED ATTEMPT {attempt + 1} "
                        f"BATCH {i // BATCH_SIZE + 1} "
                        f"(absolute start: {absolute_start_record}) ---\n"
                    )
                    erf.write(f"Error: {e}\n")
                    if response_text is not None:
                        erf.write(f"Raw LLM response:\n{response_text}\n")
                    else:
                        erf.write("No response received from LLM.\n")
                    erf.write("-" * 40 + "\n")
                time.sleep((attempt + 1) * 2)

        if not success:
            print(
                f"  Batch {i // BATCH_SIZE + 1} failed after {max_retries} attempts. "
                f"See {error_log}"
            )

    print(f"Extraction complete! Results written to {output_csv}")


if __name__ == "__main__":
    main()