import os
import csv
import json
import argparse
import time
import re
import ollama
from corpus_utils import get_akkadian_context_lines, remove_nul

CONTEXT_LINES = 2
BATCH_SIZE = 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="qwen2.5-coder:1.5b", help="Ollama model to use (default: mistral)")
    parser.add_argument("--limit", type=int, default=0, help="Process only N entries")
    parser.add_argument("--start", type=int, default=1, help="Start processing from specific record number (1-indexed)")
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt for debugging")
    parser.add_argument("--host", type=str, default="http://localhost:11434", help="Ollama server host (default: http://localhost:11434)")
    args = parser.parse_args()
    
    input_file = "workspace/publications.csv"
    output_jsonl = "workspace/outputs/publications/publication_translations_ollama.jsonl"
    
    records_to_process = []
    
    print("Loading CSV...")
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(remove_nul(f))
        for row in reader:
            has_akk = row.get("has_akkadian", "").strip().lower()
            if has_akk == "true":
                records_to_process.append(row)

    print(f"Total {len(records_to_process)} records with Akkadian text found in CSV.")
    
    # Adjust for --start logic
    start_index = max(0, args.start - 1)
    records_to_process = records_to_process[start_index:]
    
    # If the user sets a limit, we should limit AFTER applying the start slice
    if args.limit > 0:
        records_to_process = records_to_process[:args.limit]
        
    print(f"Starting execution at record {args.start}. {len(records_to_process)} records queued for processing.")

    prompt_template = """You are an expert Assyriologist. Your task is to process the following OCR text from academic publications and identify possible Akkadian texts and its translation to the document language, do not try to translate yourself, just extract the translation pairs found at the text verbatim.

Input Texts:
{batched_pages}

Extract the pairs from all the provided texts into a valid JSON object matching this schema. Note: The source text translations could be in English, German, French, Turkish, etc. Use an appropriate 3-letter ISO language code (e.g., "eng", "deu", "fra", "tur", etc) as the key for the modern language translation, use "akk" for akaddian; put the pairs in this order akk->other disregard the order in which they were found on the text, no need to reverse translation pairs, only include the translation pair if they are effectively found on text, do not try to translate yourself, just copy texts. If you cannot confidently identify any Akkadian text, return an empty list for "translations". If there are Akkadian texts that cannot be confidently paired with a translation, include them in the "unpaired" list and do not include non akkadian texts on the unpaired section. Do not include any text in the "unpaired" list that is already included in the "translations" pairs.:
{{
  "translations": [
    {{
      "akk": "transliterated or normalized Akkadian text",
      "<iso3_code>": "translated text in modern language"
    }}
  ],
  "unpaired": [
    "akkadian text that could not be paired with a translation, do not repeat if already paired, don't put unpaired texts in other languages than Akkadian"
  ]
}}

No need to preserve the line breaks in any texts, just return the raw texts in one line.
If there is no Akkadian text, or if you cannot confidently extract pairs or unpaired phrases, return empty lists. Do not add any markdown formatting, just provide the raw JSON without md entities or wrappers. STRICT REQUIREMENT: Ensure any literal backslash characters (\) found in the text are properly double-escaped (\\) so the JSON is completely valid.
"""

    # Normal execution
    print(f"Connecting to Ollama at {args.host}...")
    client = ollama.Client(host=args.host)
    
    # Test connection
    try:
        # Try to list models to verify connection
        client.list()
        print(f"Connected to Ollama. Using model: {args.model}")
    except Exception as e:
        print(f"Failed to connect to Ollama at {args.host}: {e}")
        return
    
    os.makedirs(os.path.dirname(output_jsonl), exist_ok=True)
    
    print("Starting LLM extraction...")
    with open(output_jsonl, "a", encoding="utf-8") as out_f:
        for i in range(0, len(records_to_process), BATCH_SIZE):
            batch = records_to_process[i:i+BATCH_SIZE]
            batched_texts = []
            
            # To display accurate absolute record numbers
            absolute_start_record = args.start + i
            absolute_end_record = absolute_start_record + len(batch) - 1
            
            print(f"Processing batch {i // BATCH_SIZE + 1}/{(len(records_to_process) + BATCH_SIZE - 1) // BATCH_SIZE} (absolute records {absolute_start_record} to {absolute_end_record})")
            
            for row in batch:
                pdf_name = row.get("pdf_name", "")
                page = row.get("page", "")
                page_text = row.get("page_text", "")
                page_text = get_akkadian_context_lines(page_text, lines_margin=CONTEXT_LINES)
                if page_text:
                    batched_texts.append(f"--- Page {page} from {pdf_name} ---\n{page_text}")
            
            if not batched_texts:
                print(f"Skipping batch {i // BATCH_SIZE + 1} as no Akkadian text was found in any pages.")
                continue
            
            batched_pages = "\n----\n".join(batched_texts)
            prompt = prompt_template.format(batched_pages=batched_pages)
            print("prompt len:", len(prompt))
            
            if args.show_prompt:
                print(f"--- Prompt (Batch {i // BATCH_SIZE + 1}) ---")
                print(prompt)
                print("-" * 40)
            
            error_log = "workspace/outputs/publications/error_log_ollama.txt"
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    response = client.generate(
                        model=args.model,
                        prompt=prompt,
                        stream=False,
                        options={
                            #"temperature": 0.1,
                            #"num_predict": 2048,
                        }
                    )
                    
                    response_text = response.get("response", "").strip()
                    
                    try:
                        print("got response, will parse")
                        parsed = json.loads(response_text, strict=False)
                    except json.JSONDecodeError as de:
                        # Fallback pattern for unescaped literal slashes generated by language models (e.g. \m or \a)
                        cleaned_text = response_text.replace("\\", "\\\\").replace("\\\\n", "\\n").replace('\\\\"', '\\"')
                        parsed = json.loads(cleaned_text, strict=False)
                    
                    # Output without per-document distinction as requested
                    out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                    out_f.flush()
                    print("parsed and written")
                    success = True
                    break
                    
                except Exception as e:
                    print(f"Attempt {attempt + 1} failed for batch {i // BATCH_SIZE + 1}: {e}")
                    with open(error_log, "a", encoding="utf-8") as erf:
                        erf.write(f"--- FAILED ATTEMPT {attempt + 1} BATCH {i // BATCH_SIZE + 1} (absolute start: {absolute_start_record}) ---\n")
                        erf.write(f"Error: {e}\n")
                        if 'response_text' in locals():
                            erf.write(f"Bad response from LLM:\n{response_text}\n")
                        else:
                            erf.write("No response from LLM or other error occurred.\n")
                        erf.write("-----------------------\n")
                    time.sleep((attempt + 1) * 2) # Backoff delay
            
            if not success:
                print(f"Failed to process batch {i // BATCH_SIZE + 1} after {max_retries} attempts. See {error_log}")
    
    print(f"Extraction complete! Results appended to {output_jsonl}")

if __name__ == "__main__":
    main()
