import os
import csv
import json
import argparse
import time
import re
from google import genai
from google.genai import types
from corpus_utils import get_akkadian_context_lines, remove_nul

PRICE_PER_M_INPUT = 0.10 # gemini-2.5-flash-lite
PRICE_PER_M_OUTPUT = 0.40 # gemini-2.5-flash-lite
BATCH_SIZE = 2

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Estimate costs without querying LLM")
    parser.add_argument("--limit", type=int, default=0, help="Process only N entries")
    parser.add_argument("--start", type=int, default=1, help="Start processing from specific record number (1-indexed)")
    parser.add_argument("--show-prompt", action="store_true", help="Print the prompt for debugging")
    args = parser.parse_args()
    
    input_file = "workspace/publications.csv"
    output_jsonl = "workspace/outputs/publications/publication_translations.jsonl"
    
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

Extract the pairs from all the provided texts into a valid JSON object matching this schema. Note: The source text translations could be in English, German, French, Turkish, etc. Use an appropriate 3-letter ISO language code (e.g., "eng", "deu", "fra", "tur") as the key for the modern language translation:
{{
  "translations": [
    {{
      "<iso3_code>": "translated text in modern language",
      "akk": "transliterated or normalized Akkadian text"
    }}
  ],
  "unpaired": [
    "akkadian text that could not be paired with a translation, do not repeat if already paired"
  ]
}}

No need to preserve the line breaks in any texts, just return the raw texts in one line.
If there is no Akkadian text, or if you cannot confidently extract pairs or unpaired phrases, return empty lists. Do not add any markdown formatting, just provide the raw JSON without md entities or wrappers. STRICT REQUIREMENT: Ensure any literal backslash characters (\) found in the text are properly double-escaped (\\) so the JSON is completely valid.
"""

    if args.dry_run:
        total_input_chars = 0
        total_output_tokens_estimated = 0
        
        for i in range(0, len(records_to_process), BATCH_SIZE):
            batch = records_to_process[i:i+BATCH_SIZE]
            batched_texts = []
            
            for row in batch:
                pdf_name = row.get("pdf_name", "")
                page = row.get("page", "")
                page_text = row.get("page_text", "")
                page_text = get_akkadian_context_lines(page_text)
                batched_texts.append(f"--- Page {page} from {pdf_name} ---\n{page_text}")
            
            batched_pages = "\n----\n".join(batched_texts)
            prompt = prompt_template.format(batched_pages=batched_pages)
            
            if args.show_prompt and i == 0:
                print(f"--- Sample Prompt (Batch 1) ---")
                print(prompt)
                print("-" * 40)
            
            total_input_chars += len(prompt)
            # rough estimate: 200 output tokens per page in the batch
            total_output_tokens_estimated += 200 * len(batch)
            
        total_input_tokens = total_input_chars / 4
        
        input_cost = (total_input_tokens / 1000000) * PRICE_PER_M_INPUT
        output_cost = (total_output_tokens_estimated / 1000000) * PRICE_PER_M_OUTPUT
        
        print(f"--- Dry Run Estimation ---")
        print(f"Estimated Input Tokens: {total_input_tokens:,.0f} tokens")
        print(f"Estimated Output Tokens: {total_output_tokens_estimated:,.0f} tokens")
        print(f"Estimated Input Cost: ${input_cost:.4f}")
        print(f"Estimated Output Cost: ${output_cost:.4f}")
        print(f"Total Estimated Cost: ${(input_cost + output_cost):.4f}")
        return

    # Normal execution
    if not os.environ.get("GEMINI_API_KEY"):
        print("Please set the GEMINI_API_KEY environment variable.")
        return

    client = genai.Client()
    
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
                page_text = get_akkadian_context_lines(page_text)
                batched_texts.append(f"--- Page {page} from {pdf_name} ---\n{page_text}")
            
            batched_pages = "\n----\n".join(batched_texts)
            prompt = prompt_template.format(batched_pages=batched_pages)
            max_tokens = (len(prompt) / 4) * 1.5
            print("prompt len:", len(prompt))
            
            if args.show_prompt:
                print(f"--- Prompt (Batch {i // BATCH_SIZE + 1}) ---")
                print(prompt)
                print("-" * 40)
            
            error_log = "workspace/outputs/publications/error_log.txt"
            max_retries = 3
            success = False
            
            for attempt in range(max_retries):
                try:
                    response = client.models.generate_content(
                        model='gemini-2.5-flash-lite',
                        contents=prompt,
                        config=types.GenerateContentConfig(
                            response_mime_type="application/json",
                            max_output_tokens=max_tokens,
                            temperature=0.1,
                        )
                    )
                    
                    try:
                        print("got response, will parse")
                        parsed = json.loads(response.text, strict=False)
                    except json.JSONDecodeError as de:
                        # Fallback pattern for unescaped literal slashes generated by language models (e.g. \m or \a)
                        cleaned_text = response.text.replace("\\", "\\\\").replace("\\\\n", "\\n").replace('\\\\"', '\\"')
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
                        if 'response' in locals() and hasattr(response, 'text'):
                            erf.write(f"Bad response from LLM:\n{response.text}\n")
                        else:
                            erf.write("No response from LLM or other error occurred.\n")
                        erf.write("-----------------------\n")
                    time.sleep((attempt + 1) * 2) # Backoff delay
            
            if not success:
                print(f"Failed to process batch {i // BATCH_SIZE + 1} after {max_retries} attempts. See {error_log}")
            
            #time.sleep(2) # Rate limit delay
            
    print(f"Extraction complete! Results appended to {output_jsonl}")

if __name__ == "__main__":
    main()
