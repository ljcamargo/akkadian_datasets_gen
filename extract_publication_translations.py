import os
import csv
import json
import argparse
import time
import re
from google import genai
from google.genai import types

def remove_nul(file_iter):
    for line in file_iter:
        yield line.replace('\0', '')

def get_akkadian_context_lines(page_text):
    separator = '\\n' if '\\n' in page_text and '\n' not in page_text else '\n'
    lines = page_text.split(separator)
    #print(f">>>>>>>>>>>>>>> Total lines: {len(lines)}")
    pattern = re.compile(r'(?:[\w\.]+[-])+[\w\.]+', re.UNICODE) 
    
    akk_line_indices = []
    for i, line in enumerate(lines):
        matches = pattern.findall(line)
        if len(matches) >= 2:
            akk_line_indices.append(i)
            
    if not akk_line_indices:
        print(">>>>>>>>>>>>>>> No Akkadian text matched")
        return page_text
        
    include_indices = set()
    for idx in akk_line_indices:
        for j in range(max(0, idx - 2), min(len(lines), idx + 3)):
            include_indices.add(j)
            
    sorted_indices = sorted(list(include_indices))
    #print(f">>>>>>>>>>>>>>> Akkadian text matched {len(sorted_indices)} lines from {len(lines)} lines")
    
    result_lines = []
    for idx in sorted_indices:
        result_lines.append(lines[idx])
        
    return '\n'.join(result_lines)

PRICE_PER_M_INPUT = 0.10 # gemini-2.5-flash-lite
PRICE_PER_M_OUTPUT = 0.40 # gemini-2.5-flash-lite
BATCH_SIZE = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Estimate costs without querying LLM")
    parser.add_argument("--limit", type=int, default=0, help="Process only N entries")
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
                if args.limit > 0 and len(records_to_process) >= args.limit:
                    break

    print(f"Found {len(records_to_process)} records with Akkadian text to process.")

    prompt_template = """You are an expert Assyriologist. Your task is to process the following OCR text from academic publications and identify possible Akkadian text and its English (or German/French/etc.) translation pairs.

Input Texts:
{batched_pages}

Extract the pairs from all the provided texts into a valid JSON object matching this schema:
{{
  "translations": [
    {{
      "eng": "translated text in modern language, could be eng or others",
      "akk": "transliterated or normalized Akkadian text"
    }}
  ],
  "unpaired_akkadian": [
    "akkadian text that could not be paired with a translation"
  ]
}}

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
            
            print(f"Processing batch {i // BATCH_SIZE + 1}/{(len(records_to_process) + BATCH_SIZE - 1) // BATCH_SIZE} (records {i+1} to {min(i+BATCH_SIZE, len(records_to_process))})")
            
            for row in batch:
                pdf_name = row.get("pdf_name", "")
                page = row.get("page", "")
                page_text = row.get("page_text", "")
                page_text = get_akkadian_context_lines(page_text)
                batched_texts.append(f"--- Page {page} from {pdf_name} ---\n{page_text}")
            
            batched_pages = "\n----\n".join(batched_texts)
            prompt = prompt_template.format(batched_pages=batched_pages)
            
            if args.show_prompt:
                print(f"--- Prompt (Batch {i // BATCH_SIZE + 1}) ---")
                print(prompt)
                print("-" * 40)
            
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash-lite',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    )
                )
                
                try:
                    parsed = json.loads(response.text, strict=False)
                except json.JSONDecodeError as de:
                    # Fallback pattern for unescaped literal slashes generated by language models (e.g. \m or \a)
                    cleaned_text = response.text.replace("\\", "\\\\").replace("\\\\n", "\\n").replace('\\\\"', '\\"')
                    parsed = json.loads(cleaned_text, strict=False)
                
                # Output without per-document distinction as requested
                out_f.write(json.dumps(parsed, ensure_ascii=False) + "\n")
                out_f.flush()
                
            except Exception as e:
                print(f"Failed to process batch {i // BATCH_SIZE + 1}: {e}")
                # Print response text to log what failed
                if 'response' in locals() and hasattr(response, 'text'):
                    print(f"Bad response from LLM: {response.text}")
            
            time.sleep(2) # Rate limit delay
            
    print(f"Extraction complete! Results appended to {output_jsonl}")

if __name__ == "__main__":
    main()
