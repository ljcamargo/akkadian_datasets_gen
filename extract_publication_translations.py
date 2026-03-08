import os
import csv
import json
import argparse
import time
from google import genai
from google.genai import types

def remove_nul(file_iter):
    for line in file_iter:
        yield line.replace('\0', '')

PRICE_PER_M_INPUT = 0.10 # gemini-2.5-flash-lite
PRICE_PER_M_OUTPUT = 0.40 # gemini-2.5-flash-lite

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true", help="Estimate costs without querying LLM")
    parser.add_argument("--limit", type=int, default=0, help="Process only N entries")
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

    prompt_template = """You are an expert Assyriologist. Your task is to process the following OCR text from an academic publication and identify possible Akkadian text and its English (or German/French/etc.) translation pairs.

Input Text (Page {page} from {pdf}):
{page_text}

Extract the pairs into a valid JSON object matching this schema:
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

If there is no Akkadian text, or if you cannot confidently extract pairs or unpaired phrases, return empty lists. Do not add any markdown formatting, just provide the raw JSON without md entities or wrappers.
"""

    if args.dry_run:
        total_input_chars = 0
        total_output_tokens_estimated = 0
        
        for row in records_to_process:
            pdf_name = row.get("pdf_name", "")
            page = row.get("page", "")
            page_text = row.get("page_text", "")
            
            prompt = prompt_template.format(page=page, pdf=pdf_name, page_text=page_text)
            total_input_chars += len(prompt)
            # rough estimate: 200 output tokens per page
            total_output_tokens_estimated += 200
            
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
        for idx, row in enumerate(records_to_process):
            pdf_name = row.get("pdf_name", "")
            page = row.get("page", "")
            page_text = row.get("page_text", "")
            
            print(f"Processing record {idx+1}/{len(records_to_process)}: {pdf_name} p.{page}")
            
            prompt = prompt_template.format(page=page, pdf=pdf_name, page_text=page_text)
            
            try:
                response = client.models.generate_content(
                    model='gemini-2.5-flash',
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type="application/json",
                        temperature=0.1,
                    )
                )
                
                parsed = json.loads(response.text)
                
                res_obj = {
                    "pdf_name": pdf_name,
                    "page": page,
                    "extracted": parsed
                }
                out_f.write(json.dumps(res_obj, ensure_ascii=False) + "\n")
                out_f.flush()
                
            except Exception as e:
                print(f"Failed to process record {pdf_name} p.{page}: {e}")
            
            time.sleep(2) # Rate limit delay
            
    print(f"Extraction complete! Results appended to {output_jsonl}")

if __name__ == "__main__":
    main()
