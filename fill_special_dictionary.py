import json
import os
import time
from google import genai
from google.genai import types

INPUT_JSONL = "workspace/outputs/dictionary/dictionary_parsed.jsonl"
OUTPUT_JSONL = "workspace/outputs/dictionary/dictionary_parsed_filled.jsonl"
BATCH_SIZE = 10

def get_llm_prompt(batch):
    prompt = """You are an expert Assyriologist. Your task is to extract meanings, lemmas, and grammar from the following irregular Akkadian dictionary entries.
For each entry, you will receive its 'original_word', 'original_definition', and 'original_derived_from' text.
Return the parsed information strictly following this JSON schema for an array of entries.

Rules:
1. 'word' should be the cleaned word, without Roman numeral disambiguators.
2. 'meanings' should be a list of objects, each containing:
   - 'definition': The core English meaning (no quotes needed around it).
   - 'lemmas': A list of related lemmas found in derived_from or definition.
   - 'grammar': A list of grammar parses, usually found in parentheses.
   - 'references': A list of references, usually starting with 'cf.'
3. If the definition is empty or represents a name with no translated meaning, you can leave meanings empty.

Input Entries:
"""
    for entry in batch:
        prompt += f"\n---\n"
        prompt += f"Original Word: {entry.get('original_word')}\n"
        prompt += f"Original Definition: {entry.get('original_definition')}\n"
        prompt += f"Original Derived From: {entry.get('original_derived_from')}\n"
        
    return prompt

def process_batch(client, batch):
    prompt = get_llm_prompt(batch)
    
    # Using the standard gemini-2.5-flash model for extraction
    response = client.models.generate_content(
        model='gemini-2.5-flash',
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1,
        )
    )
    
    try:
        parsed_results = json.loads(response.text)
        return parsed_results
    except json.JSONDecodeError as e:
        print(f"Failed to parse JSON response: {e}")
        return []

def main():
    if not os.environ.get("GEMINI_API_KEY"):
        print("Please set the GEMINI_API_KEY environment variable.")
        return

    client = genai.Client()

    records = []
    input_file = OUTPUT_JSONL if os.path.exists(OUTPUT_JSONL) else INPUT_JSONL
    print(f"Reading records from {input_file}...")
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    special_indices = [i for i, r in enumerate(records) if r.get('special') is True]
    print(f"Found {len(special_indices)} special records remaining to process in this file.")

    for start_idx in range(0, len(special_indices), BATCH_SIZE):
        batch_indices = special_indices[start_idx:start_idx + BATCH_SIZE]
        batch_records = [records[i] for i in batch_indices]
        
        print(f"Processing batch {start_idx // BATCH_SIZE + 1} / {(len(special_indices) + BATCH_SIZE - 1) // BATCH_SIZE}...")
        
        results = process_batch(client, batch_records)
        
        # Match results back to records
        if results and isinstance(results, list) and len(results) == len(batch_records):
            for idx, res in zip(batch_indices, results):
                records[idx]['word'] = res.get('word', records[idx]['word'])
                records[idx]['meanings'] = res.get('meanings', [])
                records[idx]['special'] = False # Marked as resolved
                
            # Save progress incrementally after each successful batch
            with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
                for r in records:
                    f.write(json.dumps(r, ensure_ascii=False) + '\n')
            print(f"Batch saved to {OUTPUT_JSONL}")
        else:
            print(f"Warning: Batch returned {len(results) if isinstance(results, list) else 0} results, expected {len(batch_records)}.")
        
        # Rate limiting delay
        #time.sleep(2)
            
    print(f"LLM processing complete. Final records saved to {OUTPUT_JSONL}")
    print("You can now rename this file to dictionary_parsed.jsonl and re-run generate_dictionary_csvs.py")

if __name__ == '__main__':
    main()
