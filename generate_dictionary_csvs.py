import json
import csv
import os
from corpus_utils import *

INPUT_JSONL = "workspace/outputs/dictionary/dictionary_parsed.jsonl"
OUTPUT_DIR = "workspace/outputs/dictionary"

def get_dictionary_rosetta_header():
    return "| Akkadian Word | Lemma | Definition | Grammar |\n|---|---|---|---|\n"

def main():
    if not os.path.exists(INPUT_JSONL):
        print(f"Error: {INPUT_JSONL} not found.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Initialize writers
    ft_files = {
        "lemma_finetune": open(f"{OUTPUT_DIR}/lemma_finetune.csv", "w", encoding="utf-8", newline=""),
        "grammar_finetune": open(f"{OUTPUT_DIR}/grammar_finetune.csv", "w", encoding="utf-8", newline=""),
        "meaning_finetune": open(f"{OUTPUT_DIR}/meaning_finetune.csv", "w", encoding="utf-8", newline=""),
        "translations_finetune": open(f"{OUTPUT_DIR}/translations_finetune.csv", "w", encoding="utf-8", newline="")
    }
    pt_files = {
        "dictionary_pretrain": open(f"{OUTPUT_DIR}/dictionary_pretrain.csv", "w", encoding="utf-8", newline=""),
        "rosetta_pretrain": open(f"{OUTPUT_DIR}/rosetta_pretrain.csv", "w", encoding="utf-8", newline="")
    }
    
    ft_writers = {k: csv.writer(v, **CSV_DIALECT_FINETUNE) for k, v in ft_files.items()}
    for w in ft_writers.values():
        w.writerow(["instruct", "query", "result"])
        
    pt_writers = {k: csv.writer(v, **CSV_DIALECT_PRETRAIN) for k, v in pt_files.items()}
    for w in pt_writers.values():
        w.writerow(["content"])

    db_path = f"{OUTPUT_DIR}/dictionary_dedup.db"
    dedup = Deduplicator(db_path)
    
    rosetta_buffer = []

    type_name = "word"
    
    with open(INPUT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            if record.get("special"):
                continue
                
            word = record.get("word")
            meanings = record.get("meanings", [])
            
            if not word or not meanings:
                continue

            # Process each meaning block
            for meaning_obj in meanings:
                definition = meaning_obj.get("definition")
                lemmas = meaning_obj.get("forms", [])
                grammar = meaning_obj.get("grammar", [])
                
                joined_lemmas = ", ".join(lemmas) if lemmas else ""
                
                grammar_strings = []
                for g in grammar:
                    if isinstance(g, dict):
                        grammar_strings.append(g.get("parse", ""))
                    else:
                        grammar_strings.append(str(g))
                        
                joined_grammar = ", ".join(grammar_strings) if grammar_strings else ""

                # --- FINETUNING ---
                if definition:
                    if dedup.is_unique("meaning_ft", word, definition):
                        ft_writers["meaning_finetune"].writerow([
                            linearize(PROMPT_MEANING_FINETUNE_WORD),
                            linearize(word),
                            linearize(definition)
                        ])
                        
                    cleaned_def = clean_translation(definition)
                    if dedup.is_unique("trans_ft", word, cleaned_def):
                        ft_writers["translations_finetune"].writerow([
                            linearize(PROMPT_TRANS_AKK_TO_ENG.replace("%type_name%", type_name)),
                            linearize(word),
                            linearize(cleaned_def)
                        ])
                        ft_writers["translations_finetune"].writerow([
                            linearize(PROMPT_TRANS_ENG_TO_AKK_WORD),
                            linearize(cleaned_def),
                            linearize(word)
                        ])

                if joined_lemmas:
                    if dedup.is_unique("lemma_ft", word, joined_lemmas):
                        ft_writers["lemma_finetune"].writerow([
                            linearize(PROMPT_LEMMA_FINETUNE.replace("%type_name%", type_name)),
                            linearize(word),
                            linearize(joined_lemmas)
                        ])

                if joined_grammar:
                    if dedup.is_unique("grammar_ft", word, joined_grammar):
                        ft_writers["grammar_finetune"].writerow([
                            linearize(PROMPT_GRAMMAR_FINETUNE.replace("%type_name%", type_name)),
                            linearize(word),
                            linearize(joined_grammar)
                        ])
                        
                # --- PRETRAINING ---
                # Linearized Pretrain
                if definition and dedup.is_unique("dict_pt", word, definition, joined_lemmas, joined_grammar):
                    content = f"# Dictionary Entry\nWord: {word}\n"
                    if joined_lemmas: content += f"Lemma: {joined_lemmas}\n"
                    content += f"Definition: {definition}\n"
                    if joined_grammar: content += f"Grammar: {joined_grammar}\n"
                    
                    pt_writers["dictionary_pretrain"].writerow([linearize(content)])
                    
                # Buffer for Rosetta Pretrain
                if definition:
                    rosetta_buffer.append((word, joined_lemmas, definition, joined_grammar))

            # Flush Rosetta buffer
            ROSETTA_CHUNK = 20
            while len(rosetta_buffer) >= ROSETTA_CHUNK:
                chunk = rosetta_buffer[:ROSETTA_CHUNK]
                rosetta_buffer = rosetta_buffer[ROSETTA_CHUNK:]
                title = "# Akkadian Dictionary Alignment Table"
                table = f"{title}\n\n" + get_dictionary_rosetta_header()
                for w_col, l_col, d_col, g_col in chunk: 
                    table += f"| {w_col} | {l_col} | {d_col} | {g_col} |\n"
                pt_writers["rosetta_pretrain"].writerow([linearize(table)])

    # Flush remaining Rosetta
    if rosetta_buffer:
        title = "# Akkadian Dictionary Alignment Table"
        table = f"{title}\n\n" + get_dictionary_rosetta_header()
        for w_col, l_col, d_col, g_col in rosetta_buffer: 
            table += f"| {w_col} | {l_col} | {d_col} | {g_col} |\n"
        pt_writers["rosetta_pretrain"].writerow([linearize(table)])

    for f in ft_files.values(): f.close()
    for f in pt_files.values(): f.close()
    dedup.close()
    
    print(f"Dictionary CSV generation complete. Output in {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
