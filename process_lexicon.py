import csv
import json
import os
from collections import defaultdict
from corpus_utils import (
    CSV_DIALECT_FINETUNE, CSV_DIALECT_PRETRAIN, Deduplicator, 
    ROSETTA_TABLE_HEADER_LEXICON, TITLE_ROSETTA_PT, 
    PROMPT_LEMMA_FINETUNE, PROMPT_LEMMA_PRETRAIN_CONTENT, linearize
)

# Ensure output directory exists
output_dir = "workspace/outputs/lexicon"
os.makedirs(output_dir, exist_ok=True)

def compact_epig(form):
    return form.replace("-", "").replace(".", "")

def process_lexicon():
    input_file = "workspace/OA_Lexicon_eBL.csv"
    dedup = Deduplicator(os.path.join(output_dir, "dedup.db"))
    
    # Store data for derivatives
    # lexeme -> set of (norm, form)
    derivatives_map = defaultdict(set)
    
    # Files to generate
    f_lemma_finetune = open(os.path.join(output_dir, "lemma_finetune.csv"), "w", encoding="utf-8")
    writer_lemma_ft = csv.writer(f_lemma_finetune, **CSV_DIALECT_FINETUNE)
    writer_lemma_ft.writerow(["instruct", "query", "result"])

    f_rosetta_pretrain = open(os.path.join(output_dir, "rosetta_pretrain.csv"), "w", encoding="utf-8")
    writer_rosetta_pt = csv.writer(f_rosetta_pretrain, **CSV_DIALECT_PRETRAIN)
    writer_rosetta_pt.writerow(["content"])
    
    current_rosetta_rows = []
    
    def flush_rosetta():
        nonlocal current_rosetta_rows
        if not current_rosetta_rows: return
        table_str = f"{TITLE_ROSETTA_PT}\n\n{ROSETTA_TABLE_HEADER_LEXICON}"
        for row in current_rosetta_rows:
            table_str += f"| {' | '.join(row)} |\n"
        writer_rosetta_pt.writerow([linearize(table_str)])
        current_rosetta_rows = []
    
    # Read the CSV
    with open(input_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            w_type = row.get("type", "").strip()
            if w_type == "PN":
                w_type = "proper noun"
                
            form = row.get("form", "").strip()
            norm = row.get("norm", "").strip()
            lexeme = row.get("lexeme", "").strip()
            
            if not form or not lexeme:
                continue
                
            derivatives_map[lexeme].add((norm, form))
            comp = compact_epig(form)
            
            # Finetune Lemma
            if dedup.is_unique("lemma_ft_epig", form, lexeme):
                writer_lemma_ft.writerow([PROMPT_LEMMA_FINETUNE.replace("%type_name%", "epigraphic transliteration"), form, lexeme])
            
            if dedup.is_unique("lemma_ft_comp", comp, lexeme):
                writer_lemma_ft.writerow([PROMPT_LEMMA_FINETUNE.replace("%type_name%", "compact epigraphic transliteration"), comp, lexeme])
                
            if norm and dedup.is_unique("lemma_ft_norm", norm, lexeme):
                writer_lemma_ft.writerow([PROMPT_LEMMA_FINETUNE.replace("%type_name%", "akkadian orthography"), norm, lexeme])
            
            # Rosetta
            if dedup.is_unique("rosetta", form, comp, norm, lexeme, w_type):
                current_rosetta_rows.append([form, comp, norm, lexeme, w_type])
                if len(current_rosetta_rows) >= 20:
                    flush_rosetta()

    flush_rosetta()
    f_lemma_finetune.close()
    f_rosetta_pretrain.close()
    
    # Generate Pretrain Lemma file & json
    f_lemma_pretrain = open(os.path.join(output_dir, "lemma_pretrain.csv"), "w", encoding="utf-8")
    writer_lemma_pt = csv.writer(f_lemma_pretrain, **CSV_DIALECT_PRETRAIN)
    writer_lemma_pt.writerow(["content"])
    
    json_data = {}
    
    # Sort for deterministic output
    for lexeme in sorted(derivatives_map.keys()):
        forms_list = []
        json_forms = []
        for norm, form in sorted(list(derivatives_map[lexeme])):
            if norm:
                item_str = f"{norm} ({form})"
            else:
                item_str = form
                
            forms_list.append(item_str)
            json_forms.append(item_str)
            
        json_data[lexeme] = json_forms
        
        content_str = PROMPT_LEMMA_PRETRAIN_CONTENT.replace("%lexeme%", lexeme).replace("%derivatives%", ', '.join(forms_list))
        writer_lemma_pt.writerow([linearize(content_str)])
        
    f_lemma_pretrain.close()
    
    with open(os.path.join(output_dir, "lemma_derivatives.json"), "w", encoding="utf-8") as jf:
        json.dump(json_data, jf, ensure_ascii=False, indent=2)

    dedup.close()
    print("Lexicon processing complete.")

if __name__ == "__main__":
    process_lexicon()
