import csv
import json
import re
import os
from corpus_utils import *

GRAMMAR_MAP = {
    "person": {"1": "first", "2": "second", "3": "third", "1st": "first", "2nd": "second", "3rd": "third"},
    "gender": {"m.": "masculine", "f.": "feminine", "c.": "common"},
    "number": {"sg.": "singular", "pl.": "plural", "du.": "dual"},
    "case": {"acc.": "accusative", "dat.": "dative", "gen.": "genitive", "nom.": "nominative", "loc.": "locative", "term.": "terminative", "adv.": "adverbial"},
    "suffix": {"suff.": True, "suffix": True}
}

def clean_word(word):
    """Remove trailing Roman numerals like ' I', ' II'."""
    return re.sub(r'\s+[IVX]+$', '', word).strip()

def clean_lemma(derived_from):
    """Extract lemmas from the derived_from column."""
    if not derived_from:
        return []
    # Remove leading "cf. " if present
    df = re.sub(r'^cf\.\s*', '', derived_from)
    # Split by spaces, commas or semicolons
    parts = re.split(r'[,; ]+', df)
    lemmas = []
    
    roman_numeral_pattern = re.compile(r'^[IVX]+$')
    
    for p in parts:
        cleaned = clean_word(p)
        if not cleaned or cleaned == 'cf.':
            continue
        # Exclude if it has a dot (e.g., 'Sum.', 'Aram.') or is entirely a Roman numeral
        if '.' in cleaned or roman_numeral_pattern.match(cleaned):
            continue
        
        lemmas.append(cleaned)
    return lemmas

def validate_grammar(grammar_str):
    """
    Check if a grammar string is well-formed.
    Valid if every token contains a '.', ',', a digit, or has no letters (e.g., '=').
    """
    tokens = grammar_str.split()
    for token in tokens:
        if not (re.search(r'[.,\d]', token) or not re.search(r'[A-Za-z]', token)):
            return False
    return True

def parse_grammar_string(grammar_str):
    """Attempt to structure the grammar string by matching abbreviations."""
    result = {"parse": grammar_str}
    tokens = [t.strip(',') for t in grammar_str.split()]
    for token in tokens:
        token_lower = token.lower()
        for category, mapping in GRAMMAR_MAP.items():
            if token_lower in mapping:
                result[category] = mapping[token_lower]
                break
    return result

def parse_definition(definition_text, global_lemmas):
    """Parse definitions into structured meanings."""
    if not definition_text:
        return [], False

    meanings = []
    special = False
    
    # User specified: lack of double quotes usually implies uncertain/special meaning.
    if '"' not in definition_text:
        special = True
        return [], special

    # Extract all pattern: "meaning" (grammar/ref)
    matches = re.finditer(r'"(.*?)"(?:\s*\((.*?)\))?', definition_text)
    
    found_any = False
    for m in matches:
        found_any = True
        meaning_str = m.group(1).strip()
        paren_str = m.group(2)
        
        grammar = []
        references = []
        
        if paren_str:
            if paren_str.startswith('cf.'):
                references.append(paren_str)
            else:
                if validate_grammar(paren_str):
                    grammar.append(parse_grammar_string(paren_str))
                else:
                    special = True
        
        meaning_parts = [p.strip() for p in re.split(r'[,;]+', meaning_str) if p.strip()]
        
        for part in meaning_parts:
            meanings.append({
                "definition": part,
                "forms": list(global_lemmas),
                "grammar": list(grammar),
                "references": list(references)
            })
        
    if not found_any:
        special = True
        
    return meanings, special

def normalize_special(definition_text):
    """Additional checks for special flagging as requested by user."""
    if not definition_text: return False
    if definition_text.startswith('(') and '"' not in definition_text:
        return True
    if re.search(r'\(\w+[^)]+\)\?$', definition_text):
        return True
    return False

def extract_dictionary_patterns(input_file, output_jsonl):
    print("Extracting dictionary patterns...")
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        with open(output_jsonl, 'w', encoding='utf-8') as out:
            for row in reader:
                orig_word = row.get('word', '').strip()
                orig_def = row.get('definition', '').strip()
                orig_derived = row.get('derived_from', '').strip()
                
                word = clean_word(orig_word)
                global_lemmas = clean_lemma(orig_derived)
                
                meanings, special = parse_definition(orig_def, global_lemmas)
                
                if not special and normalize_special(orig_def):
                    special = True
                if not orig_def:
                    special = False
                    
                json_obj = {
                    "word": word,
                    "meanings": meanings,
                    "references": [],
                    "special": special,
                    "original_word": orig_word,
                    "original_definition": orig_def,
                    "original_derived_from": orig_derived
                }
                out.write(json.dumps(json_obj, ensure_ascii=False) + '\n')
    print(f"Extraction complete. Output saved to {output_jsonl}")

def generate_dictionary_csvs(input_jsonl, output_dir):
    print("Generating dictionary CSVs...")
    if not os.path.exists(input_jsonl):
        print(f"Error: {input_jsonl} not found.")
        return

    os.makedirs(output_dir, exist_ok=True)
    
    ft_files = {
        "lemma_finetune": open(f"{output_dir}/lemma_finetune.csv", "w", encoding="utf-8", newline=""),
        "grammar_finetune": open(f"{output_dir}/grammar_finetune.csv", "w", encoding="utf-8", newline=""),
        "meaning_finetune": open(f"{output_dir}/meaning_finetune.csv", "w", encoding="utf-8", newline=""),
        "translations_finetune": open(f"{output_dir}/translations_finetune.csv", "w", encoding="utf-8", newline="")
    }
    pt_files = {
        "dictionary_pretrain": open(f"{output_dir}/dictionary_pretrain.csv", "w", encoding="utf-8", newline=""),
        "rosetta_pretrain": open(f"{output_dir}/rosetta_pretrain.csv", "w", encoding="utf-8", newline="")
    }
    
    ft_writers = {k: csv.writer(v, **CSV_DIALECT_FINETUNE) for k, v in ft_files.items()}
    for w in ft_writers.values():
        w.writerow(["instruct", "query", "result"])
        
    pt_writers = {k: csv.writer(v, **CSV_DIALECT_PRETRAIN) for k, v in pt_files.items()}
    for w in pt_writers.values():
        w.writerow(["content"])

    db_path = f"{output_dir}/dictionary_dedup.db"
    dedup = Deduplicator(db_path)
    
    rosetta_buffer = []
    type_name = TYPE_EPIGRAPHIC
    
    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            record = json.loads(line)
            
            if record.get("special"):
                continue
                
            word = record.get("word")
            meanings = record.get("meanings", [])
            
            if not word or not meanings:
                continue

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

                if definition:
                    if dedup.is_unique("meaning_ft", word, definition):
                        ft_writers["meaning_finetune"].writerow([
                            linearize(PROMPT_MEANING_FINETUNE_WORD, is_finetune=True),
                            linearize(word, is_finetune=True),
                            linearize(definition, is_finetune=True)
                        ])
                        
                    cleaned_def = clean_translation(definition)
                    if dedup.is_unique("trans_ft", word, cleaned_def):
                        ft_writers["translations_finetune"].writerow([
                            linearize(PROMPT_TRANS_AKK_TO_ENG.replace("%type_name%", type_name), is_finetune=True),
                            linearize(word, is_finetune=True),
                            linearize(cleaned_def, is_finetune=True)
                        ])
                        ft_writers["translations_finetune"].writerow([
                            linearize(PROMPT_TRANS_ENG_TO_AKK.replace("%type_name%", type_name), is_finetune=True),
                            linearize(cleaned_def, is_finetune=True),
                            linearize(word, is_finetune=True)
                        ])

                if joined_lemmas:
                    if dedup.is_unique("lemma_ft", word, joined_lemmas):
                        ft_writers["lemma_finetune"].writerow([
                            linearize(PROMPT_LEMMA_FINETUNE.replace("%type_name%", type_name), is_finetune=True),
                            linearize(word, is_finetune=True),
                            linearize(joined_lemmas, is_finetune=True)
                        ])

                if joined_grammar:
                    if dedup.is_unique("grammar_ft", word, joined_grammar):
                        ft_writers["grammar_finetune"].writerow([
                            linearize(PROMPT_GRAMMAR_FINETUNE.replace("%type_name%", type_name), is_finetune=True),
                            linearize(word, is_finetune=True),
                            linearize(joined_grammar, is_finetune=True)
                        ])
                        
                if definition and dedup.is_unique("dict_pt", word, definition, joined_lemmas, joined_grammar):
                    #content = f"Akkadian Transliteration Dictionary Entry\nWORD: {word}\n"
                    content = f"WORD: {word}\n"
                    if joined_lemmas: content += f"LEMMA: {joined_lemmas}\n"
                    content += f"MEANING: {definition}\n"
                    if joined_grammar: content += f"GRAMMAR: {joined_grammar}\n"
                    pt_writers["dictionary_pretrain"].writerow([linearize(content)])
                    
                if definition:
                    rosetta_buffer.append((word, joined_lemmas, definition, joined_grammar))

            ROSETTA_CHUNK = 20
            while len(rosetta_buffer) >= ROSETTA_CHUNK:
                chunk = rosetta_buffer[:ROSETTA_CHUNK]
                rosetta_buffer = rosetta_buffer[ROSETTA_CHUNK:]
                title = "# Akkadian Dictionary Alignment Table"
                #table = f"{title}\n\n" + ROSETTA_HEADER_DICTIONARY
                table = ROSETTA_HEADER_DICTIONARY
                for w_col, l_col, d_col, g_col in chunk: 
                    table += f"| {w_col} | {l_col} | {d_col} | {g_col} |\n"
                pt_writers["rosetta_pretrain"].writerow([linearize(table)])

    if rosetta_buffer:
        title = "# Akkadian Dictionary Alignment Table"
        table = f"{title}\n\n" + ROSETTA_HEADER_DICTIONARY
        for w_col, l_col, d_col, g_col in rosetta_buffer: 
            table += f"| {w_col} | {l_col} | {d_col} | {g_col} |\n"
        pt_writers["rosetta_pretrain"].writerow([linearize(table)])

    for f in ft_files.values(): f.close()
    for f in pt_files.values(): f.close()
    dedup.close()
    
    print(f"Dictionary CSV generation complete. Output in {output_dir}")

def main():
    os.makedirs('workspace/outputs/dictionary', exist_ok=True)
    input_file = 'workspace/eBL_Dictionary.csv'
    jsonl_file = 'workspace/outputs/dictionary/dictionary_parsed.jsonl'
    output_dir = 'workspace/outputs/dictionary'
    
    extract_dictionary_patterns(input_file, jsonl_file)
    generate_dictionary_csvs(jsonl_file, output_dir)

if __name__ == "__main__":
    main()
