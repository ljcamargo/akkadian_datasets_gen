import csv
import json
import re
import os

def clean_word(word):
    """Remove trailing Roman numerals like ' I', ' II'."""
    return re.sub(r'\s+[IVX]+$', '', word).strip()

def clean_lemma(derived_from):
    """Extract lemmas from the derived_from column."""
    if not derived_from:
        return []
    # Remove leading "cf. " if present
    df = re.sub(r'^cf\.\s*', '', derived_from)
    # Split by spaces, commas, or semicolons
    parts = re.split(r'[,; ]+', df)
    lemmas = []
    for p in parts:
        cleaned = clean_word(p)
        if cleaned and cleaned != 'cf.':
            lemmas.append(cleaned)
    return lemmas

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
    # This regex looks for text in double quotes, and optionally grabs the next parentheses block.
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
                grammar.append(paren_str)
                
        meanings.append({
            "definition": meaning_str,
            "lemmas": global_lemmas, # Associate the global lemmas with this meaning
            "grammar": grammar,
            "references": references
        })
        
    if not found_any:
        special = True
        
    return meanings, special

def normalize_special(definition_text):
    """Additional checks for special flagging as requested by user."""
    if not definition_text: return False
    # If the text starts with a parenthesis and doesn't have quotes, or ends in ?)
    if definition_text.startswith('(') and '"' not in definition_text:
        return True
    if re.search(r'\(\w+[^)]+\)\?$', definition_text):
        return True
    return False

def main():
    os.makedirs('workspace/outputs/dictionary', exist_ok=True)
    input_file = 'eBL_Dictionary.csv'
    output_file = 'workspace/outputs/dictionary/dictionary_parsed.jsonl'
    
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        
        with open(output_file, 'w', encoding='utf-8') as out:
            for row in reader:
                orig_word = row.get('word', '').strip()
                orig_def = row.get('definition', '').strip()
                orig_derived = row.get('derived_from', '').strip()
                
                word = clean_word(orig_word)
                global_lemmas = clean_lemma(orig_derived)
                
                meanings, special = parse_definition(orig_def, global_lemmas)
                
                # Check additional heuristics
                if not special and normalize_special(orig_def):
                    special = True
                
                if not orig_def:
                    special = False
                    
                # To be conservative, if 'special' is false but there are numbers like "1." 
                # and we only captured part of the text, we might optionally flag it, but 
                # let's stick to the prompt's defined heuristics.
                
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

    print(f"Extraction complete. Output saved to {output_file}")

if __name__ == '__main__':
    main()
