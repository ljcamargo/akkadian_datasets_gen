import csv
import json
import re
import os

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
                if validate_grammar(paren_str):
                    grammar.append(parse_grammar_string(paren_str))
                else:
                    # Invalid grammar sets the row to special but DOES NOT keep the invalid grammar
                    special = True
        
        # Split definition into multiple parts by comma or semicolon
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
    # If the text starts with a parenthesis and doesn't have quotes, or ends in ?)
    if definition_text.startswith('(') and '"' not in definition_text:
        return True
    if re.search(r'\(\w+[^)]+\)\?$', definition_text):
        return True
    return False

def main():
    os.makedirs('workspace/outputs/dictionary', exist_ok=True)
    input_file = 'workspace/eBL_Dictionary.csv'
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
