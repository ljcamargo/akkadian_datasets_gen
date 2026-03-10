import csv
import os
import sqlite3
import hashlib
import re

# --- CONFIGURATION CONSTANTS ---
CSV_DIALECT_FINETUNE = {
    "quoting": csv.QUOTE_ALL,
    "lineterminator": "\n"
}

CSV_DIALECT_PRETRAIN = {
    "quoting": csv.QUOTE_ALL,
    "lineterminator": "\n"
}

# --- PROMPT TEMPLATES ---
HEADER_TEMPLATE = "# Cuneiform Tablet %text_id%\n## %title%\n\n"
TITLE_EPIGRAPHIC = "Akkadian Epigraphic Transliteration"
TITLE_COMPACT = " Akkadian Compact Transliteration"
TITLE_SPELLING = "Akkadian Normalized Transliteration"
TITLE_GRAMMAR_TEMPLATE = "Grammar Analysis (%base%)"

PROMPT_TEXT_PRETRAIN = "%type_name% of the cuneiform tablet %pub_info%"
PROMPT_GRAMMAR_FINETUNE = "Provide the grammar annotation of this %type_name%"
PROMPT_GRAMMAR_PRETRAIN_TITLE = "# Grammar Annotation of %type_name%"
PROMPT_MEANING_FINETUNE_TRANS = "Provide the lexical definition of this %type_name%"
PROMPT_MEANING_FINETUNE_WORD = "Provide the lexical definition of this Akkadian Normalized Transliteration"
PROMPT_LEMMA_FINETUNE = "Identify the lemma of this %type_name%"
PROMPT_LEMMA_PRETRAIN_CONTENT = "# Lexeme: %lexeme%\nDerivates: %derivatives%"
PROMPT_TRANS_AKK_TO_ENG = "Translate from %type_name% to english"
PROMPT_TRANS_ENG_TO_AKK = "Translate from english to %type_name%"
PROMPT_TRANS_ENG_TO_AKK_WORD = "Translate this English lexical definition into an Akkadian Normalized Transliteration"
PROMPT_TRANSFORM_EPIG_TO_SPELL = "Convert this Akkadian Transliteration from Epigraphic to Normalized"
PROMPT_TRANSFORM_SPELL_TO_EPIG = "Convert this Akkadian Transliteration from Normalized to Epigraphic"
PROMPT_TRANSFORM_COMPACT_TO_SPELL = "Convert this Akkadian Transliteration from Compact to Normalized"
PROMPT_TRANSFORM_SPELL_TO_COMPACT = "Convert this Akkadian Transliteration from Normalized to Compact"

TITLE_TRANS_PT_TO_ENG = "# Akkadian %type_name% Translation to English"
TITLE_TRANS_PT_FROM_ENG = "# English to Akkadian %type_name% Translation"
TITLE_ROSETTA_PT = "# Akkadian Transliteration Alignment"
ROSETTA_HEADER_DICTIONARY = "| Akkadian Transliteration | Lemma | Definition | Grammar |\n|---|---|---|---|\n"

PROMPT_GRAMMAR_PRETRAIN_CONTENT = "%title%\n%word%\n%grammar%"
PROMPT_GRAMMAR_ITEM_PREFIX = "- %variable%: %value%"

TRANS_TABLE_TEMPLATE = "| %h1% | %h2% |\n|---|---|\n"
ROSETTA_TABLE_HEADER = "| Epigraphic | Normalized | Lemma | Definition |\n|---|---|---|---|---|\n"
ROSETTA_TABLE_HEADER_LEXICON = "| Epigraphic | Normalized | Lemma | Type |\n|---|---|---|---|---|\n"


# --- HELPERS ---

def clean_translation(text):
    """Remove parentheses and extra spaces from English translations."""
    if not text: return ""
    return " ".join(text.replace('(', '').replace(')', '').split())

def replace_gaps(text):
    """Replace gaps (xxx, ..., …, [...]) with <gap> token."""
    if not text: return ""
    # Replace pre-existing known gap tokens
    text = re.sub(r'\[?\.\.\.\]?|\[?…\]?|…', '<gap>', text)
    text = re.sub(r'\b[xX]+\b', '<gap>', text)
    text = text.replace('[x]', '<gap>')
    text = text.replace('(break)', '<gap>')
    text = text.replace('(large break)', '<gap>')
    text = re.sub(r'\([nN\d]+ broken lines?\)', '<gap>', text)
    text = text.replace('<big_gap>', '<gap>')
    
    # Reduce multiple consecutive gaps and surrounding dashes/spaces to a single gap
    text = re.sub(r'(?:[- ]*<gap>[- ]*)+', '<gap>', text)
    return text

def standardize_orthography(text):
    if not text: return ""
    text = text.replace('(d)', '{d}').replace('(ki)', '{ki}').replace('(TÚG)', 'TÚG')
    text = re.sub(r'(\d+\.\d{4})\d+', r'\1', text)
    return text

def clean_finetune_lints(text):
    if not text: return ""
    text = re.sub(r'[\[\]˹˺]', '', text)
    text = text.replace('<gap>', '___GAP___')
    text = re.sub(r'[<>]', '', text)
    text = text.replace('___GAP___', '<gap>')
    return text

def linearize(text, is_finetune=False):
    if not text: return ""
    text = standardize_orthography(text)
    if is_finetune:
        text = clean_finetune_lints(text)
    return text.replace("\n", "\\n")

def remove_nul(file_iter):
    for line in file_iter:
        yield line.replace('\0', '')

def get_akkadian_context_lines(page_text):
    separator = '\\n' if '\\n' in page_text and '\n' not in page_text else '\n'
    lines = page_text.split(separator)
    
    # Syllable pattern: matches non-numbers separated by -, e.g. a-na, ma-nu-um
    # We use a pattern that strictly accepts letters and specific transliteration chars, but no digits 0-9.
    syl_pattern = re.compile(r'(?<!\d)[a-zA-ZšḫṣŠ₄ṭ₅Ḫ₁₀₈₆₃Ṣ₇]+(?:-[a-zA-ZšḫṣŠ₄ṭ₅Ḫ₁₀₈₆₃Ṣ₇]+)+(?!\d)')
    special_char_pattern = re.compile(r'[šḫṣŠ₄ṭ₅Ḫ₁₀₈₆₃Ṣ₇]')
    keywords = ["a-na", "i-na", "ša", "šu", "i-ma", "u-ma", "um-", "-tim", "KÙ", "DUMU", "GIN", "IGI", "URDU", "KIŠIB"]
    
    akk_line_indices = []
    for i, line in enumerate(lines):
        # Improved Syllable Pattern at least two times
        if len(syl_pattern.findall(line)) >= 2:
            # AND ( special chars >= 2 OR keyword >= 1 )
            special_count = len(special_char_pattern.findall(line))
            has_keyword = any(kw in line for kw in keywords)
            
            if special_count >= 2 or has_keyword:
                akk_line_indices.append(i)
            
    if not akk_line_indices:
        #print(">>>>>>>>>>>>>>> No Akkadian text matched", page_text[:100])
        return ""
        
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

def get_markdown_title(type_name, is_grammar=False):
    base = ""
    if "compact" in type_name: base = TITLE_COMPACT
    elif "spelling" in type_name: base = TITLE_SPELLING
    else: base = TITLE_EPIGRAPHIC
    
    if is_grammar:
        return TITLE_GRAMMAR_TEMPLATE.replace("%base%", base)
    return base

def get_text_id(text_meta):
    """Returns a consistent row identifier: name > publicationPrefix/Number > uuid."""
    name = text_meta.get("name")
    if name:
        return name
    
    p = text_meta.get("publicationPrefix")
    n = text_meta.get("publicationNumber")
    if p or n:
        return f"{p or ''} {n or ''}".strip()
        
    return text_meta.get("uuid", "N/A")

def get_markdown_header(text_meta, type_name, is_grammar=False):
    text_id = get_text_id(text_meta)
    title = get_markdown_title(type_name, is_grammar)
    return HEADER_TEMPLATE.replace("%text_id%", text_id).replace("%title%", title)

def get_grammar_result(group):
    results = []
    for u in group:
        parse_list = u.get("parseInfo")
        if not parse_list: continue
        for item in parse_list:
            var = item.get("variableName")
            val = item.get("value")
            if var and val:
                results.append(PROMPT_GRAMMAR_ITEM_PREFIX.replace("%variable%", var).replace("%value%", val))
    return "\n".join(results)


class Deduplicator:
    def __init__(self, db_path):
        if os.path.exists(db_path):
            os.remove(db_path)
        self.conn = sqlite3.connect(db_path)
        self.cursor = self.conn.cursor()
        # Create table to store hashes of seen items
        self.cursor.execute("CREATE TABLE seen (task TEXT, item_hash TEXT, PRIMARY KEY (task, item_hash))")
        # Speed optimizations
        self.cursor.execute("PRAGMA synchronous = OFF")
        self.cursor.execute("PRAGMA journal_mode = MEMORY")
        self.conn.commit()

    def is_unique(self, task, *args):
        """Returns True if the combination of task and args is new, False if seen."""
        # Join args with a separator to form a content string
        content = "|".join([str(a) for a in args])
        item_hash = hashlib.md5(content.encode("utf-8")).hexdigest()
        
        try:
            self.cursor.execute("INSERT INTO seen VALUES (?, ?)", (task, item_hash))
            return True
        except sqlite3.IntegrityError:
            return False

    def close(self):
        self.conn.close()
