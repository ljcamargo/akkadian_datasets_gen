import json
import csv
import os
import random
import re
import yaml
import sqlite3
import hashlib
import unicodedata
from collections import defaultdict
from pathlib import Path

# --- CONFIGURATION CONSTANTS (from corpus_utils) ---
CSV_DIALECT_FINETUNE = {"quoting": csv.QUOTE_ALL, "lineterminator": "\n"}
CSV_DIALECT_PRETRAIN = {"quoting": csv.QUOTE_ALL, "lineterminator": "\n"}
TYPE_EPIGRAPHIC = "akkadian"
PROMPT_TRANS_AKK_TO_ENG = "Translate from %type_name% to english"
PROMPT_TRANS_ENG_TO_AKK = "Translate from english to %type_name%"
ROSETTA_HEADER_DICTIONARY = "| Akkadian | Lemma | Definition | Grammar |\n|---|---|---|---|\n"
ROSETTA_TABLE_HEADER = "| Epigraphic | Normalized | Lemma | Definition |\n|---|---|---|---|---|\n"
HEADER_TEMPLATE = "Cuneiform Tablet %text_id%\n%title%\n\n"
TITLE_EPIGRAPHIC = "Akkadian Transliteration"

# --- CONFIGURATION CONSTANTS (Local) ---
INPUT_DEFAULT = "workspace/oare_epigraphies.jsonl"
OUTPUT_DIR_PUBLISHED = "workspace/outputs/published_texts"
PRETRAIN_CHUNK_SIZE = 40
ROSETTA_CHUNK_SIZE = 20
ENG_TO_AKK_DROP_RATE = 0.75

GRAMMAR_MAP = {
    "person": {"1": "first", "2": "second", "3": "third", "1st": "first", "2nd": "second", "3rd": "third"},
    "gender": {"m.": "masculine", "f.": "feminine", "c.": "common"},
    "number": {"sg.": "singular", "pl.": "plural", "du.": "dual"},
    "case": {"acc.": "accusative", "dat.": "dative", "gen.": "genitive", "nom.": "nominative", "loc.": "locative", "term.": "terminative", "adv.": "adverbial"},
    "suffix": {"suff.": True, "suffix": True}
}

INPUT_DICT = 'workspace/eBL_Dictionary.csv'
OUTPUT_DIR_DICT = 'workspace/outputs/dictionary'
JSONL_DICT = f"{OUTPUT_DIR_DICT}/dictionary_parsed.jsonl"

INPUT_PUB = "workspace/publications.csv"
OUTPUT_DIR_PUB = "workspace/outputs/publications"

INPUT_TRAIN = "workspace/train.csv"
OUTPUT_DIR_TRAIN = "workspace/outputs/train"

LEMMA_DERIVATIVES_JSON = "workspace/outputs/lexicon/lemma_derivatives.json"

DICT_PATH_MERGE = Path(JSONL_DICT)
PUBTEXTS_PATH_MERGE = Path(f"{OUTPUT_DIR_PUBLISHED}/dictionary_parsed.jsonl")
OUTPUT_MERGED_DICT = Path("workspace/outputs/final_dictionary.json")

PRETRAIN_FILES = [
    f"{OUTPUT_DIR_PUBLISHED}/translations_pretrain.csv",
    f"{OUTPUT_DIR_PUBLISHED}/texts_pretrain.csv",
    f"{OUTPUT_DIR_PUB}/publications_pretrain.csv",
    f"{OUTPUT_DIR_TRAIN}/translations_pretrain.csv"
]
FINETUNE_FILES = [[
    f"{OUTPUT_DIR_PUBLISHED}/translations_finetune.csv",
    f"{OUTPUT_DIR_DICT}/translations_finetune.csv",
    f"{OUTPUT_DIR_TRAIN}/translations_finetune.csv",
    f"{OUTPUT_DIR_TRAIN}/reasoned_translations_finetune.csv"
]]

# --- HELPERS (from corpus_utils) ---
def clean_translation(text):
    if not text: return ""
    return " ".join(text.replace('(', '').replace(')', '').split())

def replace_gaps(text):
    if not text: return ""
    text = re.sub(r'\[?\.\.\.\]?|\[?…\]?|…', '<gap>', text)
    text = re.sub(r'\b[xX]+\b', '<gap>', text)
    text = text.replace('[x]', '<gap>').replace('(break)', '<gap>').replace('(large break)', '<gap>').replace('<big_gap>', '<gap>')
    text = re.sub(r'\([nN\d]+ broken lines?\)', '<gap>', text)
    return re.sub(r'(?:[- ]*<gap>[- ]*)+', '<gap>', text)

def standardize_orthography(text):
    if not text: return ""
    text = text.replace('(d)', '{d}').replace('(ki)', '{ki}').replace('(TÚG)', 'TÚG')
    text = re.sub(r'(\d+\.\d{4})\d+', r'\1', text).translate(str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')).translate(str.maketrans('⁰¹²³⁴⁵⁶⁷⁸⁹', '0123456789'))
    return unicodedata.normalize('NFC', text)

def linearize(text, is_finetune=True):
    if not text: return ""
    text = standardize_orthography(text)
    text = replace_gaps(text)
    # clean_finetune_lints logic
    text = re.sub(r'[\[\]˹˺]', '', text).replace('<gap>', '___GAP___')
    text = re.sub(r'[<>]', '', text).replace('___GAP___', '<gap>')
    return text.replace("\n", "\\n")

def remove_nul(file_iter):
    for line in file_iter: yield line.replace('\0', '')

def get_akkadian_context_lines(page_text, lines_margin=0):
    sep = '\\n' if '\\n' in page_text and '\n' not in page_text else '\n'
    ls = page_text.split(sep)
    syl_p = re.compile(r'(?<!\d)[a-zA-ZšḫṣŠ₄ṭ₅Ḫ₁₀₈₆₃Ṣ₇]+(?:-[a-zA-ZšḫṣŠ₄ṭ₅Ḫ₁₀₈₆₃Ṣ₇]+)+(?!\d)')
    special_p = re.compile(r'[šḫṣŠ₄ṭ₅Ḫ₁₀₈₆₃Ṣ₇]')
    kws = ["a-na", "i-na", "ša", "šu", "i-ma", "u-ma", "um-", "-tim", "KÙ", "DUMU", "GIN", "IGI", "URDU", "KIŠIB"]
    idxs = [i for i, l in enumerate(ls) if len(syl_p.findall(l)) >= 3 and (len(special_p.findall(l)) >= 3 or any(k in l for k in kws))]
    if not idxs: return ""
    inc = sorted({j for i in idxs for j in range(max(0, i - lines_margin), min(len(ls), i + lines_margin + 1))})
    return '\n'.join([standardize_orthography(replace_gaps(re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F-\x9F]', '', ls[idx]))) for idx in inc])

class Deduplicator:
    def __init__(self, db_path):
        if os.path.exists(db_path): os.remove(db_path)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute("CREATE TABLE seen (task TEXT, item_hash TEXT, PRIMARY KEY (task, item_hash))")
        self.conn.execute("PRAGMA synchronous = OFF"); self.conn.execute("PRAGMA journal_mode = MEMORY")
    def is_unique(self, task, *args):
        h = hashlib.md5("|".join([str(a) for a in args]).encode("utf-8")).hexdigest()
        try:
            self.conn.execute("INSERT INTO seen VALUES (?, ?)", (task, h))
            return True
        except sqlite3.IntegrityError: return False
    def close(self): self.conn.close()

# --- HELPER FUNCTIONS (Local) ---
def group_units_by_spelling(units, key):
    if not units: return []
    groups, current_group, current_uuid = [], [], None
    for u in units:
        val, uuid = u.get(key), u.get('spellingUuid')
        if val is None or uuid is None:
            if current_group:
                groups.append(current_group)
                current_group, current_uuid = [], None
            continue
        if uuid == current_uuid and current_uuid is not None:
            current_group.append(val)
        else:
            if current_group: groups.append(current_group)
            current_group, current_uuid = [val], uuid
    if current_group: groups.append(current_group)
    return groups

def format_epigraphy(units, compact=False):
    structure = defaultdict(lambda: defaultdict(list))
    for u in units:
        if u.get('side') and u.get('line') is not None:
            structure[u['side']][u['line']].append(u)
    lines_output, sides_processed = [], []
    sorted_sides = sorted(structure.keys(), key=lambda x: 0 if x == "obv." else 1 if x == "rev." else 2)
    for side in sorted_sides:
        if side == "rev." and ("obv." in sides_processed or any("obv" in str(s).lower() for s in sides_processed)):
            lines_output.append("\n")
        sides_processed.append(side)
        for line_num in sorted(structure[side].keys()):
            groups = group_units_by_spelling(structure[side][line_num], "epigReading")
            formatted = [("" if compact else "-").join(g) for g in groups]
            if formatted: lines_output.append(" ".join(formatted))
    return standardize_orthography(" ".join(lines_output))

# --- CORE PROCESSING FUNCTIONS ---
def process_published():
    random.seed(42)
    print("Starting process_published...")
    os.makedirs(OUTPUT_DIR_PUBLISHED, exist_ok=True)
    dedup = Deduplicator(f"{OUTPUT_DIR_PUBLISHED}/dedup.db")
    
    ft = open(f"{OUTPUT_DIR_PUBLISHED}/translations_finetune.csv", "w", encoding="utf-8")
    pt1 = open(f"{OUTPUT_DIR_PUBLISHED}/texts_pretrain.csv", "w", encoding="utf-8")
    pt2 = open(f"{OUTPUT_DIR_PUBLISHED}/translations_pretrain.csv", "w", encoding="utf-8")
    
    w_ft = csv.writer(ft, **CSV_DIALECT_FINETUNE)
    w_pt1 = csv.writer(pt1, **CSV_DIALECT_PRETRAIN)
    w_pt2 = csv.writer(pt2, **CSV_DIALECT_PRETRAIN)
    
    w_ft.writerow(["instruct", "query", "result"])
    w_pt1.writerow(["content"])
    w_pt2.writerow(["content"])
    
    trans_p_buffers = defaultdict(list)
    out_dict = open(f"{OUTPUT_DIR_PUBLISHED}/dictionary_parsed.jsonl", "w", encoding="utf-8")

    with open(INPUT_DEFAULT, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if line_idx % 10000 == 0:
                print(f"  process_published: processed {line_idx} lines...")
            try: data = json.loads(line)
            except: continue
            
            units = data.get("units", [])
            content = format_epigraphy(units, False)
            if content and dedup.is_unique("texts_pretrain", TYPE_EPIGRAPHIC, content):
                w_pt1.writerow([linearize(content)])

            valid_units = [u for u in units if u.get("side") and u.get("line") is not None]
            word_groups = []
            if valid_units:
                cur_group, cur_uuid = [valid_units[0]], valid_units[0].get("spellingUuid")
                for u in valid_units[1:]:
                    uuid = u.get("spellingUuid")
                    if uuid == cur_uuid and uuid is not None: cur_group.append(u)
                    else:
                        word_groups.append(cur_group)
                        cur_group, cur_uuid = [u], uuid
                word_groups.append(cur_group)

            for group in word_groups:
                g_epig = [u["epigReading"] for u in group if u.get("epigReading")]
                g_form = [u["form"] for u in group if u.get("form")]
                u_word = next((u["word"] for u in group if u.get("word")), None)
                u_trans = next((u["translation"] for u in group if u.get("translation")), None)
                
                q_e, q_c, q_f = "-".join(g_epig), "".join(g_epig), " ".join(g_form)
                
                if u_trans:
                    for part in [p.strip() for p in u_trans.split(",") if p.strip()]:
                        if q_e:
                            actual_part = u_word if (part == "PN" or part == "GN") else part
                            cleaned_part = clean_translation(actual_part)
                            if dedup.is_unique("trans", TYPE_EPIGRAPHIC, q_e, cleaned_part):
                                w_ft.writerow([linearize(PROMPT_TRANS_AKK_TO_ENG.replace("%type_name%", TYPE_EPIGRAPHIC), is_finetune=True), linearize(q_e, is_finetune=True), linearize(cleaned_part, is_finetune=True)])
                                if random.random() > ENG_TO_AKK_DROP_RATE:
                                    w_ft.writerow([linearize(PROMPT_TRANS_ENG_TO_AKK.replace("%type_name%", TYPE_EPIGRAPHIC), is_finetune=True), linearize(cleaned_part, is_finetune=True), linearize(q_e, is_finetune=True)])
                                trans_p_buffers[(TYPE_EPIGRAPHIC, "to_eng")].append((q_e, cleaned_part))

                dict_word, dict_lemmas = u_word if u_word else q_f, [u_word] if u_word else []
                dict_def = clean_translation(u_word if u_trans == "PN" else u_trans) if u_trans else ""
                grammar_list = []
                for u in group:
                    pi = u.get("parseInfo")
                    if pi:
                        unit_grammar = {"parse": ""}
                        for item in pi:
                            var_n, var_v = item.get("variableName"), item.get("value")
                            if var_n and var_v:
                                kn, vn = var_n.lower().replace(" ", "_"), var_v.lower()
                                if kn == "grammatical_number": kn = "number"
                                elif kn == "morphological_form": kn = "form"
                                elif kn == "primary_classification": kn = "classification"
                                elif kn == "part_of_speech": kn = "pos"
                                if vn == "first person": vn = "first"
                                elif vn == "second person": vn = "second"
                                elif vn == "third person": vn = "third"
                                unit_grammar[kn] = vn
                        if len(unit_grammar) > 1: grammar_list.append(unit_grammar)
                if dict_word and (dict_def or grammar_list):
                    if dedup.is_unique("dict_jsonl", dict_word, dict_def, q_f, q_e, len(grammar_list)):
                        out_dict.write(json.dumps({
                            "word": dict_word,
                            "meanings": [{"definition": dict_def, "forms": dict_lemmas, "grammar": grammar_list, "references": []}],
                            "special": False, "epigraphic": q_e, "compact": q_c, "orthography": q_f
                        }, ensure_ascii=False) + "\n")

            for key in list(trans_p_buffers.keys()):
                buf = trans_p_buffers[key]
                while len(buf) >= PRETRAIN_CHUNK_SIZE:
                    chunk = buf[:PRETRAIN_CHUNK_SIZE]
                    buf = buf[PRETRAIN_CHUNK_SIZE:]
                    trans_p_buffers[key] = buf
                    for c1, c2 in chunk:
                        w_pt2.writerow([linearize(f"{c1} = {c2}")])

    dedup.close()
    if os.path.exists(f"{OUTPUT_DIR_PUBLISHED}/dedup.db"): os.remove(f"{OUTPUT_DIR_PUBLISHED}/dedup.db")
    for file_obj in [ft, pt1, pt2, out_dict]: file_obj.close()
    print("Finished process_published.")


def clean_word(word): return re.sub(r'\s+[IVX]+$', '', word).strip()

def clean_lemma(derived_from):
    if not derived_from: return []
    parts, lemmas, pattern = re.split(r'[,; ]+', re.sub(r'^cf\.\s*', '', derived_from)), [], re.compile(r'^[IVX]+$')
    for p in parts:
        cleaned = clean_word(p)
        if cleaned and cleaned != 'cf.' and '.' not in cleaned and not pattern.match(cleaned):
            lemmas.append(cleaned)
    return lemmas

def validate_grammar(grammar_str): return all(re.search(r'[.,\d]', t) or not re.search(r'[A-Za-z]', t) for t in grammar_str.split())

def parse_grammar_string(grammar_str):
    result = {"parse": grammar_str}
    for t in [t.strip(',') for t in grammar_str.split()]:
        for cat, mapping in GRAMMAR_MAP.items():
            if t.lower() in mapping:
                result[cat] = mapping[t.lower()]
                break
    return result

def parse_definition(definition_text, global_lemmas):
    if not definition_text: return [], False
    if '"' not in definition_text: return [], True
    meanings, found, special = [], False, False
    for m in re.finditer(r'"(.*?)"(?:\s*\((.*?)\))?', definition_text):
        found = True
        grammar, references, m_str, p_str = [], [], m.group(1).strip(), m.group(2)
        if p_str:
            if p_str.startswith('cf.'): references.append(p_str)
            elif validate_grammar(p_str): grammar.append(parse_grammar_string(p_str))
            else: special = True
        for part in [p.strip() for p in re.split(r'[,;]+', m_str) if p.strip()]:
            meanings.append({"definition": part, "forms": list(global_lemmas), "grammar": list(grammar), "references": list(references)})
    if not found: special = True
    return meanings, special

def normalize_special(definition_text):
    if not definition_text: return False
    return (definition_text.startswith('(') and '"' not in definition_text) or bool(re.search(r'\(\w+[^)]+\)\?$', definition_text))

def process_dictionaries():
    random.seed(42)
    print("Starting process_dictionaries...")
    os.makedirs(OUTPUT_DIR_DICT, exist_ok=True)
    with open(INPUT_DICT, 'r', encoding='utf-8') as f, open(JSONL_DICT, 'w', encoding='utf-8') as out:
        for row in csv.DictReader(f):
            ow, od, odf = row.get('word', '').strip(), row.get('definition', '').strip(), row.get('derived_from', '').strip()
            meanings, spec = parse_definition(od, clean_lemma(odf))
            if not spec and normalize_special(od): spec = True
            out.write(json.dumps({"word": clean_word(ow), "meanings": meanings, "references": [], "special": spec if od else False, "original_word": ow, "original_definition": od, "original_derived_from": odf}, ensure_ascii=False) + '\n')
            
    ft = open(f"{OUTPUT_DIR_DICT}/translations_finetune.csv", "w", encoding="utf-8", newline="")
    w_ft = csv.writer(ft, **CSV_DIALECT_FINETUNE)
    w_ft.writerow(["instruct", "query", "result"])
    dedup = Deduplicator(f"{OUTPUT_DIR_DICT}/dictionary_dedup.db")
    
    with open(JSONL_DICT, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            rec = json.loads(line)
            word, meanings = rec.get("word"), rec.get("meanings", [])
            if rec.get("special") or not word or not meanings: continue
            for m in meanings:
                d = m.get("definition")
                if d:
                    cd = clean_translation(d)
                    if dedup.is_unique("trans_ft", word, cd):
                        w_ft.writerow([linearize(PROMPT_TRANS_AKK_TO_ENG.replace("%type_name%", TYPE_EPIGRAPHIC), is_finetune=True), linearize(word, is_finetune=True), linearize(cd, is_finetune=True)])
                        if random.random() > ENG_TO_AKK_DROP_RATE:
                            w_ft.writerow([linearize(PROMPT_TRANS_ENG_TO_AKK.replace("%type_name%", TYPE_EPIGRAPHIC), is_finetune=True), linearize(cd, is_finetune=True), linearize(word, is_finetune=True)])
    ft.close(); dedup.close()
    print("Finished process_dictionaries.")


def process_publications():
    print("Starting process_publications...")
    os.makedirs(OUTPUT_DIR_PUB, exist_ok=True)
    with open(INPUT_PUB, "r", encoding="utf-8-sig") as f, open(f"{OUTPUT_DIR_PUB}/publications_pretrain.csv", "w", encoding="utf-8") as out_f:
        writer = csv.writer(out_f, **CSV_DIALECT_PRETRAIN)
        writer.writerow(["content"])
        for row in csv.DictReader(remove_nul(f)):
            pt = row.get("page_text", "").strip()
            if not row.get("pdf_name", "").strip() and not pt: continue
            fpt = get_akkadian_context_lines(pt)
            if fpt: writer.writerow([linearize(fpt)])
    print("Finished process_publications.")


def process_train():
    random.seed(42)
    print("Starting process_train...")
    os.makedirs(OUTPUT_DIR_TRAIN, exist_ok=True)
    dedup = Deduplicator(f"{OUTPUT_DIR_TRAIN}/dedup.db")
    ft = open(f"{OUTPUT_DIR_TRAIN}/translations_finetune.csv", "w", encoding="utf-8")
    pt = open(f"{OUTPUT_DIR_TRAIN}/translations_pretrain.csv", "w", encoding="utf-8")
    w_ft = csv.writer(ft, **CSV_DIALECT_FINETUNE)
    w_pt = csv.writer(pt, **CSV_DIALECT_PRETRAIN)
    w_ft.writerow(["instruct", "query", "result"])
    w_pt.writerow(["content"])
    
    with open(INPUT_TRAIN, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            oid, translit, translat = row.get("oare_id", "").strip(), replace_gaps(row.get("transliteration", "").strip()), replace_gaps(row.get("translation", "").strip())
            if not translit or not translat or not oid: continue
            w_pt.writerow([linearize(f"{translit} = {translat}")])
            ia = PROMPT_TRANS_AKK_TO_ENG.replace("%type_name%", TYPE_EPIGRAPHIC)
            if dedup.is_unique("trans_ft_akk2eng", ia, translit, translat): w_ft.writerow([linearize(ia, is_finetune=True), linearize(translit, is_finetune=True), linearize(translat, is_finetune=True)])
            ie = PROMPT_TRANS_ENG_TO_AKK.replace("%type_name%", TYPE_EPIGRAPHIC)
            if dedup.is_unique("trans_ft_eng2akk", ie, translat, translit) and random.random() > ENG_TO_AKK_DROP_RATE: w_ft.writerow([linearize(ie, is_finetune=True), linearize(translat, is_finetune=True), linearize(translit, is_finetune=True)])
    ft.close(); pt.close(); dedup.close()
    print("Finished process_train.")


def deep_merge(dict1, dict2):
    merged = dict1.copy()
    for k, v in dict2.items():
        if k not in merged: merged[k] = v
        else:
            v1 = merged[k]
            if isinstance(v1, list) and isinstance(v, list):
                for item in v:
                    if item not in v1: v1.append(item)
    return merged

def merge_dictionaries():
    print("Starting merge_dictionaries...")
    bw = defaultdict(list)
    for path in [DICT_PATH_MERGE, PUBTEXTS_PATH_MERGE]:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    e = json.loads(line)
                    bw[e.get('word')].append(e)
    fd = {}
    for w in sorted(bw.keys()):
        el = bw[w]
        if len(el) == 1: fd[w] = el[0]
        else:
            m = el[0].copy()
            for e in el[1:]: m = deep_merge(m, e)
            fd[w] = m
    with open(OUTPUT_MERGED_DICT, 'w', encoding='utf-8') as f: json.dump(fd, f, ensure_ascii=False, indent=2)
    print("Finished merge_dictionaries.")


def format_entry(lemma, entry):
    meanings, grammars = [], []
    for m in entry.get("meanings", [])[:1]:
        if m.get("definition") and m.get("definition") not in meanings: meanings.append(m.get("definition"))
        for g in m.get("grammar", [])[:1]:
            parse = g.get("parse") or ", ".join([str(v) for k, v in g.items() if k not in ("classification", "parse")] + (["clitic", g['clitic']] if "clitic" in g else []))
            if parse and parse not in grammars: grammars.append(parse)
    if not meanings and entry.get("original_definition"): meanings.append(entry.get("original_definition"))
    res = {"Lemma": lemma}
    if meanings: res["Meanings"] = '; '.join(meanings[:3])
    if grammars: res["Grammar"] = ", ".join(sorted({part for s in grammars for part in s.split(", ")}))
    return res

def fetch_dict_info(cand, d2l, fd, f2e):
    lemmas = d2l.get(cand, [])
    for l in (lemmas + [cand] if cand in fd and cand not in lemmas else lemmas):
        if l in fd: return format_entry(l, fd[l])
    for k in f2e.get(cand, []): return format_entry(k, fd[k])
    return None

def direct_lookup(term, d2l, fd, f2e, is_first=False, is_last=False):
    if re.match(r'^[0-9\./]+$', term): return term, {"Type": "number"}
    if term == "<gap>" or "<gap>" in term: return term, {"Type": "gap token"}
    cands = [term]
    if is_last: cands.insert(0, "-" + term)
    if is_first: cands.insert(0, term + "-")
    for cand in cands:
        res = fetch_dict_info(cand, d2l, fd, f2e)
        if res: return cand, res
    cfc = ''.join([c for c in re.sub(r'\(.*?\)', '', term) if c.isalpha()])
    if len(cfc) > 0 and cfc.isupper(): return term, {"Type": "Proper Noun"}
    return term, None

def resolve_composite(term, d2l, fd, f2e, is_first=False, is_last=False):
    cand, res = direct_lookup(term, d2l, fd, f2e, is_first, is_last)
    if res: return [(cand, res)]
    parts = term.split('-')
    if len(parts) > 1:
        lr = resolve_composite("-".join(parts[:-1]), d2l, fd, f2e, is_first=is_first, is_last=False)
        rr = resolve_composite(parts[-1], d2l, fd, f2e, is_first=False, is_last=True)
        def cr(rl): return sum(1 for c, r in rl if isinstance(r, dict) and r.get("Type") != "Unknown")
        sl = cr(lr) + cr(rr)
        lr2 = resolve_composite(parts[0], d2l, fd, f2e, is_first=is_first, is_last=False)
        rr2 = resolve_composite("-".join(parts[1:]), d2l, fd, f2e, is_first=False, is_last=is_last)
        sf = cr(lr2) + cr(rr2)
        if sl > 0 or sf > 0: return lr + rr if sl >= sf else lr2 + rr2
    return []

def process_reasoned_translations():
    print("Starting process_reasoned_translations...")
    with open(LEMMA_DERIVATIVES_JSON, "r", encoding="utf-8") as f: ld = json.load(f)
    d2l = defaultdict(list)
    for l, ds in ld.items():
        for d in ds: d2l[d].append(l)
    with open(OUTPUT_MERGED_DICT, "r", encoding="utf-8") as f: fd = json.load(f)
    f2e = defaultdict(list)
    for k, e in fd.items():
        for form in e.get("forms", []): f2e[form].append(k)
        
    f_out = open(f"{OUTPUT_DIR_TRAIN}/reasoned_translations_finetune.csv", "w", encoding="utf-8")
    w = csv.writer(f_out, **CSV_DIALECT_FINETUNE)
    w.writerow(["instruct", "query", "result"])
    pi = f"{PROMPT_TRANS_AKK_TO_ENG.replace('%type_name%', TYPE_EPIGRAPHIC)} with reasoning"
    
    with open(INPUT_TRAIN, "r", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            tlit, tlat = replace_gaps(row.get("transliteration", "").strip()), replace_gaps(row.get("translation", "").strip())
            if not tlit or not tlat: continue
            r_list = []
            for w_tok in tlit.split():
                resos = resolve_composite(w_tok, d2l, fd, f2e, True, True)
                if len(resos) == 1:
                    c, r = resos[0]
                    it = {"Word": w_tok}
                    if it["Word"] == it.get("Lemma", ""): del it["Lemma"]
                    it.update(r); r_list.append(it)
                else:
                    pl = []
                    for c, r in resos:
                        pi_it = {"Word": c}
                        if pi_it["Word"] == pi_it.get("Lemma", ""): del pi_it["Lemma"]
                        pi_it.update(r); pl.append(pi_it)
                    r_list.append({"Word": w_tok, "Parts": pl})
            rows = [linearize(pi, True), linearize(tlit, True), linearize(f"{tlat}\nREASONING:\n{yaml.dump(r_list, default_flow_style=False, sort_keys=False, allow_unicode=True).strip()}", True)]
            if sum(len(r) for r in rows) <= 1024 * 2.33: w.writerow(rows)
    f_out.close()
    print("Finished process_reasoned_translations.")


def process_file_list(fl, eh):
    ar = []
    for inf in fl:
        if not os.path.exists(inf): continue
        with open(inf, 'r', encoding='utf-8') as f:
            r = csv.reader(f)
            try:
                if next(r) == eh: ar.extend([row for row in r if any(x.strip() for x in row)])
            except StopIteration: pass
    return ar

def merge_csvs():
    print("Starting merge_csvs...")
    random.seed(42)  # For mathematically reproducible datasets
    os.makedirs("workspace/outputs", exist_ok=True)
    with open("workspace/outputs/pretrain.csv", 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, **CSV_DIALECT_PRETRAIN)
        w.writerow(["content"])
        ar = process_file_list(PRETRAIN_FILES, ["content"])
        random.shuffle(ar)
        for r in ar: w.writerow(r)
        
    with open("workspace/outputs/finetune.csv", 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f, **CSV_DIALECT_FINETUNE)
        w.writerow(["instruct", "query", "result"])
        for stage in FINETUNE_FILES:
            ar = process_file_list(stage, ["instruct", "query", "result"])
            random.shuffle(ar)
            for r in ar: w.writerow(r)
    print("Finished merge_csvs.")

# --- EXECUTION ---
process_published()
process_dictionaries()
process_publications()
process_train()
merge_dictionaries()
process_reasoned_translations()
merge_csvs()
print("Akkadian pipeline execution complete.")
