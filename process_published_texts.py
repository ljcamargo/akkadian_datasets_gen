import json
import csv
import os
import argparse
import time
from collections import defaultdict
from corpus_utils import *


# --- CONFIGURATION CONSTANTS ---
INPUT_DEFAULT = "workspace/oare_epigraphies.jsonl"
OUTPUT_DIR = "workspace/outputs/published_texts"
PRETRAIN_CHUNK_SIZE = 40
ROSETTA_CHUNK_SIZE = 20

# --- HELPERS ---


def group_units_by_spelling(units, key):
    """Group units strictly if they are consecutive and share the same spellingUuid."""
    if not units: return []
    groups = []
    current_group = []
    current_uuid = None
    
    for u in units:
        val = u.get(key)
        uuid = u.get("spellingUuid")
        
        if val is None or uuid is None:
            if current_group:
                groups.append(current_group)
                current_group = []
                current_uuid = None
            continue
        
        if uuid == current_uuid and current_uuid is not None:
            current_group.append(val)
        else:
            if current_group:
                groups.append(current_group)
            current_group = [val]
            current_uuid = uuid
    if current_group:
        groups.append(current_group)
    return groups

def format_epigraphy(units, compact=False):
    structure = defaultdict(lambda: defaultdict(list))
    for u in units:
        if u.get("side") and u.get("line") is not None:
            structure[u["side"]][u["line"]].append(u)
    
    lines_output = []
    sides_processed = []
    sorted_sides = sorted(structure.keys(), key=lambda x: 0 if x == "obv." else 1 if x == "rev." else 2)
    
    for side in sorted_sides:
        if side == "rev." and ("obv." in sides_processed or any("obv" in str(s).lower() for s in sides_processed)):
            lines_output.append("---")
        sides_processed.append(side)
        
        sorted_lines = sorted(structure[side].keys())
        for line_num in sorted_lines:
            line_units = structure[side][line_num]
            groups = group_units_by_spelling(line_units, "epigReading")
            
            sep = "" if compact else "-"
            formatted_groups = [sep.join(g) for g in groups]
            if formatted_groups:
                lines_output.append(" ".join(formatted_groups))
            
    return "\n".join(lines_output)

def format_spelling(units):
    structure = defaultdict(lambda: defaultdict(list))
    for u in units:
        if u.get("side") and u.get("line") is not None:
            structure[u["side"]][u["line"]].append(u)
    
    lines_output = []
    sorted_sides = sorted(structure.keys(), key=lambda x: 0 if x == "obv." else 1 if x == "rev." else 2)
    for side in sorted_sides:
        if side == "rev." and lines_output:
            lines_output.append("---")
        sorted_lines = sorted(structure[side].keys())
        for line_num in sorted_lines:
            line_units = [u for u in structure[side][line_num] if u.get("form") is not None]
            if line_units:
                text = " ".join([u["form"] for u in line_units])
                lines_output.append(text)
    return "\n".join(lines_output)

# --- CORE TASKS ---

def process_corpus(args):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    dedup = Deduplicator(f"{OUTPUT_DIR}/dedup.db")
    
    # Finetune files
    ft_files = {
        "grammar_finetune": open(f"{OUTPUT_DIR}/grammar_finetune.csv", "w", encoding="utf-8"),
        "meanings_finetune": open(f"{OUTPUT_DIR}/meanings_finetune.csv", "w", encoding="utf-8"),
        "lemma_finetune": open(f"{OUTPUT_DIR}/lemma_finetune.csv", "w", encoding="utf-8"),
        "translations_finetune": open(f"{OUTPUT_DIR}/translations_finetune.csv", "w", encoding="utf-8"),
        "transforms_finetune": open(f"{OUTPUT_DIR}/transforms_finetune.csv", "w", encoding="utf-8"),
    }
    
    # Pretrain files
    pt_files = {
        "texts_pretrain": open(f"{OUTPUT_DIR}/texts_pretrain.csv", "w", encoding="utf-8"),
        "grammar_pretrain": open(f"{OUTPUT_DIR}/grammar_pretrain.csv", "w", encoding="utf-8"),
        "translations_pretrain": open(f"{OUTPUT_DIR}/translations_pretrain.csv", "w", encoding="utf-8"),
        "rosetta_pretrain": open(f"{OUTPUT_DIR}/rosetta_pretrain.csv", "w", encoding="utf-8"),
    }
    
    ft_writers = {k: csv.writer(v, **CSV_DIALECT_FINETUNE) for k, v in ft_files.items()}
    pt_writers = {k: csv.writer(v, **CSV_DIALECT_PRETRAIN) for k, v in pt_files.items()}
    
    for w in ft_writers.values():
        w.writerow(["instruct", "query", "result"])
    for w in pt_writers.values():
        w.writerow(["content"])

    # Pretrain Buffers
    trans_p_buffers = defaultdict(list) 
    rosetta_buffer = []

    with open(args.input, "r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f, start=1):
            if args.start and line_idx < args.start: continue
            if args.end and line_idx > args.end: break
            
            try:
                data = json.loads(line)
            except:
                continue
                
            text_meta = data.get("text", {})
            units = data.get("units", [])
            pub_info = get_text_id(text_meta)
            
            # --- 1. TEXT PRETRAIN ---
            # Uniqueness: Full text content per mode
            variants_md = [
                ("epigraphic transliteration", format_epigraphy(units, False)),
                ("compact epigraphic transliteration", format_epigraphy(units, True)),
                ("akkadian orthography", format_spelling(units))
            ]
            for type_name, content in variants_md:
                if content:
                    md_full = get_markdown_header(text_meta, type_name) + content
                    if dedup.is_unique("texts_pretrain", type_name, content):
                        pt_writers["texts_pretrain"].writerow([linearize(md_full)])

            # --- 2. Sequential Word Groups ---
            valid_units = [u for u in units if u.get("side") and u.get("line") is not None]
            word_groups = []
            if valid_units:
                cur_group = [valid_units[0]]
                cur_uuid = valid_units[0].get("spellingUuid")
                for u in valid_units[1:]:
                    uuid = u.get("spellingUuid")
                    if uuid == cur_uuid and uuid is not None:
                        cur_group.append(u)
                    else:
                        word_groups.append(cur_group)
                        cur_group = [u]
                        cur_uuid = uuid
                word_groups.append(cur_group)

            # --- 3. GRAMMAR PRETRAIN & Process Words ---
            # Note: Grammar Pretrain is Word-Based. It shares logic with Grammar Finetune logic below
            
            for group in word_groups:
                g_epig = [u["epigReading"] for u in group if u.get("epigReading")]
                g_form = [u["form"] for u in group if u.get("form")]
                u_word = next((u["word"] for u in group if u.get("word")), None)
                u_trans = next((u["translation"] for u in group if u.get("translation")), None)
                parse_raw = get_grammar_result(group)
                
                q_e = "-".join(g_epig)
                q_c = "".join(g_epig)
                q_f = " ".join(g_form)
                
                word_variants = [
                    ("epigraphic transliteration", q_e),
                    ("compact epigraphic transliteration", q_c),
                    ("akkadian orthography", q_f)
                ]

                # Grammar (Finetune + Pretrain)
                if parse_raw:
                    for type_name, q_val in word_variants:
                        if q_val:
                            # Unique key: (Task, Type, Word, Grammar)
                            if dedup.is_unique("grammar", type_name, q_val, parse_raw):
                                # Finetune
                                ft_writers["grammar_finetune"].writerow([
                                    PROMPT_GRAMMAR_FINETUNE.replace("%type_name%", type_name),
                                    q_val,
                                    linearize(parse_raw)
                                ])
                                # Pretrain (Linked to same unique entry)
                                title_pt = PROMPT_GRAMMAR_PRETRAIN_TITLE.replace("%type_name%", type_name)
                                content_pt = PROMPT_GRAMMAR_PRETRAIN_CONTENT.replace("%title%", title_pt).replace("%word%", q_val).replace("%grammar%", parse_raw)
                                pt_writers["grammar_pretrain"].writerow([linearize(content_pt)])

                # Meaning Finetune
                if u_trans:
                    for type_name, q_val in word_variants:
                        if q_val:
                            # Handle PN replacement for Meanings
                            meaning_res = q_val if u_trans == "PN" else u_trans
                            res_t = linearize(meaning_res)
                            if dedup.is_unique("meaning", type_name, q_val, res_t):
                                ft_writers["meanings_finetune"].writerow([
                                    PROMPT_MEANING_FINETUNE_TRANS.replace("%type_name%", type_name),
                                    q_val,
                                    res_t
                                ])
                    if u_word:
                        word_meaning_res = u_word if u_trans == "PN" else u_trans
                        res_t = linearize(word_meaning_res)
                        if dedup.is_unique("meaning_word", u_word, res_t):
                            ft_writers["meanings_finetune"].writerow([PROMPT_MEANING_FINETUNE_WORD, u_word, res_t])

                # Lemma Finetune
                if u_word:
                    res_w = linearize(u_word)
                    for type_name, q_val in word_variants:
                        if q_val:
                            if dedup.is_unique("lemma", type_name, q_val, res_w):
                                ft_writers["lemma_finetune"].writerow([
                                    PROMPT_LEMMA_FINETUNE.replace("%type_name%", type_name),
                                    q_val,
                                    res_w
                                ])

                # Translation Finetune & Pretrain Buffer
                if u_trans:
                    unique_parts = [p.strip() for p in u_trans.split(",") if p.strip()]
                    for part in unique_parts:
                        for type_name, q_val in word_variants:
                            if q_val:
                                # Handle PN replacement for Translations
                                actual_part = q_val if part == "PN" else part
                                if dedup.is_unique("trans", type_name, q_val, actual_part):
                                    ft_writers["translations_finetune"].writerow([PROMPT_TRANS_AKK_TO_ENG.replace("%type_name%", type_name), q_val, actual_part])
                                    ft_writers["translations_finetune"].writerow([PROMPT_TRANS_ENG_TO_AKK.replace("%type_name%", type_name), actual_part, q_val])
                                    trans_p_buffers[(type_name, "to_eng")].append((q_val, actual_part))
                                    trans_p_buffers[(type_name, "from_eng")].append((actual_part, q_val))

                # Transforms Finetune
                transforms = [
                    (q_e, q_f, PROMPT_TRANSFORM_EPIG_TO_SPELL),
                    (q_f, q_e, PROMPT_TRANSFORM_SPELL_TO_EPIG),
                    (q_c, q_f, PROMPT_TRANSFORM_COMPACT_TO_SPELL),
                    (q_f, q_c, PROMPT_TRANSFORM_SPELL_TO_COMPACT)
                ]
                for src, dst, inst in transforms:
                    if src and dst and src != dst:
                        if dedup.is_unique("transform", inst, src, dst):
                            ft_writers["transforms_finetune"].writerow([inst, src, dst])

                # Rosetta Buffer
                # Handle PN for Rosetta Meaning
                rosetta_trans = u_trans
                if u_trans == "PN":
                    rosetta_trans = u_word if u_word else q_f
                
                r_tuple = (q_e or "", q_c or "", q_f or "", u_word or "", rosetta_trans or "")
                # Only add if unique
                if dedup.is_unique("rosetta", *r_tuple):
                    rosetta_buffer.append(r_tuple)

            # --- 5. Flush Pretrain Chunks (Tables) ---
            for key in list(trans_p_buffers.keys()):
                buf = trans_p_buffers[key]
                while len(buf) >= PRETRAIN_CHUNK_SIZE:
                    type_name, direction = key
                    chunk = buf[:PRETRAIN_CHUNK_SIZE]
                    buf = buf[PRETRAIN_CHUNK_SIZE:]
                    trans_p_buffers[key] = buf
                    
                    label_akk = type_name.capitalize()
                    if direction == "to_eng":
                        h1, h2 = label_akk, "English"
                        title = TITLE_TRANS_PT_TO_ENG.replace("%type_name%", type_name)
                    else:
                        h1, h2 = "English", label_akk
                        title = TITLE_TRANS_PT_FROM_ENG.replace("%type_name%", type_name)
                    
                    table = f"{title}\n\n" + TRANS_TABLE_TEMPLATE.replace("%h1%", h1).replace("%h2%", h2)
                    for c1, c2 in chunk: table += f"| {c1} | {c2} |\n"
                    pt_writers["translations_pretrain"].writerow([linearize(table)])

            while len(rosetta_buffer) >= ROSETTA_CHUNK_SIZE:
                chunk = rosetta_buffer[:ROSETTA_CHUNK_SIZE]
                rosetta_buffer = rosetta_buffer[ROSETTA_CHUNK_SIZE:]
                title = TITLE_ROSETTA_PT
                table = f"{title}\n\n" + ROSETTA_TABLE_HEADER
                for e, c, f, w, m in chunk: table += f"| {e} | {c} | {f} | {w} | {m} |\n"
                pt_writers["rosetta_pretrain"].writerow([linearize(table)])

    # Delete dedup db
    dedup.close()
    if os.path.exists(f"{OUTPUT_DIR}/dedup.db"):
        os.remove(f"{OUTPUT_DIR}/dedup.db")
        
    for f in (list(ft_files.values()) + list(pt_files.values())):
        f.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=INPUT_DEFAULT)
    parser.add_argument("--start", type=int)
    parser.add_argument("--end", type=int)
    args = parser.parse_args()
    
    print(f"Starting corpus generation for {args.input}...")
    process_corpus(args)
    print("Done.")
