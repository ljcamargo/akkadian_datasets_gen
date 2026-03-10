import csv
import os

import json
import re
import yaml
from corpus_utils import CSV_DIALECT_FINETUNE, replace_gaps

print("Loading dictionaries...")
with open("workspace/outputs/lexicon/lemma_derivatives.json", "r", encoding="utf-8") as f:
    lemma_derivatives = json.load(f)

derivative_to_lemma = {}
for lemma, derivs in lemma_derivatives.items():
    for d in derivs:
        if d not in derivative_to_lemma:
            derivative_to_lemma[d] = []
        derivative_to_lemma[d].append(lemma)

with open("workspace/outputs/final_dictionary.json", "r", encoding="utf-8") as f:
    final_dictionary = json.load(f)

form_to_entry = {}
for key, entry in final_dictionary.items():
    for form in entry.get("forms", []):
        if form not in form_to_entry:
            form_to_entry[form] = []
        form_to_entry[form].append(key)

print("Dictionaries loaded.")

def format_entry(lemma, entry):
    meanings = []
    grammars = []
    for m in entry.get("meanings", []):
        if m.get("definition") and m.get("definition") not in meanings:
            meanings.append(m.get("definition"))
        for g in m.get("grammar", []):
            if g.get("parse"):
                if g.get("parse") not in grammars:
                    grammars.append(g.get("parse"))
            else:
                if g not in grammars:
                    g_copy = g.copy()
                    del g_copy["parse"]
                    grammars.append(g_copy)
    
    if not meanings and entry.get("original_definition"):
        meanings.append(entry.get("original_definition"))
        
    res = {"Lemma": lemma}
    if meanings:
        res["Meanings"] = '; '.join(meanings)
    if grammars:
        res["Grammar"] = grammars
    return res

def fetch_dict_info(cand):
    lemmas = derivative_to_lemma.get(cand, [])
    if cand in final_dictionary and cand not in lemmas:
        lemmas_copy = lemmas + [cand]
    else:
        lemmas_copy = lemmas
        
    for lemma in lemmas_copy:
        if lemma in final_dictionary:
            return format_entry(lemma, final_dictionary[lemma])
            
    entry_keys = form_to_entry.get(cand, [])
    for k in entry_keys:
        return format_entry(k, final_dictionary[k])
        
    return None

def direct_lookup(term, is_first=False, is_last=False):
    if re.match(r'^[0-9\./]+$', term):
        return term, {"Type": "number"}
    
    if term == "<|GAP|>" or "<|GAP|>" in term:
        return term, {"Type": "gap/missing token"}
        
    candidates = [term]
    if is_last:
        candidates.insert(0, "-" + term)
    if is_first:
        candidates.insert(0, term + "-")
        
    for cand in candidates:
        res = fetch_dict_info(cand)
        if res:
            return cand, res
        
    clean_for_caps = re.sub(r'\(.*?\)', '', term)
    clean_for_caps = ''.join([c for c in clean_for_caps if c.isalpha()])
    if len(clean_for_caps) > 0 and clean_for_caps.isupper():
        return term, {"Type": "Proper Noun"}
        
    return term, None

def resolve_composite(term, is_first=False, is_last=False):
    cand, res = direct_lookup(term, is_first, is_last)
    #print(f"Resolving '{term}': direct lookup candidate '{cand}' with result?: {res is not None}")
    if res:
        return [(cand, res)]
        
    parts = term.split('-')
    if len(parts) > 1:
        # Try split by last
        left_term = "-".join(parts[:-1])
        right_term = parts[-1]
        
        left_res = resolve_composite(left_term, is_first=is_first, is_last=False)
        right_res = resolve_composite(right_term, is_first=False, is_last=True)
        
        def count_resolved(res_list):
            return sum(1 for c, r in res_list if isinstance(r, dict) and r.get("Type") != "Unknown")
            
        score_last = count_resolved(left_res) + count_resolved(right_res)
        
        # Try split by first
        left_term2 = parts[0]
        right_term2 = "-".join(parts[1:])
        
        left_res2 = resolve_composite(left_term2, is_first=is_first, is_last=False)
        right_res2 = resolve_composite(right_term2, is_first=False, is_last=is_last)
        
        score_first = count_resolved(left_res2) + count_resolved(right_res2)
        
        if score_last > 0 or score_first > 0:
            if score_last >= score_first:
                return left_res + right_res
            else:
                return left_res2 + right_res2
                
    return [(term, {"Type": "Unknown"})]

def process_reasoned():
    input_file = "workspace/train.csv"
    output_dir = "workspace/outputs/train"
    os.makedirs(output_dir, exist_ok=True)
    
    out_path = os.path.join(output_dir, "reasoned_translations_finetune.csv")
    f_out = open(out_path, "w", encoding="utf-8")
    writer = csv.writer(f_out, **CSV_DIALECT_FINETUNE)
    writer.writerow(["instruct", "query", "result"])

    prompt_inst = "Translate this Akkadian cuneiform epigraphic transliteration into English with reasoned explanation"

    print("Processing train.csv dataset to generate reasoned translations...")
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        count = 0
        for row in reader:
            translit = row.get("transliteration", "").strip()
            translat = row.get("translation", "").strip()

            if not translit or not translat:
                continue
                
            translit = replace_gaps(translit)
            translat = replace_gaps(translat)
            #print(">>>>> Processing entry with transliteration:", translit)
            reasoning_list = []
            words = translit.split()
            for w in words:
                #print(f">>>>> >>>>> WORD: {w}")
                resolutions = resolve_composite(w, is_first=True, is_last=True)
                if len(resolutions) == 1:
                    cand, r = resolutions[0]
                    item = {"Word": w}
                    item.update(r)
                    reasoning_list.append(item)
                else:
                    parts_list = []
                    for cand, r in resolutions:
                        part_item = {"Word": cand}
                        part_item.update(r)
                        parts_list.append(part_item)
                    reasoning_list.append({"Word": w, "Parts": parts_list})
            
            reasoning_str = yaml.dump(reasoning_list, default_flow_style=False, sort_keys=False, allow_unicode=True)
            # escape all newlines in the translation to avoid CSV issues
            result_str = f"REASONING:\n{reasoning_str.strip()}\nTRANSLATION:\n{translat}"
            result_str = result_str.replace("\n", "\\n")
            writer.writerow([prompt_inst, translit, result_str])
            count += 1
            if count % 1000 == 0:
                print(f"Processed {count} entries...")

    f_out.close()
    print("Reasoned train dataset processing complete.")

if __name__ == "__main__":
    process_reasoned()
