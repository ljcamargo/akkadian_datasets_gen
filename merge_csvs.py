import csv
import os
import random
from corpus_utils import CSV_DIALECT_FINETUNE, CSV_DIALECT_PRETRAIN


PRETRAIN_FILES = [
    "workspace/outputs/published_texts/grammar_pretrain.csv",
    "workspace/outputs/published_texts/translations_pretrain.csv",
    "workspace/outputs/published_texts/texts_pretrain.csv",
    "workspace/outputs/published_texts/rosetta_pretrain.csv",

    "workspace/outputs/dictionary/dictionary_pretrain.csv",
    "workspace/outputs/dictionary/rosetta_pretrain.csv",

    "workspace/outputs/lexicon/lemma_pretrain.csv",
    "workspace/outputs/lexicon/rosetta_pretrain.csv",

    "workspace/outputs/publications/publications_pretrain.csv",

    "workspace/outputs/train/translations_pretrain.csv",
]

FINETUNE_FILES = [
    # EASY
    [
        "workspace/outputs/dictionary/lemma_finetune.csv",
        "workspace/outputs/dictionary/grammar_finetune.csv",
        "workspace/outputs/dictionary/meaning_finetune.csv",

        "workspace/outputs/published_texts/lemma_finetune.csv",
        "workspace/outputs/published_texts/grammar_finetune.csv",
        "workspace/outputs/published_texts/meanings_finetune.csv",
        #"workspace/outputs/published_texts/transforms_finetune.csv",

        "workspace/outputs/lexicon/lemma_finetune.csv",
    ],
    # MEDIUM
    [
        "workspace/outputs/published_texts/translations_finetune.csv",
        "workspace/outputs/dictionary/translations_finetune.csv",
    ],
    # HARD
    [
        "workspace/outputs/train/translations_finetune.csv",
        "workspace/outputs/train/reasoned_translations_finetune.csv",
    ],
]

def process_file_list(file_list, expected_header):
    all_rows = []
    
    for input_file in file_list:
        if not os.path.exists(input_file):
            print(f"Warning: {input_file} not found. Skipping.")
            continue
            
        with open(input_file, 'r', encoding='utf-8') as in_f:
            reader = csv.reader(in_f)
            try:
                header = next(reader)
                if header != expected_header:
                    print(f"Error: {input_file} has header {header}, expected {expected_header}. Skipping.")
                    continue
                    
                count = 0
                for row in reader:
                    if any(x.strip() for x in row):
                        all_rows.append(row)
                        count += 1
                        
                print(f"  + Extracted {count} rows from {input_file}")
            except StopIteration:
                print(f"Warning: {input_file} is empty. Skipping.")
                
    return all_rows

def merge_csvs_for_target(output_file, input_files, expected_header, output_dialect, preserve_curriculum=False):
    print(f"Generating {output_file}...")
    
    total_written = 0
    with open(output_file, 'w', encoding='utf-8', newline='') as out_f:
        writer = csv.writer(out_f, **output_dialect)
        writer.writerow(expected_header)
        
        if preserve_curriculum:
            # input_files is a list of lists (curriculum stages)
            for stage_idx, stage_files in enumerate(input_files):
                print(f"Processing curriculum stage {stage_idx + 1}...")
                stage_rows = process_file_list(stage_files, expected_header)
                random.shuffle(stage_rows)
                
                for row in stage_rows:
                    writer.writerow(row)
                
                total_written += len(stage_rows)
                print(f"  -> Shuffled and added {len(stage_rows)} rows for stage {stage_idx + 1}")
        else:
            # input_files is a flat list
            all_rows = process_file_list(input_files, expected_header)
            random.shuffle(all_rows)
            
            for row in all_rows:
                writer.writerow(row)
            
            total_written += len(all_rows)

    file_size_bytes = os.path.getsize(output_file)
    file_size_mb = file_size_bytes / (1024 * 1024)
    print(f"Done. Wrote {total_written} rows to {output_file}.")
    print(f"Final file size: {file_size_bytes} bytes ({file_size_mb:.2f} MB)\n")

def merge_csvs():
    random.seed(42)  # Fixed seed for mathematically reproducible datasets
    
    output_dir = "workspace/outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    finetune_out = os.path.join(output_dir, "finetune.csv")
    merge_csvs_for_target(
        finetune_out, 
        FINETUNE_FILES, 
        ["instruct", "query", "result"], 
        CSV_DIALECT_FINETUNE,
        preserve_curriculum=True
    )
    
    pretrain_out = os.path.join(output_dir, "pretrain.csv")
    merge_csvs_for_target(
        pretrain_out, 
        PRETRAIN_FILES, 
        ["content"], 
        CSV_DIALECT_PRETRAIN,
        preserve_curriculum=False
    )

if __name__ == "__main__":
    merge_csvs()
