import csv
import os
from corpus_utils import (
    CSV_DIALECT_FINETUNE, CSV_DIALECT_PRETRAIN, Deduplicator, linearize,
    PROMPT_TRANS_AKK_TO_ENG, PROMPT_TRANS_ENG_TO_AKK
)

def process_train():
    input_file = "workspace/train.csv"
    output_dir = "workspace/outputs/train"
    os.makedirs(output_dir, exist_ok=True)
    
    dedup = Deduplicator(os.path.join(output_dir, "dedup.db"))

    f_trans_finetune = open(os.path.join(output_dir, "translations_finetune.csv"), "w", encoding="utf-8")
    writer_ft = csv.writer(f_trans_finetune, **CSV_DIALECT_FINETUNE)
    writer_ft.writerow(["instruct", "query", "result"])

    f_trans_pretrain = open(os.path.join(output_dir, "translations_pretrain.csv"), "w", encoding="utf-8")
    writer_pt = csv.writer(f_trans_pretrain, **CSV_DIALECT_PRETRAIN)
    writer_pt.writerow(["content"])

    print("Processing train.csv dataset...")
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            oare_id = row.get("oare_id", "").strip()
            translit = row.get("transliteration", "").strip()
            translat = row.get("translation", "").strip()

            if not translit or not translat or not oare_id:
                continue

            # Trim OARE ID to 8 characters to save tokens, matching standard git log formatting
            short_id = oare_id[:8]

            # 1. Pretrain Content
            content_str = f"# Akkadian Transliteration\n(OARE:{short_id})\n{translit}"
            writer_pt.writerow([linearize(content_str)])

            # 2. Finetune translation pairs
            # Akkadian -> English
            inst_akk_to_eng = PROMPT_TRANS_AKK_TO_ENG.replace("%type_name%", "epigraphic transliteration")
            if dedup.is_unique("trans_ft_akk2eng", inst_akk_to_eng, translit, translat):
                writer_ft.writerow([inst_akk_to_eng, translit, translat])

            # English -> Akkadian
            inst_eng_to_akk = PROMPT_TRANS_ENG_TO_AKK.replace("%type_name%", "epigraphic transliteration")
            if dedup.is_unique("trans_ft_eng2akk", inst_eng_to_akk, translat, translit):
                writer_ft.writerow([inst_eng_to_akk, translat, translit])

    f_trans_finetune.close()
    f_trans_pretrain.close()
    dedup.close()
    print("Train dataset processing complete.")

if __name__ == "__main__":
    process_train()
