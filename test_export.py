import os
import shutil
import subprocess
import filecmp

def compare_file(path_true, path_test):
    if not os.path.exists(path_true):
        print(f"Skipping: {path_true} not in ground truth.")
        return True
    if not os.path.exists(path_test):
        print(f"FAILED: {path_test} was not generated.")
        return False
    
    # For JSON files, we might treat them specially if formatting differs, 
    # but the export script should produce 1:1 if handled correctly.
    if filecmp.cmp(path_true, path_test, shallow=False):
        print(f"PASSED: {path_test} matches ground truth.")
        return True
    else:
        # Check if they are CSVs and maybe if line counts match
        print(f"FAILED: {path_test} does not match ground truth.")
        return False

def test():
    print("--- STARTING EXPORT VALIDATION ---")
    
    # 1. Clean and prepare workspace
    if os.path.exists("workspace/outputs"):
        shutil.rmtree("workspace/outputs")
    os.makedirs("workspace/outputs/lexicon", exist_ok=True)
    # Lexicon is expected to be there as it is not part of this export logic (per user prompt)
    if os.path.exists("workspace/outputs_true/lexicon"):
        shutil.copytree("workspace/outputs_true/lexicon", "workspace/outputs/lexicon", dirs_exist_ok=True)

    # 2. Run export.py
    print("Running export.py...")
    subprocess.run(["python3", "export.py"], check=True)
    print("export.py execution finished.\n")

    # 3. Compare outputs
    files_to_check = [
        "published_texts/dictionary_parsed.jsonl",
        "published_texts/texts_pretrain.csv",
        "published_texts/translations_pretrain.csv",
        "published_texts/translations_finetune.csv",
        "dictionary/dictionary_parsed.jsonl",
        "dictionary/translations_finetune.csv",
        "publications/publications_pretrain.csv",
        "train/translations_pretrain.csv",
        "train/translations_finetune.csv",
        "train/reasoned_translations_finetune.csv",
        "final_dictionary.json",
        "pretrain.csv",
        "finetune.csv"
    ]

    all_passed = True
    for f in files_to_check:
        path_true = os.path.join("workspace/outputs_true", f)
        path_test = os.path.join("workspace/outputs", f)
        if not compare_file(path_true, path_test):
            all_passed = False

    if all_passed:
        print("\nSUCCESS: All files match ground truth exactly.")
    else:
        print("\nFAILURE: Some files do not match.")

if __name__ == "__main__":
    test()
