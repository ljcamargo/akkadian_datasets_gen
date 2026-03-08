import csv
import os
from corpus_utils import CSV_DIALECT_PRETRAIN, linearize

input_file = "workspace/publications.csv"
output_dir = "workspace/outputs/publications"
os.makedirs(output_dir, exist_ok=True)

def remove_nul(file_iter):
    for line in file_iter:
        yield line.replace('\0', '')

def process_publications():
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(remove_nul(f))
        
        with open(os.path.join(output_dir, "publications_pretrain.csv"), "w", encoding="utf-8") as out_f:
            writer = csv.writer(out_f, **CSV_DIALECT_PRETRAIN)
            writer.writerow(["content"])
            
            for row in reader:
                pdf_name = row.get("pdf_name", "").strip()
                page = row.get("page", "").strip()
                page_text = row.get("page_text", "").strip()
                
                if not pdf_name and not page_text:
                    continue
                
                content = f"# {pdf_name}\n## Page: {page}\n{page_text}"
                writer.writerow([linearize(content)])
                
    print("Publications pretrain processing complete.")

if __name__ == "__main__":
    process_publications()
