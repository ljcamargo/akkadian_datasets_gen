import csv
import os
from corpus_utils import CSV_DIALECT_PRETRAIN, linearize, get_akkadian_context_lines, remove_nul

input_file = "workspace/publications.csv"
output_dir = "workspace/outputs/publications"
os.makedirs(output_dir, exist_ok=True)

def process_publications():
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(remove_nul(f))
        
        with open(os.path.join(output_dir, "publications_pretrain.csv"), "w", encoding="utf-8") as out_f:
            writer = csv.writer(out_f, **CSV_DIALECT_PRETRAIN)
            writer.writerow(["content"])
            
            total_original_lines = 0
            total_filtered_lines = 0
            total_pages = 0
            no_match_pages = 0
            
            for row in reader:
                pdf_name = row.get("pdf_name", "").strip()
                page = row.get("page", "").strip()
                page_text = row.get("page_text", "").strip()
                
                if not pdf_name and not page_text:
                    continue
                
                filtered_page_text = get_akkadian_context_lines(page_text)
                
                separator = '\\n' if '\\n' in page_text and '\n' not in page_text else '\n'
                orig_lines = len(page_text.split(separator)) if page_text else 0
                
                total_original_lines += orig_lines
                total_pages += 1
                
                if not filtered_page_text:
                    no_match_pages += 1
                    continue
                
                filt_lines = len(filtered_page_text.split('\n'))
                total_filtered_lines += filt_lines
                
                #content = f"{pdf_name} (p.{page})\n{filtered_page_text}"
                content = filtered_page_text
                writer.writerow([linearize(content)])
                
    reduction_pct = (1.0 - (total_filtered_lines / total_original_lines)) * 100 if total_original_lines > 0 else 0
    no_match_pct = (no_match_pages / total_pages) * 100 if total_pages > 0 else 0
    
    print("\n------------------------------")
    print(f"Total Original Lines: {total_original_lines}")
    print(f"Total Filtered Lines: {total_filtered_lines}")
    print(f"Line Reduction Percentage: {reduction_pct:.2f}%")
    print(f"Total Pages Analyzed: {total_pages}")
    print(f"Pages with No Match: {no_match_pages} ({no_match_pct:.2f}%)")
    print("------------------------------\n")
    print("Publications pretrain processing complete.")

if __name__ == "__main__":
    process_publications()
