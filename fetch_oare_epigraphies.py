import csv
import json
import os
import time
import requests
import argparse

def fetch_epigraphy(oare_id):
    url = f"https://oare.byu.edu/api/v2/text_epigraphies/text/{oare_id}?forceAllowAdminView=false"
    headers = {
        "accept": "application/json",
        "user-agent": "Mozilla/5.0"
    }
    try:
        response = requests.get(url, headers=headers, timeout=15)
        if response.status_code == 200:
            return response.json(), 200
        else:
            return None, response.status_code
    except Exception as e:
        print(f"Error fetching {oare_id}: {e}")
        return None, None

def main():
    parser = argparse.ArgumentParser(description="Fetch OARE epigraphies.")
    parser.add_argument("--input", default="workspace/published_texts.csv", help="Input CSV path")
    parser.add_argument("--output", default="workspace/oare_epigraphies.jsonl", help="Output JSONL path")
    parser.add_argument("--errors", default="workspace/oare_fetch_errors.csv", help="Errors CSV path")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between requests")
    parser.add_argument("--resume-line", type=int, default=1, help="Row number to start from (1-indexed, header is 1)")
    
    args = parser.parse_args()
    
    # Determine the number of rows to skip (if resume-line is 2, skip 1 row)
    skip_count = max(0, args.resume_line - 2) if args.resume_line > 1 else 0

    # Ensure error file has header if starting fresh
    if args.resume_line <= 1 and not os.path.exists(args.errors):
        with open(args.errors, 'w', encoding='utf-8') as ef:
            ef.write("oare_id,status_code,timestamp\n")

    consecutive_4xx = 0

    with open(args.input, 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        
        # Manually skip rows based on argument
        for _ in range(skip_count):
            try:
                next(reader)
            except StopIteration:
                break

        # Strictly appending to avoid loading existing files into memory
        with open(args.output, 'a', encoding='utf-8') as outfile, \
             open(args.errors, 'a', encoding='utf-8') as errorfile:
            
            error_writer = csv.writer(errorfile)
            
            for i, row in enumerate(reader, start=args.resume_line if args.resume_line > 1 else 2):
                oare_id = row.get('oare_id')
                if not oare_id:
                    continue
                
                print(f"[{i}/8000] Fetching {oare_id}...")
                data, status = fetch_epigraphy(oare_id)
                
                if status == 200 and data:
                    data['oare_id'] = oare_id
                    outfile.write(json.dumps(data, ensure_ascii=False) + '\n')
                    outfile.flush()
                    consecutive_4xx = 0
                else:
                    # Log the error for later retry
                    error_writer.writerow([oare_id, status, time.strftime('%Y-%m-%d %H:%M:%S')])
                    errorfile.flush()
                    
                    if status and 400 <= status < 500:
                        consecutive_4xx += 1
                        if consecutive_4xx >= 3:
                            print(f"Halted after 3 consecutive 4xx errors at line {i}.")
                            break
                    else:
                        consecutive_4xx = 0
                
                time.sleep(args.delay)

if __name__ == "__main__":
    main()
