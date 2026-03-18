import csv
import json
import os
import statistics
from collections import Counter
from corpus_utils import standardize_orthography, replace_gaps

def generate_lexicostats():
    input_file = "workspace/train.csv"
    input_file_2 = "workspace/outputs/published_texts/texts_pretrain.csv"
    output_dir = "workspace/outputs/train"
    samples_file = "workspace/outputs/lexicostats_samples.txt"
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, "lexicostats.json")
    
    char_counter = Counter()
    word_counter = Counter()
    syllable_counter = Counter()
    prefix_counter = Counter()
    postfix_counter = Counter()
    
    word_lengths = []
    syllable_counts = []
    
    print(f"Reading {input_file} to generate lexicostatistics...")
    all_texts = []
    
    def process_file(filepath):
        """Process a CSV file and update counters."""
        with open(filepath, "r", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            for row in reader:
                translit = row.get("transliteration", "").strip()
                if not translit:
                    continue
                translit = replace_gaps(translit)
                translit = standardize_orthography(translit)
                all_texts.append(translit)
                translit = translit.replace("<gap>", " ")

                    
                # 1. Character frequency (excluding spaces)
                for char in translit:
                    if not char.isspace():
                        char_counter[char] += 1
                        
                # 2. Word frequency
                words = translit.split()
                for word in words:
                    word_counter[word] += 1
                    word_lengths.append(len(word))
                    
                    # 3. Syllable frequency (split by '-')
                    syllables = word.split('-')
                    num_syllables = len(syllables)
                    syllable_counts.append(num_syllables)
                    
                    for syl in syllables:
                        if syl:
                            syllable_counter[syl] += 1
                            
                    # 4. Prefix and Postfix frequency (words with > 1 syllable)
                    if num_syllables > 1:
                        prefix = syllables[0]
                        postfix = syllables[-1]
                        if prefix:
                            prefix_counter[prefix] += 1
                        if postfix:
                            postfix_counter[postfix] += 1
    
    # Process both input files
    process_file(input_file)
    print(f"Reading {input_file_2} to generate lexicostatistics...")
    process_file(input_file_2)

    # Save a sample of processed texts for reference
    with open(samples_file, "w", encoding="utf-8") as f:
        for text in all_texts:
            f.write(text + "\n")
             
    def compute_stats(data_list):
        if not data_list:
            return {"mean": 0, "min": 0, "max": 0, "std": 0}
        return {
            "mean": sum(data_list) / len(data_list),
            "min": min(data_list),
            "max": max(data_list),
            "std": statistics.stdev(data_list) if len(data_list) > 1 else 0.0
        }
                        
    stats = {
        "metadata": {
            "total_character_volume": sum(char_counter.values()),
            "total_words": sum(word_counter.values()),
            "word_length_stats": compute_stats(word_lengths),
            "syllable_count_stats": compute_stats(syllable_counts)
        },
        "character_frequency": dict(char_counter.most_common(200)),
        "word_frequency": dict(word_counter.most_common(1000)),
        "syllable_frequency": dict(syllable_counter.most_common(1000)),
        "prefix_frequency": dict(prefix_counter.most_common(500)),
        "postfix_frequency": dict(postfix_counter.most_common(500)),
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
        
    print(f"Lexicostatistics generated and saved to {output_file}")

if __name__ == "__main__":
    generate_lexicostats()
