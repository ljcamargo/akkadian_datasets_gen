#!/usr/bin/env python3
"""
Deep merge two JSONL dictionary files into a single JSON file.
Groups entries by 'word' and enriches lists, warns on non-list collisions.
"""

import json
from pathlib import Path
from collections import defaultdict

DICT_PATH = Path("workspace/outputs/dictionary/dictionary_parsed.jsonl")
PUBTEXTS_PATH = Path("workspace/outputs/published_texts/dictionary_parsed.jsonl")
OUTPUT_PATH = Path("workspace/outputs/final_dictionary.json")


def load_jsonl(filepath):
    """Load JSONL file and return list of parsed objects."""
    entries = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def deep_merge(dict1, dict2, word):
    """
    Deep merge two dictionaries with the same 'word'.
    Lists are combined; non-list collisions produce warnings.
    """
    merged = dict1.copy()
    
    for key, value2 in dict2.items():
        if key not in merged:
            merged[key] = value2
        else:
            value1 = merged[key]
            
            # If both are lists, extend (avoiding exact duplicates)
            if isinstance(value1, list) and isinstance(value2, list):
                # Add items from value2 that aren't already in value1
                for item in value2:
                    if item not in value1:
                        value1.append(item)
            # If one is a list and the other isn't, handle specially
            elif isinstance(value1, list) or isinstance(value2, list):
                # Just log and keep first value
                if value1 != value2:
                    print(f"⚠️  Warning for word '{word}': {key} mismatch (one is list, one isn't). Keeping first.")
            # Both are non-lists
            else:
                if value1 != value2:
                    print(f"⚠️  Warning for word '{word}': {key} collision: {value1!r} vs {value2!r}. Keeping first.")
    
    return merged


def main():
    print("Loading JSONL files...")
    dict_entries = load_jsonl(DICT_PATH)
    pubtexts_entries = load_jsonl(PUBTEXTS_PATH)
    
    print(f"  Dictionary entries: {len(dict_entries)}")
    print(f"  Published texts entries: {len(pubtexts_entries)}")
    
    # Group by word
    by_word = defaultdict(list)
    
    for entry in dict_entries:
        word = entry.get('word')
        by_word[word].append(('dict', entry))
    
    for entry in pubtexts_entries:
        word = entry.get('word')
        by_word[word].append(('pubtexts', entry))
    
    # Merge and build final dictionary
    final_dict = {}
    duplicates_count = 0
    
    for word in sorted(by_word.keys()):
        entries_list = by_word[word]
        
        if len(entries_list) == 1:
            # No conflict, just use the entry
            _, entry = entries_list[0]
            final_dict[word] = entry
        else:
            # Multiple sources for this word
            duplicates_count += 1
            merged = entries_list[0][1].copy()
            
            for source, entry in entries_list[1:]:
                merged = deep_merge(merged, entry, word)
            
            final_dict[word] = merged
    
    # Write output
    print(f"\nMerging complete:")
    print(f"  Total unique words: {len(final_dict)}")
    print(f"  Words with conflicts (merged): {duplicates_count}")
    
    print(f"\nWriting to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
        json.dump(final_dict, f, ensure_ascii=False, indent=2)
    
    print("✓ Done!")


if __name__ == '__main__':
    main()
