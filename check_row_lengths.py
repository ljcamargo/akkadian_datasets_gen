#!/usr/bin/env python3
from statistics import mean, stdev

def analyze_csv(filepath):
    row_lengths = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            row_lengths.append(len(line.rstrip('\n')))
    
    if not row_lengths:
        return None
    
    return {
        'count': len(row_lengths),
        'avg': mean(row_lengths),
        'min': min(row_lengths),
        'max': max(row_lengths),
        'stdev': stdev(row_lengths) if len(row_lengths) > 1 else 0
    }

# Analyze both files
files = ['workspace/outputs/finetune.csv', 'workspace/outputs/pretrain.csv']

for filepath in files:
    stats = analyze_csv(filepath)
    if stats:
        print(f"\n{filepath}")
        print(f"  Rows: {stats['count']}")
        print(f"  Avg length: {stats['avg']:.2f}")
        print(f"  Min length: {stats['min']}")
        print(f"  Max length: {stats['max']}")
        print(f"  Std dev: {stats['stdev']:.2f}")
