#!/usr/bin/env python3
from statistics import mean, stdev

TOKEN_RATIO = 2.33

def analyze_csv(filepath):
    row_lengths = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            row_lengths.append(len(line.rstrip('\n')))
    
    if not row_lengths:
        return None
    
    std = stdev(row_lengths) if len(row_lengths) > 1 else 0
    upper_sigma3 = (3 * std) + mean(row_lengths)
    
    return {
        'count': len(row_lengths),
        'avg': mean(row_lengths),
        'min': min(row_lengths),
        'max': max(row_lengths),
        'stdev': std,
        'upper_sigma3': upper_sigma3
    }

# Analyze both files
files = ['workspace/outputs/finetune.csv', 'workspace/outputs/pretrain.csv']

for filepath in files:
    stats = analyze_csv(filepath)
    if stats:
        print(f"\n{filepath}")
        print(f"  Rows: {stats['count']}")
        print(f"  Avg length: {stats['avg']:.2f} ({stats['avg'] / TOKEN_RATIO:.2f})")
        print(f"  Min length: {stats['min']} ({stats['min'] / TOKEN_RATIO:.2f})")
        print(f"  Max length: {stats['max']} ({stats['max'] / TOKEN_RATIO:.2f})")
        print(f"  Std dev: {stats['stdev']:.2f} ({stats['stdev'] / TOKEN_RATIO:.2f})")
        print(f"  Upper sigma3: {stats['upper_sigma3']:.2f} ({stats['upper_sigma3'] / TOKEN_RATIO:.2f})")
