#!/usr/bin/env bash

# Helper script to sequentially run all Akkadian pre-processing pipelines

set -e

echo "Starting Akkadian pipeline execution..."

echo "==> Running process_published_texts.py"
python3 process_published_texts.py

echo "==> Running process_dictionaries.py"
python3 process_dictionaries.py

echo "==> Running process_publications.py"
python3 process_publications.py

echo "==> Running process_lexicon.py"
python3 process_lexicon.py

echo "==> Running process_train.py"
python3 process_train.py

echo "==> Merge dictionaries"
python3 merge_dictionaries.py

echo "==> Running process_reasoned_translations.py"
python3 process_reasoned_translations.py

echo "==> Running merge_csvs.py"
python3 merge_csvs.py

echo "All pipelines executed successfully."
