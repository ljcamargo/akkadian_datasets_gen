# Akkadian LLM Dataset Processing Pipeline

This repository contains the data processing pipeline for transforming raw Akkadian cuneiform datasets into highly structured, deduplicated formats optimal for Large Language Model (LLM) fine-tuning and pre-training.

## Output Structure: Pre-train vs. Fine-tune

The pipeline generates two distinct types of `.csv` datasets, each serving a specific phase of LLM training:

- **`_pretrain.csv`**: Contains raw, uninstructed data pairs or serialized property mappings. This data is designed for the initial continuous pre-training phase, allowing the LLM to learn the fundamental structure, vocabulary, and grammar of Akkadian without the overhead of task-specific instruction framing.
- **`_finetune.csv`**: Contains data structured into explicit instruction-input-output formats (e.g., `<user> Translate this to English: <word> \n <model> <translation>`). This dataset is crucial for the supervised fine-tuning phase, teaching the model how to respond to specific user prompts (like translation, morphological parsing, and lemmatization) consistently.

## System Architecture & Workflows

The system currently processes data from two major sources: a corpus of "Published Texts" and a "Dictionary" database. Both are independently processed into a homologated `dictionary_parsed.jsonl` schema and subsequently output as task-specific `.csv` datasets (e.g., translation, grammar parsing, lemma identification).

---

### 1. Data Ingestion Pipeline
**Script:** `fetch_oare_epigraphies.py`
**Purpose:** Fetches raw epigraphic textual data from the Open Access Repository for the Epigraphy of the Ancient Near East (OARE) API. It handles pagination, robust error logging, and rate limiting to systematically download the corpus.

**Sample Command:**
```bash
python3 fetch_oare_epigraphies.py --output workspace/oare_epigraphies.jsonl --delay 0.5
```
**Arguments:**
- `--output`: Path to save the downloaded JSONL items.
- `--input`: Path to an existing file to append to (if resuming).
- `--errors`: Path for logging failed fetches.
- `--delay`: Sleep duration between API requests to avoid rate limits (default: 0.1s).
- `--resume-line`: Start fetching from a specific line/offset if the script was interrupted.

**Expected Outputs:**
- `workspace/oare_epigraphies.jsonl`: Raw nested JSON items representing the epigraphic database.

**Caveats:**
- API rate limits apply; avoid setting `--delay` to 0. Handling 4xx errors is automated via the error log.

---

### 2. The `published_texts` Pipeline
**Script:** `process_published_texts.py`
**Purpose:** Processes the raw `oare_epigraphies.jsonl` file to generate `_finetune.csv` and `_pretrain.csv` datasets representing instructions and data for transliteration modes (Epigraphic, Compact Epigraphic, Akkadian Orthography), grammar analysis, meaning, lemmatization, and Rosetta alignment tables.

**Sample Command:**
```bash
python3 process_published_texts.py --input workspace/oare_epigraphies.jsonl --start 1 --end 5000
```
**Arguments:**
- `--input`: Path to the input `.jsonl` file. Defaults to `workspace/oare_epigraphies.jsonl`.
- `--start`: Optional line index to start processing.
- `--end`: Optional line index to stop processing (useful for testing).

**Expected Outputs:**
- `workspace/outputs/published_texts/*_finetune.csv` and `*_pretrain.csv` (e.g., `translations_finetune.csv`, `grammar_pretrain.csv`).
- `workspace/outputs/published_texts/dictionary_parsed.jsonl`: A dynamically generated JSON representation matching the strict structure defined by the dictionary pipeline.

**Caveats:**
- The script uses SQLite to deduplicate entries globally based on md5 hashing. 
- Transliteration units are strictly sequential, which means `big_gap` formatting traits may inherently disappear if not actively preserved.

---

### 3. The `dictionary` Pipeline
This relies on three distinct scripts to address the complex irregularities of `eBL_Dictionary.csv`.

#### Step 3a: Pattern Extraction
**Script:** `extract_dictionary_patterns.py`
**Purpose:** Applies regex heuristics to decompose complex dictionary CSV strings. Strips Roman numeral disambiguators, cleanly splits multiple definition blocks (`;` or `,`), and cleanly translates abbreviated grammar variables (`"3 m. sg. acc. suff."`) into pure JSON mappings (`{"person": "third", "gender": "masculine", "number": "singular", "case": "accusative", "suffix": true}`).
**Sample Command:**
```bash
python3 extract_dictionary_patterns.py
```
**Expected Outputs:**
- `workspace/outputs/dictionary/dictionary_parsed.jsonl` 
- Features the `special` boolean flag. If the regex falls short on a difficult entry, it flags `"special": true`.

#### Step 3b: LLM Intervention
**Script:** `fill_special_dictionary.py`
**Purpose:** Processes all rows in `dictionary_parsed.jsonl` marked `"special": true` using the Gemini API. The LLM coerces unstructured dictionary edge-cases to match the clean JSON nested-array schema, and then marks them `"special": false`.
**Sample Command:**
```bash
export GEMINI_API_KEY="your-api-key"
python3 fill_special_dictionary.py
```
**Caveats:**
- Requires export of an active `GEMINI_API_KEY`. Without it, the script will halt.

#### Step 3c: Assembly
**Script:** `generate_dictionary_csvs.py`
**Purpose:** Reads the cleaned `dictionary_parsed.jsonl` and compiles it into discrete `_finetune.csv` and `_pretrain.csv` targets matching the outputs from the published texts pipeline.
**Sample Command:**
```bash
python3 generate_dictionary_csvs.py
```
**Caveats:** 
- Output datasets correctly generate mirrored English-to-Akkadian inverse instructions (translating English word definitions back to Akkadian form queries).

---

### 4. Core Engine & Utilities
**Script:** `corpus_utils.py`
**Purpose:** A centralized utility holding variables, textual translation cleaners (such as parenthesis-strippers `clean_translation()`), `PROMPT_*` string templates, and the internal `Deduplicator` class via SQLite. Modifying a prompt here universally updates both the Dictionary and Published Texts pipelines natively.

**Script:** `dump_grammar.py`
**Purpose:** Sub-tool designed to rip all isolated key/value mappings of `"parseInfo"` tags out of `oare_epigraphies.jsonl` and export them to `grammar_keys_dump.json`.

---

### 5. CSV Output Catalogue

Both pipelines output to their respective folders (`workspace/outputs/published_texts/` and `workspace/outputs/dictionary/`). Commonalities include:

- `translations_finetune.csv` & `translations_pretrain.csv`: Bidirectional instructions and raw alignments pairing English translations with Akkadian modes (Epigraphic, Orthography, etc.).
- `grammar_finetune.csv` & `grammar_pretrain.csv`: Data specifying the morphological parsing of individual words (e.g., Identifying gender, case, person).
- `meanings_finetune.csv` (or `meaning_finetune.csv`): Explicit translation definition queries mapped between Akkadian form and English meaning.
- `lemma_finetune.csv`: Instructions identifying an inflected form to its base dictionary lemma.
- `rosetta_pretrain.csv`: A combined horizontal mapping table aligning the various transliteration and translation strings across the exact same semantic unit.
- `transforms_finetune.csv` (Published Texts only): Specifically models text transforms *between* Akkadian formatting structures (e.g., converting Compact Epigraphic format to full Orthography).
- `texts_pretrain.csv` (Published Texts only): Contiguous chunks of raw Akkadian string datasets.
- `dictionary_pretrain.csv` (Dictionary only): Master baseline export of the dictionary JSON schemas mapped into serialized strings.

---

## Development Journal (Pipeline Homologation and Progress)

We've achieved full integration and homologation of the `published_texts` and `dictionary` pipelines. Both pipelines now inherently construct and enforce the same logical structure, ultimately generating compatible `dictionary_parsed.jsonl` files and homogeneous `finetune`/`pretrain` CSVs.

1. **Centralized Engine Abstraction:** Extracted shared functions, deduplication mechanics, and prompt text constants into the universal `corpus_utils.py` library.
2. **Dictionary Pipeline Implementation:** Designed a heuristic-and-LLM approach to parse unstructured strings in `eBL_Dictionary.csv`. 
3. **Regex Expansion:** Refined extraction routines to avoid disambiguators, parse compound semantic meanings accurately into explicitly arrayed elements, and interpret abbreviated morphological strings.
4. **Grammar Object Parsing Framework:** Converted flat string grammar tags into explicitly defined JSON structures (Mapping `"3 m. sg."` to `{"person": "third", "gender": "masculine", "number": "singular"}`).
5. **Double-Sided Finetuning Logic:** Added functionality enabling bidirectional prompt instruction parsing. Moving from Akkadian form parsing to generating the required reverse english-to-akkadian translation requests.
6. **Schema Homologation & System Unification:** Integrated the `dictionary` schema directly into the `process_published_texts.py` architecture. The `parseInfo` components observed across both corpora structures are completely unified; variable names (`grammatical_number` to `number`) and values (`third` instead of `3`) are homologated.
7. **`fetch_oare_epigraphies.py` documentation added** along with distinct clarifications for `pretrain` vs `finetune` methodologies. Both independent pipelines are now fully aligned and functionally capable of being merged into a singular Master Akkadian Model Training dataset.
