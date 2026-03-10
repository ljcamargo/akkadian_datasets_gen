## Progress
### published_texts.csv
- Status: DONE need rev
- Converted to finetune and pretrain
- Missing check replacement convetion for scripting and tags like <big_gap> PERSONAL NOTE: It seems the algorithm correctly splits into sequential units so gaps are removed inherently but need to check for "masking features while trainin

# Datasets
## Parsable
- published_texts.csv (DONE)
- eBL_Dictionary.csv , (DONE) PARSED into structure and finetune/pretrain
- OA_Lexicon_eBL.csv , (DONE) need to parse into TABLE, duplicates from pubished_texts?
- publications.csv , huge list of docs, parse into MD, possible extract Assyrian (??)
- Sentences_Oare_FirstWorld_LinNum.csv, ??? 
- train.csv, (DONE) finetune and possibly pretrain with MAPS

## For Crawling
- bibliography.csv
- resources.csv

## Other:
- test.csv
- sample_submission.csv


# Current Dataset Stats:
## SIZE:
710M  total -> 641M
## LINES:
216603 workspace/outputs/publications/publications_pretrain.csv
960078 workspace/outputs/train/reasoned_translations_finetune.csv
3123 workspace/outputs/train/translations_finetune.csv
1562 workspace/outputs/train/translations_pretrain.csv
6353 workspace/outputs/lexicon/lemma_pretrain.csv
92299 workspace/outputs/lexicon/lemma_finetune.csv
1929 workspace/outputs/lexicon/rosetta_pretrain.csv
53676 workspace/outputs/published_texts/grammar_finetune.csv
86723 workspace/outputs/published_texts/transforms_finetune.csv
53676 workspace/outputs/published_texts/grammar_pretrain.csv
248739 workspace/outputs/published_texts/translations_finetune.csv
6217 workspace/outputs/published_texts/translations_pretrain.csv
21921 workspace/outputs/published_texts/texts_pretrain.csv
56832 workspace/outputs/published_texts/lemma_finetune.csv
1120 workspace/outputs/published_texts/rosetta_pretrain.csv
61619 workspace/outputs/published_texts/meanings_finetune.csv
27416 workspace/outputs/dictionary/dictionary_pretrain.csv
195 workspace/outputs/dictionary/grammar_finetune.csv
54691 workspace/outputs/dictionary/translations_finetune.csv
7478 workspace/outputs/dictionary/lemma_finetune.csv
1434 workspace/outputs/dictionary/rosetta_pretrain.csv
27382 workspace/outputs/dictionary/meaning_finetune.csv
1991066 total

then 

216603 workspace/outputs/publications/publications_pretrain.csv
960078 workspace/outputs/train/reasoned_translations_finetune.csv
3123 workspace/outputs/train/translations_finetune.csv
1562 workspace/outputs/train/translations_pretrain.csv
6353 workspace/outputs/lexicon/lemma_pretrain.csv
56088 workspace/outputs/lexicon/lemma_finetune.csv
1929 workspace/outputs/lexicon/rosetta_pretrain.csv
20412 workspace/outputs/published_texts/grammar_finetune.csv
43555 workspace/outputs/published_texts/transforms_finetune.csv
20412 workspace/outputs/published_texts/grammar_pretrain.csv
91439 workspace/outputs/published_texts/translations_finetune.csv
2285 workspace/outputs/published_texts/translations_pretrain.csv
7309 workspace/outputs/published_texts/texts_pretrain.csv
21442 workspace/outputs/published_texts/lemma_finetune.csv
1120 workspace/outputs/published_texts/rosetta_pretrain.csv
26270 workspace/outputs/published_texts/meanings_finetune.csv
27416 workspace/outputs/dictionary/dictionary_pretrain.csv
195 workspace/outputs/dictionary/grammar_finetune.csv
54691 workspace/outputs/dictionary/translations_finetune.csv
7478 workspace/outputs/dictionary/lemma_finetune.csv
1434 workspace/outputs/dictionary/rosetta_pretrain.csv
27382 workspace/outputs/dictionary/meaning_finetune.csv
1598576 total
