# Is this Dialectal?
[![Pylint](https://github.com/AMR-KELEG/Arabic-Formality/actions/workflows/pylint.yml/badge.svg)](https://github.com/AMR-KELEG/Arabic-Formality/actions/workflows/pylint.yml)

```
.
├── backtranslate.py: Generate backtranslated versions of an Arabic text
├── data/: Directory of dataset files
├── prepare_AOC.py: Transform the AOC annotation file into single-annotation rows
├── prepare_DIAL2MSA.py: Filter out samples with non-perfect confidence from DIAL2MSA
├── prepare_bible.py: Merge the bible MSA/TUN/MOR translations
├── finetune_BERT_models.py: Fine-tune a regression head on top of a BERT model and load it to make predictions on a test set
├── run_dialectness_score_experiment.py: Evalute an ALDi method on one of the selected corpora
└── requirements.txt: List of pythonckages
```
