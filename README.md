# Arabic Level of Dialectness (ALDi)
[![Pylint](https://github.com/AMR-KELEG/Arabic-Formality/actions/workflows/pylint.yml/badge.svg)](https://github.com/AMR-KELEG/Arabic-Formality/actions/workflows/pylint.yml)

## Data

The following scripts download the datasets from their respective sources, then applies the preprocessing steps described in the paper, generating `.tsv` data files to the `data/` directory.
```
# Create the splits for the AOC-ALDi dataset
python prepare_AOC.py

# Form parallel corpora files
python prepare_DIAL2MSA.py
python prepare_bible.py
```

## Installation
For using the DI model of `camel_tools`, run `camel_data -i defaults`

## Fine-tuning the Sentence-ALDi model
```
# Fine-tune the model
MODEL_NAME="UBC-NLP/MARBERT"
# Set the ID of the GPU device
CUDA_ID="1"

CUDA_VISIBLE_DEVICES="$CUDA_ID" python finetune_BERT_models.py train -d "data/AOC/train*" --dev "data/AOC/dev*"  -m "$MODEL_NAME" -o ALDi_model
```

## Technical information
The models and experiments were run on a single Quadro RTX 8000 GPU with 48GB of VRAM.

