<img src="assets/ALDi_logo.svg" alt="ALDi logo">

The codebase for the **ALDi: Quantifying the Arabic Level of Dialectness of Text** paper.

## Dependencies
* Create a conda env:
```
conda create -n ALDi python=3.9.16
```

* Activate the environment, and install the dependencies:
```
conda activate ALDi
pip install -r requirements.txt
```

* For using the Dialect Identification model of `camel_tools`:
```
camel_data -i defaults
```

## Data

The following scripts download the datasets from their respective sources, then applies the preprocessing steps described in the paper, generating `.tsv` data files to the `data/` directory.
* Create the splits for the AOC-ALDi dataset:
```
python prepare_AOC.py
```

* Form parallel corpora files:
```
python prepare_DIAL2MSA.py
python prepare_bible.py
```

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

