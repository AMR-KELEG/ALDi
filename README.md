<img src="assets/ALDi_logo.svg" alt="ALDi logo">

[![Huggingface Space](https://img.shields.io/badge/🤗-Demo%20-yellow.svg)](https://huggingface.co/spaces/AMR-KELEG/ALDi)
[![arXiv](https://img.shields.io/badge/arXiv-coming_soon-00ff00.svg)](https://arxiv.org/abs/coming-soon)

The codebase for the **ALDi: Quantifying the Arabic Level of Dialectness of Text** paper accepted to [EMNLP 2023](https://2023.emnlp.org/).

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

The following scripts download the datasets from their respective sources, then apply the preprocessing steps described in the paper, generating `.tsv` data files to the `data/` directory.
* Create the splits for the AOC-ALDi dataset:
```
python prepare_AOC.py
```

* Form parallel corpora files:
```
python prepare_DIAL2MSA.py
python prepare_bible.py
```

* For generating the MSA lexicon used for the baseline model, download the [UN proceedings](https://conferences.unite.un.org/UNCorpus/Home/DownloadOverview) into `data/MSA_raw_corpora/`

    * Create the directory:
    ```mkdir -p data/MSA_raw_corpora/```
    * Download the tar file parts into the directory following these urls:
        * https://conferences.unite.un.org/UNCorpus/Home/DownloadFile?filename=UNv1.0.ar-en.tar.gz.00
        * https://conferences.unite.un.org/UNCorpus/Home/DownloadFile?filename=UNv1.0.ar-en.tar.gz.01
    * Extract the downloaded tar file:
    ```cd data/MSA_raw_corpora/ && cat UNv1.0.ar-en.tar.gz.0* | tar -xzv && mv ar-en/ UN-en-ar/```

* To fine-tune the token DI model, download the "Modern Standard Arabic - Egyptian Arabic (MSA - EA)" data file from [LinCE](https://ritual.uh.edu/lince/datasets) into `data/LinCE`
    * Create the directory:
    ```mkdir -p data/LinCE/```
    * Download the data zip file (named `lid_msaea.zip`) into the directory.
    * Decompress the file into the directory: `unzip lid_msaea.zip`
        * Expected structure:
        ```
        data/LinCE
        └── lid_msaea
            ├── dev.conll
            ├── test.conll
            └── train.conll
        ```

## Models

### Sentence-ALDi model
* Fine-tuning the Sentence-ALDi model
```
# Set the ID of the GPU device
CUDA_ID="1"

# Fine-tune the model
SEED="42"
MODEL_NAME="UBC-NLP/MARBERT"
CUDA_VISIBLE_DEVICES="$CUDA_ID" python finetune_BERT_models.py train --train data/AOC/train.tsv --dev data/AOC/dev.tsv -model_name "$MODEL_NAME" -o Sentence_ALDi -s "$SEED"
```

### Baseline models
* Building the MSA lexicon baseline (generates a pkl file to `data/MSA_raw_corpora`)
```
python form_msa_lexicon.py form_lexicon -c UN
```

* Fine-tuning the Token-DI model
```
# Set the ID of the GPU device
CUDA_ID="1"

# Fine-tune the model
SEED="42"
CUDA_VISIBLE_DEVICES="$CUDA_ID" python finetune_BERT_for_tagging.py -s "$SEED" -o TOKEN_DI
```

## Technical information
The models and experiments were run on a single Quadro RTX 8000 GPU with 48GB of VRAM.

