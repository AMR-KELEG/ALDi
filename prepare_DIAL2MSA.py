import os
import logging
import zipfile
import pandas as pd
from pathlib import Path
from urllib.request import urlretrieve

logging.basicConfig(format="%(asctime)s - %(message)s", level=logging.INFO)

BASE_DATASET_DIR = "data/DIAL2MSA"
DATASET_URL = "https://alt.qcri.org/~hmubarak/EGY-MGR-LEV-GLF-2-MSA.zip"
DISTINCTIVE_TERMS_URL = "https://alt.qcri.org/~hmubarak/EGY-MGR-LEV-GLF-StrongWords.zip"
DIALECT_ABBREV_TO_NAME_MAP = {
    "EGY": "Egyptian",
    "GLF": "Gulf",
    "LEV": "Levantine",
    "MGR": "Maghrebi",
}


def download_DIAL2MSA():
    # Create the dataset directory
    os.makedirs(BASE_DATASET_DIR, exist_ok=True)

    # Retrieve the dataset
    dataset_filename = str(Path(BASE_DATASET_DIR, DATASET_URL.split("/")[-1]))
    urlretrieve(DATASET_URL, dataset_filename)
    with zipfile.ZipFile(dataset_filename, "r") as zip_ref:
        zip_ref.extractall(BASE_DATASET_DIR)

    distinctive_terms_filename = str(
        Path(BASE_DATASET_DIR, DISTINCTIVE_TERMS_URL.split("/")[-1])
    )
    urlretrieve(DISTINCTIVE_TERMS_URL, distinctive_terms_filename)
    with zipfile.ZipFile(distinctive_terms_filename, "r") as zip_ref:
        zip_ref.extractall(BASE_DATASET_DIR)


def load_distinctive_terms(dialect):
    assert dialect in DIALECT_ABBREV_TO_NAME_MAP.values()
    words_list_filename = str(Path(BASE_DATASET_DIR, f"{dialect}StrongWords.txt"))
    with open(words_list_filename, "r") as f:
        WORDS = set([l.strip() for l in f])
    return WORDS


def load_DIAL2MSA_dataset(dialect_abbrev):
    assert dialect_abbrev in DIALECT_ABBREV_TO_NAME_MAP.keys()
    try:
        filename = next(Path(BASE_DATASET_DIR).glob(f"{dialect_abbrev}2MSA*.xlsx"))
    except:
        filename = next(Path(BASE_DATASET_DIR).glob(f"{dialect_abbrev}2MSA*.xls"))

    df = pd.read_excel(filename)

    dialect_text_column = "cleanedtweet" if "cleanedtweet" in df.columns else "tweet"
    msa_text_column = "msa"

    logging.info(f"Size of the original '{dialect_abbrev}' dataset is: {df.shape[0]}")

    confidence_column = f"{dialect_abbrev.lower()}tomsa:confidence"
    if confidence_column in df:
        df = df[df[confidence_column] == 1.0].copy()
        logging.info(
            f"Size of '{dialect_abbrev}' dataset after dropping samples of confidence != 1.0 is: {df.shape[0]}"
        )

    dialect_name = DIALECT_ABBREV_TO_NAME_MAP[dialect_abbrev]
    DIALECT_DISTINCTIVE_TERMS = load_distinctive_terms(dialect_name)

    df = df[
        ~df[msa_text_column].apply(
            lambda s: any([w in DIALECT_DISTINCTIVE_TERMS for w in str(s).split()])
        )
    ]
    logging.info(
        f"Size of '{dialect_abbrev}' dataset after dropping translations having a distinctive term is: {df.shape[0]}"
    )

    # Note: MSA transltions within the dataset are not too long (< 50 tokens) making them adequate sentences

    df.rename(
        columns={msa_text_column: "MSA_text", dialect_text_column: "DA_text"},
        inplace=True,
    )

    # TODO: Try to keep translations of the same tweet in the same split

    df[["DA_text", "MSA_text"]].to_csv(
        Path(BASE_DATASET_DIR, f"{dialect_abbrev}.tsv"), index=False, sep="\t"
    )


def main():
    # TODO: The levantine and Gulf xls files have issues
    # The current solution is to open the file with LibreOffice and export it as xlsx
    download_DIAL2MSA()

    for dialect in ["EGY", "MGR"]:
        load_DIAL2MSA_dataset(dialect)


if __name__ == "__main__":
    main()
