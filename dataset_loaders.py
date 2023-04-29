import pandas as pd
from pathlib import Path


def load_DIAL2MSA(split, dialect):
    assert split in ["train", "test"]
    assert dialect in ["EGY", "MGR", "LEV", "GLF"]

    BASE_DIR = "data/DIAL2MSA/"
    filename = str(Path(BASE_DIR, f"{split}_{dialect}.tsv"))

    df = pd.read_csv(filename, sep="\t")
    return df


def load_BIBLE(split, dialect):
    assert split in ["train", "test"]
    assert dialect in ["tn", "ma"]

    BASE_DIR = "data/Bible/"
    filename = str(Path(BASE_DIR, f"{split}.tsv"))

    df = pd.read_csv(filename, sep="\t")
    df.rename(columns={"msa": "MSA_text", dialect: "DA_text"}, inplace=True)
    return df[["DA_text", "MSA_text"]]


def load_AOC(split, source):
    assert split in ["train", "dev", "test"]
    assert source in ["youm7_c", "alghad_c", "alriyadh_c"]

    BASE_DIR = "data/AOC/"
    filename = str(Path(BASE_DIR, f"{split}_{source}.tsv"))

    df = pd.read_csv(filename, sep="\t")
    return df


if __name__ == "__main__":
    # df = load_DIAL2MSA("test", "EGY")
    # print(df.head())

    # df = load_BIBLE("test", "tn")
    # print(df.head())

    df = load_AOC("train", "youm7_c")
    print(df.head())
