import pandas as pd
from pathlib import Path


def load_contrastive_pairs():
    """Load the dataset of contrastive DA/MSA pairs."""
    df = pd.read_csv("data/contrastive_pairs.tsv", sep="\t")
    df.rename(
        {
            "MSA": "MSA_text",
            "DA": "DA_text",
            "English": "English_text",
            "Sample ID": "ID",
            "Gender": "Gender",
        },
        axis=1,
        inplace=True,
    )
    return df[
        [
            "ID",
            "Feature name",
            "MSA_text",
            "DA_text",
            "Word order",
            "Gender",
            "English_text",
        ]
    ].dropna()


def load_DIAL2MSA(dialect):
    assert dialect in ["EGY", "MGR", "LEV", "GLF"]

    BASE_DIR = "data/DIAL2MSA/"
    filename = str(Path(BASE_DIR, f"{dialect}.tsv"))

    df = pd.read_csv(filename, sep="\t")
    return df


def load_BIBLE(dialect):
    assert dialect in ["tn", "ma"]

    BASE_DIR = "data/Bible/"
    filename = str(Path(BASE_DIR, f"bible.tsv"))

    df = pd.read_csv(filename, sep="\t")
    df.rename(columns={"msa": "MSA_text", dialect: "DA_text"}, inplace=True)
    return df[["DA_text", "MSA_text"]]


def load_AOC(split, source=None):
    assert split in ["train", "test", "dev"]
    assert source in ["youm7_c", "alghad_c", "alriyadh_c", None]

    BASE_DIR = "data/AOC"
    filename = str(Path(BASE_DIR, f"{split}.tsv"))

    # TODO: Handle only loading samples from a specific source

    df = pd.read_csv(filename, sep="\t")
    return df


def load_LinCE(split):
    assert split in ["train", "dev", "test"]
    filename = f"data/LinCE/lid_msaea/{split}.conll"

    with open(filename, "r") as f:
        data, tokens, labels = [], [], []
        for i, line in enumerate(f):
            # ANERCORP/Lince splits sentences with \n
            if line == "\n":
                if len(tokens) > 0:
                    data.append((tokens, labels))
                    tokens, labels = [], []
                continue

            # Lince has an additional sentence id number
            if line.startswith("# sent_enum ="):
                continue

            try:
                if "\t" in line:
                    splits = line.split("\t")
                else:
                    splits = line.split()
                assert len(splits) == 2
            except:
                print("ERROR", i, repr(line))
                continue

            # Drop whitespace tokens that are tagged as "O" in Lince
            if not splits[0].strip():
                print("ERROR", i, line[:-1])
                continue
            tokens.append(splits[0])
            labels.append(splits[1].strip())

        if len(tokens) > 0:
            data.append((tokens, labels))

    return data


if __name__ == "__main__":
    # df = load_DIAL2MSA("test", "EGY")
    # print(df.head())

    # df = load_BIBLE("test", "tn")
    # print(df.head())

    df = load_AOC("train", "youm7_c")
    print(df.head())
