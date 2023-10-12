import pandas as pd
from utils import commasize_number


def generate_stats(df, split="train"):
    d = df["source"].value_counts(dropna=False).to_dict()
    total_samples = df.shape[0]

    sources = ["alghad_c", "alghad_a", "alriyadh_c", "alriyadh_a", "youm7_c", "youm7_a"]

    if split == "train":
        print("split &", " & ".join(sources))
    table_line = f"\\textbf{{{split.capitalize()}}}"
    for source in sources:
        table_line += f" & {commasize_number(d[source])}"
    table_line += " \\\\"
    print(table_line)
    return d


if __name__ == "__main__":
    no_samples = 0
    for split in ["train", "dev", "test"]:
        df = pd.read_csv(f"../data/AOC/{split}.tsv", sep="\t")
        generate_stats(df, split=split)
        no_samples += df.shape[0]
    print("\nTotal number of samples:", no_samples)
