import pandas as pd
from utils import commasize_number


def generate_stats(df, split="Train"):
    d = df["source"].value_counts(dropna=False).to_dict()
    total_samples = df.shape[0]

    sources = ["alghad_c", "alghad_a", "alriyadh_c", "alriyadh_a", "youm7_c", "youm7_a"]
    print("split &", " & ".join(sources))
    table_line = f"\\textbf{{{split.capitalize()}}}"
    for source in sources:
        table_line += f" & {commasize_number(d[source])}"
    table_line += " \\\\"
    print(table_line)
    return d


if __name__ == "__main__":
    for split in ["train", "dev", "test"]:
        generate_stats(pd.read_csv(f"../data/AOC/{split}.tsv", sep="\t"), split=split)
