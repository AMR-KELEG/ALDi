import pandas as pd
from utils import commasize_number


def generate_stats(subset_df, source="Control"):
    d = subset_df["DLevel"].value_counts(dropna=False).to_dict()
    total_samples = subset_df.shape[0]

    labels = ["msa", "little", "mixed", "most", "junk", "missing"]
    labels_map = {"msa": "MSA", "missing": "Missing", "junk": "Not Arabic or symbols"}
    print(
        "Source & "
        + " & ".join(
            [f"\\textbf{{{labels_map.get(l, l.capitalize())}}}" for l in labels]
        )
        + " \\\\"
    )
    table_line = f"{source}"
    for label in labels:
        table_line += f" & {commasize_number(d[label])} ({round(100 * d[label]/total_samples, 2)}\%)"
    table_line += " \\\\"
    print(table_line)
    return d


if __name__ == "__main__":
    df2 = pd.read_csv("../data/AOC/2_AOC_augmented.tsv", sep="\t")

    generate_stats(df2.loc[~df2["is_control_sentence"]], source="Cmnts")
    print()
    generate_stats(df2.loc[df2["is_control_sentence"]], source="Cntrl")
    print()
    generate_stats(df2, source="All")
