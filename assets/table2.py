import pandas as pd

if __name__ == "__main__":
    df = pd.read_csv("../data/AOC/1_AOC_exploded.tsv", sep="\t")
    print(df["source"].value_counts())
