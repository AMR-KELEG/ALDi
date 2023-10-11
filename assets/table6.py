import math
import pandas as pd


def compute_RMSE(df, source_type=""):
    """Compute the RMSE for the given dataframe.

    Args:
        df: a dataframe with the columns "average_dialectness_level", "DA_score", and "source".
        source_type: a string that is either "a" or "c" to filter the dataframe by source.

    Returns:
        The RMSE of the dataframe.
    """
    df = df[df["source"].apply(lambda s: s.endswith(source_type))]
    gold_standard = df["average_dialectness_level"]
    predictions = df["DA_score"]
    sq_error = (gold_standard - predictions) ** 2
    RMSE = math.sqrt(sq_error.sum() / sq_error.shape[0])
    return round(RMSE, 2)


if __name__ == "__main__":
    for model in ["LEXICON", "SENTENCE_DI", "TOKEN_DI", "REGRESSION"]:
        df = pd.read_csv(f"../AOC_ALDi_RMSE/{model}_AOC.tsv", sep="\t")

        if model == "LEXICON":
            # Print stats about the number of control and comment samples
            n_control = df["source"].apply(lambda s: s.endswith("a")).sum()
            n_comment = df["source"].apply(lambda s: s.endswith("c")).sum()
            print("\t", "Control", "Comment", "All")
            print("\t", n_control, n_comment, df.shape[0], "\n----")

        # Compute RMSE for control, comment, and all samples
        print(
            model,
            compute_RMSE(df, source_type="a"),
            compute_RMSE(df, source_type="c"),
            compute_RMSE(df),
            sep=" & ",
        )
