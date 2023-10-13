import math
import statistics
import pandas as pd


def compute_D_statistic(df):
    u1, u2 = df["DA_score"].mean(), df["MSA_score"].mean()
    std1, std2 = df["DA_score"].std(), df["MSA_score"].std()
    return abs(u1 - u2) / math.sqrt(0.5 * (std1**2 + std2**2))


def aggregate_stats(scores):
    mean_score = statistics.mean(scores)
    std_scores = round(math.sqrt(statistics.pvariance(scores)), 2)
    return f"{round(mean_score, 2)} ± {std_scores}"


if __name__ == "__main__":
    DATA_DIR = "../PARALLEL_CORPORA"
    results = {}
    for model, model_name in zip(
        ["LEXICON", "SENTENCE_DI", "TOKEN_DI", "REGRESSION"],
        ["MSA Lexicon", "Sentence DI", "Token DI", "Sentence ALDi"],
    ):
        results[model] = {}
        for SEED in [30, 42, 50]:
            for corpus in [
                "BIBLE_tn",
                "BIBLE_ma",
            ]:
                seed_prefix = f"_{SEED}" if model in ["TOKEN_DI", "REGRESSION"] else ""
                df = pd.read_csv(
                    f"{DATA_DIR}/{model}_{corpus}{seed_prefix}.tsv", sep="\t"
                )
                D_prime = compute_D_statistic(df)
                if corpus not in results[model]:
                    results[model][corpus] = []
                results[model][corpus].append(D_prime)

            for corpus in [
                "DIAL2MSA_EGY",
                "DIAL2MSA_MGR",
            ]:
                seed_prefix = f"_{SEED}" if model in ["TOKEN_DI", "REGRESSION"] else ""
                df = pd.read_csv(
                    f"{DATA_DIR}/{model}_{corpus}{seed_prefix}.tsv", sep="\t"
                )
                D_prime = compute_D_statistic(df)
                if corpus not in results[model]:
                    results[model][corpus] = []
                results[model][corpus].append(D_prime)
    df = pd.DataFrame(results)
    print(
        df.apply(
            lambda row: pd.Series({k: aggregate_stats(row[k]) for k in row.keys()}),
            axis=1,
        ).T.to_latex(index=True)
    )
