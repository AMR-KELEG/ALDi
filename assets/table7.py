import pandas as pd
import matplotlib

pd.set_option("display.max_colwidth", None)
cmap = matplotlib.cm.get_cmap("PRGn_r")


def generate_combined_column(dfs, column_name="MSA_score"):
    combined_df = pd.concat([df[[column_name]] for df in dfs], axis=1)
    mean_scores = combined_df.mean(axis=1)
    std_scores = combined_df.std(axis=1)
    return pd.DataFrame(
        {
            column_name: [
                round(mean_score, 2)
                for mean_score, std_score in zip(mean_scores, std_scores)
            ]
        }
    )


def generate_cell_color(score):
    if score <= 1 / 9:
        r, g, b, a = cmap(0.25)
    else:
        # 0.25 -> 1 to 0.6 -> 1
        r, g, b, a = cmap(score * (30 / 90) + (60 / 90))
    color = f"\\textcolor[rgb]{{{round(r, 2)}, {round(g, 2)}, {round(b, 2)}}}{{"
    return color + f"{round(score*1.0, 2)}}}"


def generate_colored_cells(df):
    return df.apply(
        lambda row: generate_cell_color(row["DA_score"], row["MSA_score"]),
        axis=1,
    )


def generate_agg_df(dfs):
    DA_columns = generate_combined_column(dfs, "DA_score")
    MSA_columns = generate_combined_column(dfs, "MSA_score")

    input_df = dfs[0][
        [
            "ID",
            "Feature name",
            "MSA_text",
            "DA_text",
            "Word order",
            "Gender",
            "English_text",
        ]
    ]
    return pd.concat([input_df, MSA_columns, DA_columns], axis=1)


if __name__ == "__main__":
    # Load the dataframes!
    di_df = pd.read_csv("../Contrastive_scores/SENTENCE_DI_CONTRAST.tsv", sep="\t")
    lex_df = pd.read_csv("../Contrastive_scores/LEXICON_CONTRAST.tsv", sep="\t")
    # tag_df = pd.read_csv("../Contrastive_scores/TAGGING_CONTRAST.tsv", sep="\t")
    # reg_df = pd.read_csv("../Contrastive_scores/REGRESSION_CONTRAST.tsv", sep="\t")

    seeds = [30, 42, 50]
    reg_dfs = [
        pd.read_csv(f"../Contrastive_scores/REGRESSION_CONTRAST_{seed}.tsv", sep="\t")
        for seed in seeds
    ]
    token_di_dfs = [
        pd.read_csv(f"../Contrastive_scores/TOKEN_DI_CONTRAST_{seed}.tsv", sep="\t")
        for seed in seeds
    ]
    reg_df = generate_agg_df(reg_dfs)
    tag_df = generate_agg_df(token_di_dfs)

    basic_df = lex_df[
        [
            "ID",
            "MSA_text",
            "DA_text",
            "English_text",
            "Word order",
            "Gender",
        ]
    ].copy()

    for df in [lex_df, reg_df, tag_df, di_df]:
        df["DA_score"] = df["DA_score"].apply(lambda s: max(s, 0))
        df["MSA_score"] = df["MSA_score"].apply(lambda s: max(s, 0))

    MODELS = ["LEX", "DI", "TAG", "REG"]
    dfs = [lex_df, di_df, tag_df, reg_df]
    for model, df in zip(MODELS, dfs):
        basic_df[f"{model}_MSA"] = df["MSA_score"].apply(generate_cell_color)
        basic_df[f"{model}_DA"] = df["DA_score"].apply(generate_cell_color)

    DA_words = ["بتقول", "هتقول", "اتقالت", "مبتقولش", "ماتقوليش", "ماتقولش"]
    for col in ["MSA_text", "DA_text"]:
        basic_df[col] = basic_df[col].apply(
            lambda s: " ".join(
                [
                    f"\AR{{{w}}}" if w not in DA_words else f"\\underline{{\AR{{{w}}}}}"
                    for w in s.split()[-1::-1]
                ]
            )
        )

    basic_df["Feature name"] = lex_df.apply(
        lambda row: " ".join(row["Feature name"].split()[:3])
        + f"\t{row['English_text']}",
        axis=1,
    )

    final_m_df = basic_df[
        (basic_df["ID"].isin(["1", "2a", "3a", "4b", "5b"]))
        & (basic_df["Gender"].isin(["M", "B"]))
    ]

    final_f_df = basic_df[
        (basic_df["ID"].isin(["1", "2a", "3a", "4b", "5b"]))
        & (basic_df["Gender"].isin(["F", "B"]))
    ]

    final_merged_df = pd.merge(
        left=final_m_df,
        right=final_f_df,
        how="left",
        on=["ID", "Word order"],
        suffixes=("_m", "_f"),
    )

    final_merged_df.head()

    for model in ["LEX", "DI", "REG", "TAG"]:
        for dialect in ["DA", "MSA"]:
            column = f"{model}_{dialect}"
            c_m = f"{column}_m"
            c_f = f"{column}_f"
            final_merged_df[column] = final_merged_df.apply(
                lambda row: row[c_m] + " / " + row[c_f]
                if row["Gender_m"] != "B" and row[c_m] != row[c_f]
                else row[c_m],
                axis=1,
            )

    final_merged_df.columns

    features_types = final_merged_df["Feature name_f"].tolist()

    for i in range(0, len(features_types) - 1):
        if i % 2 == 0:
            features_types[i] = "\\textbf{" + features_types[i].split("\t")[0] + "}"
        else:
            features_types[i] = "\\textbf{En}: " + features_types[i].split("\t")[1]

    final_merged_df["Feature name"] = features_types

    for column in ["MSA_text", "DA_text"]:
        final_merged_df[column] = final_merged_df[f"{column}_f"]

    table_str = final_merged_df[
        [
            "Feature name",
            "MSA_text",
            "DA_text",
            "Word order",
            "LEX_MSA",
            "LEX_DA",
            "DI_MSA",
            "DI_DA",
            "TAG_MSA",
            "TAG_DA",
            "REG_MSA",
            "REG_DA",
        ]
    ].to_latex(index=False, escape=False)

    lines = [l.strip() for l in table_str.split("\n")]
    output_lines = []
    line_index = 0
    for l in lines:
        if not l.startswith("\\textbf{E"):
            output_lines.append(l)
        else:
            output_lines.append(l)
            output_lines.append("\\midrule")

    print("\n".join(output_lines))
