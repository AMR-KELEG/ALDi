import pandas as pd
import matplotlib.pyplot as plt

if __name__ == "__main__":
    DATA_DIR = "../final_results_22Jun_test"
    fig, axes = plt.subplots(
        nrows=4, ncols=1, figsize=(6.3 / 2, 3), sharex=True, sharey=True
    )
    width = 0.03
    last_x = 0

    GREEN = "#7fbf7b"
    VIOLET = "#762a83"

    # TODO: Update the names of the corpora
    for model, model_name, ax in zip(
        ["LEXICON", "DI", "TAGGING", "REGRESSION"],
        ["MSA Lexicon", "Sentence DI", "Token DI", "Sentence ALDi"],
        axes,
    ):
        last_x = 0
        for corpus in [
            "BIBLE_tn",
            "BIBLE_ma",
        ]:

            df = pd.read_csv(f"{DATA_DIR}/{model}_{corpus}.tsv", sep="\t")
            if corpus != "BIBLE_ma":
                ax.boxplot(
                    x=df["MSA_score"].apply(lambda s: min(max(s, 0), 1)),
                    widths=width,
                    positions=[last_x + 2 * width],
                    showfliers=True,
                    flierprops=dict(
                        marker="o",
                        markerfacecolor=GREEN,
                        markersize=3,
                        linestyle="none",
                        alpha=0.005,
                        markeredgecolor="none",
                    ),
                    boxprops=dict(linestyle="-", linewidth=1, color=GREEN),
                    capprops=dict(linestyle="-", linewidth=1, color="black"),
                    whiskerprops=dict(linestyle="-", linewidth=1, color=GREEN),
                )

                last_x += 2 * width
            ax.boxplot(
                x=df["DA_score"].apply(lambda s: min(max(s, 0), 1)),
                widths=width,
                positions=[last_x + 2 * width],
                showfliers=True,
                flierprops=dict(
                    marker="o",
                    markerfacecolor=VIOLET,
                    markersize=3,
                    linestyle="none",
                    alpha=0.005,
                    markeredgecolor="none",
                ),
                boxprops=dict(linestyle="-", linewidth=1, color=VIOLET),
                capprops=dict(linestyle="-", linewidth=1, color="black"),
                whiskerprops=dict(linestyle="-", linewidth=1, color=VIOLET),
            )

            last_x += 2 * width

        last_x += 2 * width

        for corpus in [
            "DIAL2MSA_EGY",
            "DIAL2MSA_MGR",
        ]:

            df = pd.read_csv(f"{DATA_DIR}/{model}_{corpus}.tsv", sep="\t")
            if corpus != "BIBLE_ma":
                ax.boxplot(
                    x=df["MSA_score"].apply(lambda s: min(max(s, 0), 1)),
                    widths=width,
                    positions=[last_x + 2 * width],
                    showfliers=True,
                    flierprops=dict(
                        marker="o",
                        markerfacecolor=GREEN,
                        markersize=3,
                        linestyle="none",
                        alpha=0.005,
                        markeredgecolor="none",
                    ),
                    boxprops=dict(linestyle="-", linewidth=1, color=GREEN),
                    capprops=dict(linestyle="-", linewidth=1, color="black"),
                    whiskerprops=dict(linestyle="-", linewidth=1, color=GREEN),
                )

                last_x += 2 * width
            ax.boxplot(
                x=df["DA_score"].apply(lambda s: min(max(s, 0), 1)),
                widths=width,
                positions=[last_x + 2 * width],
                showfliers=True,
                flierprops=dict(
                    marker="o",
                    markerfacecolor=VIOLET,
                    markersize=3,
                    linestyle="none",
                    alpha=0.005,
                    markeredgecolor="none",
                ),
                boxprops=dict(linestyle="-", linewidth=1, color=VIOLET),
                capprops=dict(linestyle="-", linewidth=1, color="black"),
                whiskerprops=dict(linestyle="-", linewidth=1, color=VIOLET),
            )

            last_x += 4 * width

        last_x -= 2 * width

        ax.annotate(model_name, xy=(last_x - 15 * width, 1.15), size=6, weight="bold")

        ax.set_xlim(0, last_x + width)
        ax.set_ylim(-0.1, 1.2)
        # plt.xticks(labels=None)
        ax.set_xticks([], [])
        ax.set_yticks([0, 1], [0, 1], size=6)
        ax.spines.top.set(visible=False)
        ax.spines.bottom.set(visible=True)
        ax.spines.left.set(visible=True)
        ax.spines.right.set(visible=False)
        if model == "REGRESSION":
            ax.set_xticks(
                [(i + 1) * 2 * width for i in range(9) if i != 3 and i != 6],
                (
                    [
                        "$BIBLE$\n$(MSA)$",
                        "$BIBLE$\n$(TUN)$",
                        "$BIBLE$\n$(MOR)$",
                    ]
                    + [
                        "$DM_{E}$\n$(MSA)$",
                        "$DM_{E}$\n$(EGY)$",
                        "$DM_{M}$\n$(MSA)$",
                        "$DM_{M}$\n$(MGR)$",
                    ]
                ),
                size=6,
            )

    fig.savefig("fig3_boxplot.pdf", bbox_inches="tight")
