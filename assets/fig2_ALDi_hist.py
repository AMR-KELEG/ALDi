import matplotlib

# Set font of what?
font = {"size": 7}
matplotlib.rc("font", **font)

# Update the spines
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.bottom"] = False
matplotlib.rcParams["axes.spines.left"] = True

matplotlib.rcParams["text.latex.preamble"] = [r"\boldmath"]

import pandas as pd
import matplotlib.pyplot as plt


def generate_bar_chart_AOC_ALDI(df):
    ranges = [(0, 0.11), (0.11, 0.44), (0.44, 0.77), (0.77, 2)]
    colors = [
        "#7fbf7b",
        "#e7d4e8",
        "#af8dc3",
        "#762a83",
    ]
    labels = [
        "MSA\n" + r"$[0,0.11[$",
        "Little\n" + r"$[0.11,0.44[$",
        "Mixed\n" + r"$[0.44,0.77[$",
        "Most\n" + r"$[0.77,1[$",
    ]

    plt.figure(figsize=(6.3 / 2, 1))
    x_before = 0
    w_before = 0
    offset = 0.1
    xticks, xlabels = [], []
    for (sc_low, sc_high), color, label in zip(ranges, colors, labels):
        score_df = df[
            (df["average_dialectness_level"] >= sc_low)
            & (df["average_dialectness_level"] < sc_high)
        ]
        n_score_samples = score_df.shape[0]

        w_current = max(0.2, round(n_score_samples / 50000, 1))
        w_current = 0.5

        x = x_before + w_before + offset

        xticks.append(x + w_current / 2)
        xlabels.append(label.split("\n")[-1])

        plt.bar(
            x,
            height=n_score_samples,
            width=w_current,
            bottom=0,
            align="edge",
            color=color,
            label=label.split()[0],
        )

        plt.annotate(
            f"{n_score_samples}",
            xy=(
                xticks[-1] - w_current / 2 + (0.09),
                n_score_samples + 2000,
            ),
            size=6,
            color="black",
            weight="bold",
        )
        x_before = x
        w_before = w_current
    #     plt.ylim(0, 42000)
    #     plt.xlim(-0.1, x_before + w_before)
    #     plt.legend(frameon=False, prop={"size": 7}, handlelength=0.8)
    plt.xticks(ticks=xticks, labels=xlabels, rotation=0, weight="bold", size=6)
    plt.xlabel("Aggregated ALDi Score", weight="bold")
    plt.ylabel("No. of samples", weight="bold")

    plt.savefig("fig2.pdf", bbox_inches="tight")


AOC_ALDi_df = pd.concat(
    [
        pd.read_csv(f"../data/AOC/{split}.tsv", sep="\t")
        for split in ["train", "dev", "test"]
    ]
)
generate_bar_chart_AOC_ALDI(AOC_ALDi_df)
