import matplotlib
import pandas as pd
import matplotlib.pyplot as plt

font = {"size": 7}
matplotlib.rc("font", **font)
matplotlib.rcParams["axes.spines.left"] = True
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.bottom"] = False
matplotlib.rcParams["text.latex.preamble"] = [r"\boldmath"]


def plot_levels_per_dialect(df):
    """Generate the distribution of the levels across the different dialects."""

    # Rename "junk" to "Not Arabic"
    df.loc[df["DClass"] == "junk", "DClass"] = "Not Arabic"
    df.loc[df["DLevel"] == "junk", "DLevel"] = "Not Arabic"

    # Map of annotations to labels
    dialects = [
        "levantine",
        "gulf",
        "egyptian",
        "general",
        "Not Arabic",
        "missing",
        "OTHERS",
    ]
    dialect_labels = ["LEV", "GLF", "EGY", "GEN", r"$\neg$ ARA", "MISSING", "REST"]
    dialect_labels_dict = {d: l for d, l in zip(dialects, dialect_labels)}

    other_dialects = [
        "maghrebi",
        "iraqi",
        "notsure",
        "other",
    ]

    plt.figure(figsize=(6.3 / 2, 1.5))

    x_before = 0
    w_before = 0
    offset = 0.1
    label_in_legend = {}
    xticks, xlabels = [], []
    for dialect in dialects:
        if dialect == "OTHERS":
            dialect_df = df.loc[df["DClass"].isin(other_dialects), "DLevel"]
        else:
            dialect_df = df.loc[df["DClass"] == dialect, "DLevel"]
        n_dialect_samples = dialect_df.shape[0]

        d = dialect_df.value_counts(dropna=False).to_dict()

        labels = ["most", "mixed", "little", "Not Arabic", "missing"]
        label_colors = {
            "little": "#e7d4e8",
            "mixed": "#af8dc3",
            "most": "#762a83",
            "Not Arabic": "#d9f0d3",
            "missing": "#7fbf7b",
        }

        patterns = [
            "+",
            "x",
            "o",
            "O",
            ".",
            "*",
            "-",
            "/",
            "\\",
            "|",
        ]
        label_patterns = {label: pattern for label, pattern in zip(labels, patterns)}
        label_patterns = {label: "" for label, pattern in zip(labels, patterns)}

        w_current = max(0.2, round(n_dialect_samples / 50000, 1))
        w_current = 0.5

        x = x_before + w_before + offset

        xticks.append(x + w_current / 2)

        xlabels.append(dialect_labels_dict[dialect])
        total_height = 0
        for label in labels:
            height = d.get(label, 0)
            if not height:
                continue
            if not label_in_legend.get(label):
                plt.bar(
                    x,
                    height=height,
                    width=w_current,
                    bottom=total_height,
                    align="edge",
                    color=label_colors[label],
                    label=label.capitalize()
                    if label != "Not Arabic"
                    else r"$\neg$ Arabic",
                    hatch=label_patterns[label],
                )
                label_in_legend[label] = True
            else:
                plt.bar(
                    x,
                    height=height,
                    width=w_current,
                    bottom=total_height,
                    align="edge",
                    color=label_colors[label],
                    hatch=label_patterns[label],
                )
            total_height += height
            if dialect in ["levantine", "gulf", "egyptian", "general"]:
                if height > 1000:
                    plt.annotate(
                        f"{round(100*height/n_dialect_samples, 1)}%",
                        xy=(
                            xticks[-1] - w_current / 2,
                            total_height - height / 2 - 1000,
                        ),
                        size=6,
                        color="white" if label != "little" else "black",
                    )

        plt.annotate(
            f"{total_height}",
            xy=(
                xticks[-1] - w_current / 2 + (0.05 if total_height < 10000 else 0),
                total_height + 1000,
            ),
            size=6,
            color="black",
            weight="bold",
        )
        x_before = x
        w_before = w_current
    plt.ylim(0, 42000)
    plt.xlim(-0.1, x_before + w_before)
    handles, labels = plt.gca().get_legend_handles_labels()
    order = [2, 1, 0, 3, 4]
    plt.legend(
        [handles[idx] for idx in order],
        [labels[idx] for idx in order],
        frameon=False,
        prop={"size": 6},
        handlelength=0.8,
        title="ALDi",
    )
    plt.xticks(ticks=xticks, labels=xlabels, rotation=90)
    plt.xlabel("Dialect label", weight="bold")
    plt.ylabel("No. of annotations", weight="bold")
    plt.savefig("fig1.pdf", bbox_inches="tight")


if __name__ == "__main__":
    df = pd.read_csv("../data/AOC/2_AOC_augmented.tsv", sep="\t")
    non_msa_df = df.loc[df["DClass"] != "msa"]
    plot_levels_per_dialect(non_msa_df)
