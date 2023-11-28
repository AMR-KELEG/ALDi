import re
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cmx

font = {"size": 7}
matplotlib.rc("font", **font)
matplotlib.rcParams["axes.spines.left"] = True
matplotlib.rcParams["axes.spines.right"] = True
matplotlib.rcParams["axes.spines.top"] = True
matplotlib.rcParams["axes.spines.bottom"] = True


def plot_RMSE():
    """Generate a bar plot of the RMSEs of the four models on AOC-ALDi's test data."""
    plt.figure(figsize=(6.3 / 2, 1.5))

    models = [
        "(Baseline 1)\nMSA Lexicon",
        "(Baseline 2)\nSentence DI",
        "(Baseline 3)\nToken DI",
        "(Our Model)\nSentence ALDi",
    ]
    RMSEs = [0.34, 0.49, 0.30, 0.18]

    cmap = cmx.Reds

    for i, (model, RMSE) in enumerate(zip(models, RMSEs)):
        plt.barh(y=-i, width=RMSE, label=model, color=cmap(2 * RMSE))
        plt.annotate(
            RMSE,
            xy=(RMSE + 0.02, -i),
            size=7,
            color="black",
        )

    plt.yticks(
        ticks=[-i for i in range(len(models))],
        labels=models,
        rotation=0,
        weight="bold",
        size=8,
    )
    plt.xlim(0, 1)
    plt.xlabel("RMSE (↓) on AOC-ALDi's test data", weight="bold")
    plt.ylabel("Model", weight="bold")
    plt.savefig(f"RMSE.pdf", bbox_inches="tight")


plot_RMSE()
