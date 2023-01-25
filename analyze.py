import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def generate_report(filename):
    df = pd.read_csv(filename)
    return compute_scores(y_true=df["label"], y_pred=df["prediction"])


def compute_scores(y_true, y_pred):
    ax = sns.heatmap(
        confusion_matrix(y_true, y_pred),
        cmap="Greens",
        annot=True,
        fmt="d",
        vmin=0,
        vmax=len(y_true) // 2,
    )
    ax.set_xlabel("prediction")
    ax.set_ylabel("true")

