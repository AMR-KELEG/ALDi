import pandas as pd
import krippendorff
from collections import Counter
from statsmodels.stats import inter_rater as irr

if __name__ == "__main__":
    AOC_df = pd.read_csv(
        "../data/AOC/4_AOC_aggregated_discard_junk_samples.tsv", sep="\t"
    )

    AOC_df["dialectness_level"] = AOC_df["dialectness_level"].apply(
        lambda l: l[1:-1].split(", ")
    )

    AOC_3ann_df = AOC_df[
        (AOC_df["number_annotations"] == 3) & (AOC_df["ratio_junk_or_missing"] == 0)
    ]

    AOC_3ann_comments_df = AOC_3ann_df[
        AOC_3ann_df["source"].apply(lambda s: s.endswith("_c"))
    ]

    AOC_3ann_control_df = AOC_3ann_df[
        AOC_3ann_df["source"].apply(lambda s: s.endswith("_a"))
    ]

    EPS = 1e-6

    def annotations_to_counts(annotations_list):
        return [
            sum([abs(float(ann) - rating) < EPS for ann in annotations_list])
            for rating in [0, 1 / 3, 2 / 3, 1]
        ]

    percentage_df = (
        AOC_3ann_df["dialectness_level"]
        .apply(lambda l: sorted(l))
        .value_counts(normalize=True)
        .apply(lambda v: round(100 * v, 2))
    ).reset_index()

    def compute_scores(df, binary_labels=False, dataset=""):
        print(f"{dataset + ' - '}N=", df.shape[0])
        if binary_labels:
            aggregated_annotations = (
                df["dialectness_level"]
                .apply(
                    lambda l: (
                        sum([abs(float(label)) < EPS for label in l]),
                        sum([abs(float(label)) > EPS for label in l]),
                    )
                )
                .tolist()
            )
            print(Counter(aggregated_annotations))
        else:
            aggregated_annotations = (
                df["dialectness_level"]
                .apply(lambda l: annotations_to_counts(l))
                .tolist()
            )

        print(
            "Fleiss Kappa (3 annotations):",
            round(irr.fleiss_kappa(aggregated_annotations, method="fleiss"), 2),
        )

        print(
            "Krippendorff's alpha for nominal metric: ",
            round(
                krippendorff.alpha(
                    value_counts=aggregated_annotations, level_of_measurement="nominal"
                ),
                2,
            ),
        )

        print(
            "Krippendorff's alpha for ordinal metric: ",
            round(
                krippendorff.alpha(
                    value_counts=aggregated_annotations, level_of_measurement="ordinal"
                ),
                2,
            ),
        )

        print(
            "Krippendorff's alpha for interval metric: ",
            round(
                krippendorff.alpha(
                    value_counts=aggregated_annotations, level_of_measurement="interval"
                ),
                2,
            ),
        )

    for df, dataset in zip(
        [AOC_3ann_df, AOC_3ann_comments_df, AOC_3ann_control_df],
        ["All", "Comments", "Control"],
    ):
        compute_scores(df, binary_labels=False, dataset=dataset)
        print()
        print("\n")
