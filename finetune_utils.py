from pathlib import Path
from datasets import Dataset


def load_filelines(filename):
    """Load the lines of a specific file"""
    with open(filename, "r") as f:
        return [l.strip() for l in f]


def load_dataset(split):
    """Load the dataset as a list of dictionaries"""
    BASE_DATA_DIR = "data/NADI_mapped"
    MSA_tweets = load_filelines(str(Path(BASE_DATA_DIR, f"{split}_MSA.txt")))
    DA_tweets = load_filelines(str(Path(BASE_DATA_DIR, f"{split}_DA.txt")))

    samples = []
    samples += [{"text": tweet, "label": 0} for tweet in MSA_tweets]
    samples += [{"text": tweet, "label": 1} for tweet in DA_tweets]

    return Dataset.from_list(samples)
