import os
import re
import glob
from pathlib import Path
from transformers import AutoTokenizer, BertForSequenceClassification
from camel_tools.dialectid import DIDModel6
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

font = {"weight": "bold", "size": 10}
matplotlib.rc("font", **font)

VIOLET = "#7F007F"
GREY = "#A9A6A7"
GREY = "#D0CFCF"
DARK_GREY = "#837f81"
GREEN = "#1B7837"

OUTPUT_DIR = "speeches_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

model_name = "AMR-KELEG/Sentence-ALDi"
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=1)
tokenizer = AutoTokenizer.from_pretrained("UBC-NLP/MARBERT")

did = DIDModel6.pretrained()


def compute_sentence_ALDi_score(sentence):
    """Compute the ALDi score for a sentence.

    Args:
        sentence: A string.

    Returns:
        The ALDi score of the sentence.
    """
    features = tokenizer(sentence, return_tensors="pt")
    o = model(**features).logits[0].tolist()[0]
    return o


def generate_plot(scores, is_dialect, output_filename, plot_title=""):
    figure_width = len(scores) * 6.3 / 75
    fig, ax = plt.subplots(1, 1, figsize=(figure_width, 0.5))

    # figure_width = 6.3
    # fig, ax = plt.subplots(1, 1, figsize=(figure_width, 1))
    if sum(is_dialect):
        (markerline, stemlines, baseline) = ax.stem(
            [i for i in range(len(scores)) if is_dialect[i]],
            [s for s, d in zip(scores, is_dialect) if d],
            markerfmt="o",
            label="DA",
        )
        plt.setp(baseline, visible=False)
        plt.setp(markerline, "color", VIOLET)
        plt.setp(markerline, "marker", "*")
        plt.setp(stemlines, "color", plt.getp(markerline, "color"))
        plt.setp(markerline, markersize=5)

    if sum(is_dialect) != len(is_dialect):
        (markerline, stemlines, baseline) = ax.stem(
            [i for i in range(len(scores)) if not is_dialect[i]],
            [s for s, d in zip(scores, is_dialect) if not d],
            markerfmt="o",
            label="MSA",
        )
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("ALDi", size=6)
    ax.set_xlabel("Sentence Index in Speech", size=6)
    plt.setp(baseline, visible=False)
    plt.setp(markerline, "color", GREEN)
    plt.setp(markerline, markersize=2.5)
    plt.setp(markerline, "marker", "D")

    plt.setp(stemlines, "color", plt.getp(markerline, "color"))
    ax.set_title(plot_title, weight="bold", size=8, loc="left")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.xticks(fontsize=6)
    plt.yticks(fontsize=6)

    n_dialectal = sum(is_dialect)
    if "6210" in output_filename:
        # plt.annotate(text="Presidential Speech", xy=(2.5, 0.3), size=7)
        # plt.annotate(text="Q&A Gas", xy=(19, 0.6), size=7)
        # plt.annotate(text="Q&A Human Rights", xy=(26, 1.05), size=7)
        plt.annotate(text="Presidential Speech", xy=(2.5, 0.3), size=5)
        plt.annotate(text="Q&A\nGas", xy=(19, 0.6), size=5)
        plt.annotate(text="Q&A\nHuman Rights", xy=(26, 1.05), size=5)
        plt.plot([18.5, 18.5], [0, 1.05], "k--", alpha=0.25)
        plt.plot([25.5, 25.5], [0, 1.05], "k--", alpha=0.25)
        plt.xlim(0, 33.5)

        # plt.legend(title="", frameon=True, prop={"size": 7}, loc="upper left")

    print(f"{n_dialectal} sentences out of {len(is_dialect)} classified as DA")
    # plt.legend(title=f"Automatic DI")
    if "10jan" in output_filename or "1feb" in output_filename:
        plt.legend(title="", frameon=False, prop={"size": 5})
    fig.savefig(
        output_filename,
        bbox_inches="tight",
    )
    # plt.close()


def compute_sisi_speech_scores(filename):
    with open(filename, "r") as f:
        lines = [l.strip() for l in f if l.strip()]
    description = re.sub('["0-9/]', "", re.sub(r"\s+", "_", lines[0]))

    lines = lines[2:]
    if lines[-1].startswith("ألقيت"):
        lines = lines[:-1]
    # Drop (تصفيق)
    lines = [re.sub(r"[(][^)]*[)]", " ", l) for l in lines]
    lines = [l.strip() for l in lines if l.strip()]
    sentences = lines

    scores = [compute_sentence_ALDi_score(l) for l in sentences]
    dialects = [p.top for p in did.predict(sentences)]
    is_dialect = [p != "MSA" for p in dialects]

    mean_score = max(0, round(sum(scores) / len(scores), 3))
    output_file_name = str(
        Path(OUTPUT_DIR, f"{mean_score}_{description}_{Path(filename).name}")
    )
    with open(output_file_name, "w") as f:
        assert len(scores) == len(sentences) == len(dialects)
        for score, sentence, d in zip(scores, sentences, dialects):
            f.write(f"{score}\t{sentence}\t{d}\n")
    return scores, is_dialect, mean_score, description, sentences


for filename in [
    f"../analysis/speeches/speeches_txt_pages/{sp}.txt"
    for sp in "6210,5327,1877".split(",")
]:
    # 18June2023 for filename in [f"17April2023/analysis/speeches/speeches_txt_pages/{sp}.txt" for sp in "6210,5327,1877".split(",")]:
    # for filename in [f"17April2023/analysis/speeches/speeches_txt_pages/{sp}.txt" for sp in "6210,5327,9894,3507,1877,7975".split(",")]:
    # for filename in [f"13May2023/analysis/speeches/speeches_txt_pages/{sp}.txt" for sp in "6210".split(",")]:
    print(filename)
    print(Path(filename).name[:-4])
    file_id = Path(filename).name[:-4]
    scores, is_dialect, mean_score, description, sentences = compute_sisi_speech_scores(
        filename
    )
    output_filename = str(
        Path(
            OUTPUT_DIR,
            f"mean_{mean_score}_percentage_{round(sum(is_dialect)/len(is_dialect), 3)}_{description}_{Path(filename).stem}.pdf",
        )
    )
    # plot_titles = {"6210": "Press Conference (22 July 2022)",
    #                "9894": "Inauguration of projects in New Minya city (2 Mar 2023)",
    #                "1877": "African Investment Conference (7 Dec 2017)",
    #                "5327": "Judiciary Day (10 Oct 2021)"}
    plot_titles = {
        "6210": "e) El-Sisi - 22/7/2022",
        "9894": "Inauguration of projects in New Minya city (2 Mar 2023)",
        "1877": "c) El-Sisi - 7/12/2017",
        "5327": "d) El-Sisi - 10/10/2021",
    }
    # plot_titles = {}
    generate_plot(scores, is_dialect, output_filename, plot_titles.get(file_id, ""))

# Check the scores of the different sentences!
scores, is_dialect, mean_score, description, sentences = compute_sisi_speech_scores(
    "../analysis/speeches/speeches_txt_pages/6210.txt"
)
for score, sentence in zip(scores, sentences):
    print(f"{sentence}\t{round(score, 2)}")


def compute_scores(filename):
    """Compute the LoD scores for a sentence.

    Args:
        filename: A txt file of sentences.
    """
    with open(filename, "r") as f:
        # Remove / from lines
        # TODO: Why?
        lines = [re.sub("/", "", l.strip()) for l in f]
        lines = [l for l in lines if l.strip()]
    scores = [compute_sentence_ALDi_score(l) for l in tqdm(lines)]
    is_dialect = [p.top != "MSA" for p in did.predict(lines)]

    filename = Path(filename).name
    with open(str(Path(OUTPUT_DIR, f"{filename}_out")), "w") as f:
        for l, s in zip(lines, scores):
            f.write(f"{l}\t{s}\t{did.predict([l])[0].top}\n")

    # plots_titles = {"speech-egypt-28jan-tofix.txt": "Egyptian Revolution (28 Jan 2011)", "speech-tunisia-13jan.txt": "Tunisian Revolution (13 Jan 2011)", "speech-tunisia-10jan.txt": "Tunisian Revolution (10 Jan 2011)", "speech-egypt-1feb.txt": "Egyptian Revolution (1 Feb 2011)", "speech-egypt-10feb.txt": "Egyptian Revolution (10 Feb 2011)", "speech-tunisia-28dec.txt": "Tunisian Revolution (28 Dec 2010)"}
    plots_titles = {
        "speech-egypt-28jan-tofix.txt": "Egyptian Revolution (28 Jan 2011)",
        "speech-tunisia-13jan.txt": "b) Ben-Ali - 13/1/2011",
        "speech-tunisia-10jan.txt": "a) Ben-Ali - 10/1/2011",
        "speech-egypt-1feb.txt": "a) Mubarak - 1/2/2011",
        "speech-egypt-10feb.txt": "b) Mubarak - 10/2/2011",
        "speech-tunisia-28dec.txt": "Tunisian Revolution (28 Dec 2010)",
    }
    plot_title = plots_titles[filename]
    # Generate plot!
    output_filename = str(Path(OUTPUT_DIR, f"{filename}.pdf"))
    generate_plot(scores, is_dialect, output_filename, plot_title=plot_title)


for file in glob.glob("../analysis/speeches/arab_spring_speeches/*.txt"):
    print(Path(file).name)
    compute_scores(file)
