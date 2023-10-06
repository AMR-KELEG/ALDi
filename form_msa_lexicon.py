import os
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter
from utils import tokenize_text

OUTPUT_TXT_DIR = "data/MSA_raw_corpora/OpenSubtitles/parsed"


def parse_utterances_from_segment(segment):
    """Extract the utterance string from a segment."""
    utterances = [s.strip() for s in segment.strings if s.strip()]
    assert len(utterances) == 1
    return utterances[0]


def transform_xml_to_txt(filename, output_filename):
    """Load utterances from an Opensubtitles xml file."""
    with open(filename, "r") as f:
        data = f.read()

    parsed_xml_file = BeautifulSoup(data, "xml")
    segments = parsed_xml_file.find_all("s")

    utterances = [parse_utterances_from_segment(segment) for segment in segments]
    assert len(utterances)

    with open(output_filename, "w") as f:
        f.write("\n".join(utterances))
    return utterances


def find_charset(corpus):
    """
    Find the set of unique characters in the corpus.

    Args:
    corpus - a list of strings
    """
    char_sets = [set([c for c in s]) for s in corpus]
    return set.union(*char_sets)


def load_txt_file(filename):
    with open(filename, "r") as f:
        return [l.strip() for l in f if l.strip()]


def form_lexicon(corpus):
    """Tokenize a corpus of sentences."""
    tokens = [
        tlist
        for sentence_list in tqdm(corpus)
        for sentence in sentence_list
        for tlist in tokenize_text(sentence)
    ]
    return Counter(tokens)


def form_un_lexicon(filename):
    with open(filename, "r") as f:
        tokens = [tlist for sentence in tqdm(f) for tlist in tokenize_text(sentence)]
        return Counter(tokens)


def parse_opensubtitles_xml_to_txt():
    # ✔ - All the files are .xml
    subtitles_filenames = [
        p for p in Path("data/MSA_raw_corpora/OpenSubtitles/raw/ar/").glob("*/*/*.xml")
    ]

    os.makedirs(OUTPUT_TXT_DIR, exist_ok=True)

    output_filenames = [
        Path(OUTPUT_TXT_DIR, "_".join(str(filename)[:-4].split("/")[-3:]) + ".txt")
        for filename in subtitles_filenames
    ]

    for filename, output_filename in tqdm(
        zip(subtitles_filenames, output_filenames), total=len(subtitles_filenames)
    ):
        transform_xml_to_txt(filename, output_filename)


def generate_lexicon(opensubtitles=False, un=False):

    if opensubtitles:
        file_paths = (
            Path(OUTPUT_TXT_DIR).glob("*.txt")
            if opensubtitles
            else Path("data/MSA_raw_corpora/").glob("UN-en-ar/UNv1.0.ar-en.ar")
            if un
            else []
        )

        subtitles_corpus = [load_txt_file(f) for f in tqdm(file_paths)]

        LEXICON = form_lexicon(subtitles_corpus)

    elif un:
        LEXICON = form_un_lexicon("data/MSA_raw_corpora/UN-en-ar/UNv1.0.ar-en.ar")

    with open(
        f"data/MSA_raw_corpora/lexicon_{'UN' if un else 'opensubtitles'}.pkl", "wb"
    ) as outputfile:
        pickle.dump(LEXICON, outputfile)


def main():
    parser = argparse.ArgumentParser(description="Form an MSA Lexicon from txt files.",)
    parser.add_argument(
        "-g",
        "--generate_txt_files",
        action="store_true",
        help="Genreate txt files from the Opensubtitles xml files.",
    )

    subparsers = parser.add_subparsers(help="Form lexicon file.")

    lexicon_generation_subparser = subparsers.add_parser(
        "form_lexicon", help="Form a Lexicon Counter object from corpus txt files.",
    )
    lexicon_generation_subparser.add_argument(
        "-c",
        choices=["UN", "OpenSubtitles"],
        required=True,
        help="Select only one corpus!",
    )

    args = parser.parse_args()

    if args.generate_txt_files:
        parse_opensubtitles_xml_to_txt()
    elif "c" in args:
        un = True if args.c == "UN" else False
        opensubtitles = True if args.c == "OpenSubtitles" else False
        generate_lexicon(opensubtitles=opensubtitles, un=un)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
