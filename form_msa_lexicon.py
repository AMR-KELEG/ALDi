import os
import re
import pickle
import argparse
from tqdm import tqdm
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter

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


## Tokenizer ##
PATTERNS = [
    # Split whitespaces
    r"(\s)",
    # Split numerals
    r"([0-9\u0660-\u0669]+[.]?[0-9\u0660-\u0669]*)",
    # Split . as long as it's not surrounded by numerals
    r"(?![0-9\u0660-\u0669])([.])(?![0-9\u0660-\u0669])",
    # Split text emojis
    r"([:;]-?\S\b)",
    # Split Non-arabic punctuation marks
    # TODO: ":" breaks the text emojis
    r"([!\"#$%&'()*+,-./:;<=>?@[\\\]^_`{\|}~])",
    # Split Arabic punctuation marks
    r"([؛،؟٬٪])",
    # Split Quranic punctuation marks
    r"([٭۞۩ﷺ])",
]


def tokenize_text(text):
    """Tokenize a string based on separator regexps."""
    tokens = [text]

    for splitting_pattern in PATTERNS:
        tokens = [
            sub_token
            for token in tokens
            for sub_token in re.split(splitting_pattern, token)
            if token
        ]
    return [t for t in tokens if t.strip()]


def form_lexicon(corpus):
    """Tokenize a corpus of sentences."""
    tokens = [
        tlist
        for sentence_list in tqdm(corpus)
        for sentence in sentence_list
        for tlist in tokenize_text(sentence)
    ]
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


def generate_lexicon():
    subtitles_corpus = [
        load_txt_file(f) for f in tqdm(Path(OUTPUT_TXT_DIR).glob("*.txt"))
    ]

    LEXICON = form_lexicon(subtitles_corpus)

    with open("data/MSA_raw_corpora/lexicon.pkl", "wb") as outputfile:
        pickle.dump(LEXICON, outputfile)


def main():
    parser = argparse.ArgumentParser(
        description="Form an MSA Lexicon from txt files.",
    )
    parser.add_argument(
        "-g",
        "--generate_txt_files",
        help="Genreate txt files from the Opensubtitles xml files.",
    )
    parser.add_argument(
        "-f",
        "--form_lexicon",
        help="Form a Lexicon Counter object from the Opensubtitles txt files.",
    )

    args = parser.parse_args()

    if args.generate_txt_files:
        parse_opensubtitles_xml_to_txt()
    elif args.form_lexicon:
        generate_lexicon()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
