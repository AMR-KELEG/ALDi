#!/usr/bin/env python

import os
import re
import urllib.request
from glob import glob
from tqdm import tqdm
from bs4 import BeautifulSoup

# Parse an article
def parse_article(filename):
    """Extract the title and the transcription from a speech's HTML page.

    Args:
        filename: The path for the HTML page.

    Returns:
        The speech's title, and a list of strings for the transcript.
    """
    with open(filename, "r") as f:
        html_str = f.read()
    soup = BeautifulSoup(html_str, "html.parser")

    article_title = soup.find_all(class_="story-title")[0].text
    paragraphs = [p.text.strip() for p in soup.find(class_="body").find_all("p")]

    return article_title, [re.sub(r"\xa0", "", p) for p in paragraphs if p]


def main():

    os.makedirs("speeches_html_pages", exist_ok=True)
    os.makedirs("speeches_txt_pages", exist_ok=True)

    # Extract the pages ids
    with open("speeches.html", "r") as f:
        html_str = "\n".join([l for l in f])

    soup = BeautifulSoup(html_str, "html.parser")
    articles = soup.find_all("article")

    # Download the articles' HTML files
    BASE_URL = "https://almanassa.com/stories/"
    articles_ids = [a.attrs["data-history-node-id"] for a in articles]

    for article_id in tqdm(articles_ids):
        url = BASE_URL + article_id
        f = urllib.request.urlopen(url)
        content = f.read()
        with open(f"speeches_html_pages/{article_id}.html", "wb") as f:
            f.write(content)

    # Parse the HTML files
    for filename in tqdm(glob("speeches_html_pages/*.html")):
        article_id = filename.split("/")[-1][:-5]
        article_title, article_text = parse_article(filename)
        with open(f"speeches_txt_pages/{article_id}.txt", "w") as f:
            f.write(article_title)
            f.write("\n=====\n")
            f.write("\n".join(article_text))


if __name__ == "__main__":
    main()
