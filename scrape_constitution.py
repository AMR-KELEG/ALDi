#!/usr/bin/python
# -*- coding: utf-8 -*-
# Source: https://gist.github.com/eon01/8382176

from bs4 import BeautifulSoup
import urllib.error, urllib.parse
from urllib.request import Request, urlopen

__author__ = "Aymen Amri"
__email__ = "amri.aymen@gmail.com"


for counter in range(1, 6):
    counter = str(counter)
    baseUrl = "http://www.marsad.tn/constitution/4/chapitre/"
    url = baseUrl + counter
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    constFile = urlopen(req)
    constHtml = constFile.read()
    soup = BeautifulSoup(constHtml)
    constArt = soup.find_all("div")
    articles = soup.findAll("div", {"class": "clear texte"})
    titles = soup.findAll("h4", {"class": "clearfloat"})

    print(("." * int(counter)))

    import csv

    fileName = counter + ".csv"
    w = csv.writer(open(fileName, "w"))

    for article, title in zip(articles, titles):
        textTitle = "".join(title.findAll(text=True))
        textArticle = "".join(article.findAll(text=True))
        out = [textTitle.encode("utf-8"), textArticle.encode("utf-8")]
        w.writerow(out)
