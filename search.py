import argparse
import json
import os
import re

from BoW import BagWords


def retrieve_articles(path):
    strings = {}

    for i, filename in enumerate(os.listdir(path)):
        text = {}
        path_to_file = f"{path}/{filename}"

        with open(path_to_file, "r") as file:
            my_string = file.read()

        title = re.sub(r"\.[\w]+$", "", filename)

        text["title"] = title
        text["text"] = my_string
        strings[i] = text

    return strings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "language", help="select a language (en - de - it)",
        type=str, metavar="lang"
    )
    args = parser.parse_args()

    if args.language not in {"de", "it", "en"}:
        print("Unsupported language!")
        return

    # collect stopwords
    with open("stopwords-iso.json") as stopwords:
        stopwords = json.load(stopwords)

    stopwords = stopwords[args.language]

    # retrieve all articles from folder bestand and stopwords
    articles = retrieve_articles("input")
    bow = BagWords(args.language, stopwords)

    for article in articles:
        bow.add_sentence(articles[article]["text"])

    bow.compute_matrix(process=True)
    bow.tf_idf()

    while True:
        user_input = input("Enter data query or '...' to exit:\n> ")
        if user_input == "...":
            break

        result = bow.similarity(user_input)

        if result:
            print(articles[result]["title"], "\n")

        else:
            print("No matching document found\n")


if __name__ == "__main__":
    main()
