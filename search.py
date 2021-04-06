#!/usr/bin/python3


from BoW import BagWords
import argparse
import os
import re


def retrieve_articles(path):
    strings = {}
    i = 1

    for filename in os.listdir(path):
        text = {}
        path_to_file = f"{path}/{filename}"

        with open(path_to_file, "r") as file:
            my_string = file.read()

        title = re.sub(r"\.[\w]+$", "", filename)

        text["title"] = title
        text["text"] = my_string
        strings[str(i)] = text
        i += 1

    return strings


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("language", help="select a language (en - de - it)",
                        type=str, metavar="lang")
    parser.add_argument("-c", help="use Cistem as a Tokenizer (only German)",
                        action="store_true")
    args = parser.parse_args()

    if args.language not in ["de", "it", "en"]:
        print("Unsupported language!")
        return

    # TODO:
    # check if input/output folders are there
    # if not create them

    # retrieve all articles from folder bestand and stopwords
    articles = retrieve_articles("input")
    bow = BagWords(args.language)

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
            print(articles[str(result + 1)]["title"], "\n")

        else:
            print("No matching document found\n")


if __name__ == "__main__":
    main()
