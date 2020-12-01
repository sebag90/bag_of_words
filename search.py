#!/usr/bin/python3

from BoW import BagWords
import argparse
import sys
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

        title = re.sub("\.[\w]+$", "", filename)

        text["title"] = title
        text["text"] = my_string
        strings[str(i)] = text
        i += 1

    return strings



# print progress bar
def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total:
        print()



def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("language", help="select a language (en - de - it)", type=str, metavar = "lang")
    parser.add_argument("-c", help="change the default tokenizer for German from Snowball to Cistem", action = "store_true")
    args = parser.parse_args()

    if args.language not in ["de", "it", "en"]:
        print("Language not supported yet\nPlease select a language between English (en), German (de) or Italian (it)")
        return
   

    # TODO:
    # check if input/output folders are there
    # if not create them

    # retrieve all articles from folder bestand and stopwords
    articles = retrieve_articles("input")
    bow = BagWords(args.language)

    for article in articles:
        bow.add_sentence(articles[article]["text"])
        #print_progress_bar(j, len(articles), prefix="Loading", length=40)
       

    bow.compute_matrix()
    
    
    while True:
        user_input =  input("Enter data query or '...' to exit:\n> ")
        if user_input == "...":
            break

        result = bow.similarity(user_input)
        if result != None:
            print(articles[str(result + 1)]["title"], "\n")
        else: 
            print("No matching document found\n")


if __name__ == "__main__":
    main()
