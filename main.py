#!/usr/bin/python3

from datetime import datetime
import functions as fn
import operator
import copy
import argparse
import sys
from progress.bar import Bar
            
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("language", help="select a language (en - de - it)", type=str, metavar = "lang")
    parser.add_argument("-c", help="change the default tokenizer for German from Snowball to Cistem", action = "store_true")
    args = parser.parse_args()
    
    if args.language not in ["de", "it", "en"]:
        print("Language not supported yet\nPlease select a language between English (en), German (de) or Italian (it)")
        return
    if args.c == True and args.language != "de":
        print("The Cistem tokenizer is only available for the German language.\nPlease select another language or remove the -c option")
        return 
    

    # TODO:
    # check if input/output folders are there
    # if not create them 

    # retrieve all articles from folder bestand and stopwords
    articles_org = fn.retrieve_articles()
    stopwords = fn.collect_stopwords(args.language)

    # clean the articles
    for element in articles_org:
        new_str = fn.clean_string(articles_org[element])
        articles_org[element] = new_str

    # stem the articles
    for key in articles_org:
        articles_org[key] = fn.str_2_vec(articles_org[key], stopwords, args.language, args.c)
    
    while True:
        user_input =  input("Enter data query or '...' to exit:\n> ")
        if user_input == "...":
            break
        # create a local copy of the cleaned and stemmed dict
        articles = copy.deepcopy(articles_org)

        # clean and stem user input
        cleaned_input = fn.clean_string(user_input)
        stemmed_input = fn.str_2_vec(cleaned_input, stopwords, args.language, args.c)
       
        if len(stemmed_input) != 0:

            # create document matrix terms
            matrix_terms = set()
            for key in articles:
                matrix_terms = fn.create_matrix_terms(matrix_terms, articles[key])

            # warning if stemmed query not in stemmed matrix terms
            input_present = False
            for term in stemmed_input:
                if term in matrix_terms:
                    input_present = True
            if input_present == False:
                print("no match found\n")
            else:
                articles["query"] = stemmed_input

                # add stemmed query to matrix terms
                matrix_terms = fn.create_matrix_terms(matrix_terms, stemmed_input)

                # calculate frequency list for every term 
                freq_list = fn.calc_freq(articles, matrix_terms)
          
                bar = Bar('Processing', fill="=", max = len(articles))
                for key in articles:
                    vec = fn.calculate_vec(matrix_terms, articles[key])
                    articles[key] = vec
                    bar.next()
                bar.finish()

                # calculate TF * IFD (1 - log(n/nj))
                articles = fn.calc_tf_idf(articles, freq_list)   

                # calculate vector cos similarity and print out the best result
                results = fn.find_best_match(articles)
                print("Best result: ", max(results.items(), key=operator.itemgetter(1))[0], "\n")
    
        else:
            print("Sorry, try with another query\n")
    

if __name__ == "__main__":
    main()
