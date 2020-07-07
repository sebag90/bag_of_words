#!/usr/bin/python3

import cistem as cistem
import os
import math   
import json       
from nltk.stem.snowball import SnowballStemmer

# retrieve articles
def retrieve_articles():
    strings = {}
    for filename in os.listdir('./input'):
        path = "./input/" + filename
        with open(path, "r") as file:
            my_string = file.read()
            strings[filename] = my_string
    return strings


# manual string manipulation
def clean_string(_input_str):
    cln_str = _input_str.replace("[", " ")
    cln_str = cln_str.replace("]", " ")
    cln_str = cln_str.replace("(", " ")
    cln_str = cln_str.replace(")", " ")
    cln_str = cln_str.replace(", ", " ")
    cln_str = cln_str.replace(". ", " ")
    cln_str = cln_str.replace("; ", " ")
    cln_str = cln_str.replace(": ", " ")
    cln_str = cln_str.replace("?", " ")
    cln_str = cln_str.replace("!", " ")
    cln_str = cln_str.replace("""'""", " ")
    cln_str = cln_str.replace('"', " ")
    cln_str = cln_str.replace("""„""", " ")
    cln_str = cln_str.replace("""“""", " ")
    return cln_str


# collect stop words after removing whitespaces and /n
def collect_stopwords(language):
    nonowords = set()
    with open("stopwords-iso.json") as stopwords:
        data = json.load(stopwords)
    for word in data[language]:
        newword = word.replace(" ", "")
        newword2 = newword.replace("\n", "")
        nonowords.add(newword2.lower())

    return nonowords


# remove stopwords and stem a string
def str_2_vec(_input_string, _nonowords, language, cis):
    # extract single words
    splits = _input_string.split()
    cleaned = []
    stemmed = []

    languages = {
        "en" : "english",
        "it" : "italian",
        "de" : "german"
    }
    
    # if word is not a stop word, save it in a new list (vector)
    for something in splits:
        if something.lower() not in _nonowords:
            cleaned.append(something)
    # stem
    if cis == True and language == "de":
        for element in cleaned:
            stemmed.append(cistem.stem(element))
    else:
        stemmer = SnowballStemmer(languages[language])
        for element in cleaned:
            stemmed.append(stemmer.stem(element))
       
    return stemmed


# create matrix term list
def create_matrix_terms(_matrix_terms, _article):
    for term in _article:
        if term not in _matrix_terms:
            _matrix_terms.add(term)
    return _matrix_terms
        

# calculate TF vector for every stemmed document
def calculate_vec(_matrix_terms, _stemmed_list):
    string_vec = []
    for single_term in _matrix_terms:
        counter = 0 
        for term in _stemmed_list:
            if single_term == term:
                counter = counter + 1
        string_vec.append(counter)
    return string_vec


# calculate the number of document in which every term is present
def calc_freq(_articles, _matrix_terms):
    counters = []
    for term in _matrix_terms:
        counter = 0
        for key in _articles:
            if term in _articles[key]:
                counter = counter + 1
        counters.append(counter)
    return counters


# calculate TF * IDF (1 + log(n/nj))
def calc_tf_idf (_articles, _freq_list):
    n = len(_articles)
    for key in _articles:
        for i in range(len(_freq_list)):
            _articles[key][i] = _articles[key][i] * (1 + math.log(n/_freq_list[i]))
    return _articles


# calculate cos similarity
def find_best_match(_articles):
    results = {}
    for key in _articles:
        if key != "query":
            # dot product, length vector query, length vector n
            pprod = 0
            len_q = 0
            len_n = 0
            for i in range(len(_articles[key])):
                pprod = pprod + _articles[key][i] * _articles["query"][i] 
                len_n = len_n + ((_articles[key][i])**2)
                len_q = len_q + ((_articles["query"][i])**2)
            cos = pprod / math.sqrt(len_n * len_q)
            results[key] = cos
    return results