#!/usr/bin/python3

import json
import string
import numpy as np
from nltk.stem.snowball import SnowballStemmer



def print_progress_bar (iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '#', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total:
        print()


class BagWords:

    def __init__(self, language):
        self.language = language
        self.vocabulary = []
        self.matrix = []
        self.texts = []

        with open("stopwords-iso.json") as stopwords:
            stopwords = json.load(stopwords)

        self.stopwords = set(stopwords[language])

    
    def clean_string(self, input_str):
        """
        remove punctuation from an input string
        """
        punctuation = [i for i in string.punctuation] + ['„', '“', '”', '–']
        
        for char in punctuation:
            input_str = input_str.replace(char, " ")

        return input_str


    def str_2_vec(self, input_string):
        """
        remove stopwords
        stem the sentence
        return a vector (list) of tokens

        eg. "Hello my name is XY" -> ["hello", "name", "XY"]
        """
        # extract single words
        splits = input_string.split()
        cleaned = []
        stemmed = []

        languages = {
            "en" : "english",
            "it" : "italian",
            "de" : "german"
        }

        # if word is not a stop word, save it in a new list (vector)
        for word in splits:
            if word.lower() not in self.stopwords:
                cleaned.append(word)
        
        # stem
        stemmer = SnowballStemmer(languages[self.language])
        for element in cleaned:
            stemmed.append(stemmer.stem(element))

        return stemmed


    # create matrix term list
    def create_vocabulary(self, sentence):
        """
        takes as input a sentence in form of a stemmed vector
        adds new tokens to the vocabulary
        """
        for term in sentence:
            if term not in self.vocabulary:
                self.vocabulary = np.append(self.vocabulary, term)


    def calculate_vec(self, sentence_vector, voc=None):
        """
        given a sentence in the form of a list of stemmed tokens
        this method calculates a term frequency vector based on 
        the vocabulary.

        Input: vector of tokens
        Output: term frequency vector
        """
        if voc == None: 
            vocabulary = np.array(self.vocabulary) 
        else:
            vocabulary = np.array(voc)

        sentence_vector = np.array(sentence_vector)

        indexes = [np.where(x == vocabulary)[0][0] for x in sentence_vector]
        result = np.zeros(vocabulary.shape)

        for i in indexes:
            result[i] += 1
        
        return result



    # PUBLIC METHODS---------------------------------------------------------------------------------



    def compute_matrix(self, process=False):
        """
        given all the sentences added, the bag of words will compute
        the term frequency matrix
        """
        self.matrix = []
        # add all sentences to vocabulary
        
        for i in range(len(self.texts)):
            if not isinstance(self.texts[i], list):
                self.texts[i] = self.str_2_vec(self.texts[i])
                self.create_vocabulary(self.texts[i])
                if process == True:
                    print_progress_bar(i+1, len(self.texts), prefix="Loading ", length=40)
         
        
        # calculate matrix
        j = 1
        for text in self.texts:
            vec = self.calculate_vec(text)
            self.matrix.append(vec)
            if process == True:
                print_progress_bar(j, len(self.texts), prefix="Indexing", length=40)
                j += 1

        self.matrix = np.array(self.matrix)


    def add_sentence(self, sentence):
        """
        adds a sentence to the texts.
        After adding all sentences it is necessary to 
        call the method compute_matrix()
        """
        cleaned = self.clean_string(sentence)
        self.texts.append(cleaned)


    def tf_idf(self):
        """
        calculate inverse term frequency
        IDF = 1 + log(N/nj)

        N = number of total vectors 
        nj = number of vectors containing the word
        """

        x = np.array(self.matrix)
        
        # number of vectors
        N = x.shape[0]
        nj = (x>0).sum(axis=0) * np.ones(x.shape)

        tfidf = x * (1 + np.log(N/nj))

        self.matrix = tfidf


    def similarity(self, new_sentence):
        """
        given a new string calculates the similarity
        between it and every other vector and returns
        the index of the text.

        For performance reasons, the matrix in not calculated again.
        Instead, the number of new tokens in the input sentence
        is calculated and for each token a new column of 0s is added
        (since the word is new there is no old sentence with this word)
        and given the new vocabulary (old + new tokens), the vector of the
        new sentence is calculated and used to compute similarity with
        the sentences in the matrix

        Input: string
        Output: index of the most similar text (int)
        """
        cleaned = self.clean_string(new_sentence)
        tokens = self.str_2_vec(cleaned)
        
        if not set(tokens).intersection(set(self.vocabulary)):
            return None

        else:
            difference = set(tokens) - set(self.vocabulary)
            to_append = np.zeros((self.matrix.shape[0], len(difference)))
            matrix = np.append(self.matrix, to_append, axis=1)
            new_voc = list(self.vocabulary) + list(difference)
            question_vector = self.calculate_vec(tokens, new_voc)
            result = np.matmul(matrix, question_vector)

            return np.argmax(result)




if __name__ == "__main__":
    bag = BagWords("en")
    
    texts = [
        "I was going to the shop and saw a new car",
        "my new car is ideal for going to the shop",
        "my computer is less expensive than my car",
        "my car is more expensive than my computer"
    ]
    for text in texts:
        bag.add_sentence(text)

    bag.compute_matrix()

    print(bag.similarity("expensive car"))

    for i in bag.matrix:
        print(i)