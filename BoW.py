import copy
import json
import string
import numpy as np
from nltk.stem.snowball import SnowballStemmer


def progress_bar(iteration, total, prefix='', suffix='', decimals=1, length=40,
                 fill='#', miss=".", end="\r", stay=True, fixed_len=False):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        miss        - Optional  : bar missing charachter (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    if fixed_len:
        bar_len = length - len(prefix) - len(suffix)
    else:
        bar_len = length

    percent = f"{100*(iteration/float(total)):.{decimals}f}"
    filled_length = int(bar_len * iteration // total)
    bar = f"{fill * filled_length}{miss * (bar_len - filled_length)}"
    to_print = f"\r{prefix} [{bar}] {percent}% {suffix}"
    print(to_print, end=end)

    # Print New Line on Complete
    if iteration >= total:
        if stay:
            print()
        else:
            # clean line given lenght of lase print
            print(" "*len(to_print), end=end)


class BagWords:

    def __init__(self, language, stopwords):
        self.language = language
        self.vocabulary = dict()
        self.matrix = list()
        self.texts = list()
        self.stopwords = set(stopwords)

    def clean_string(self, input_str):
        """
        remove punctuation from an input string
        """
        punctuation = [i for i in string.punctuation] + ['„', '“', '”', '–']

        for char in punctuation:
            input_str = input_str.replace(char, " ")

        return input_str

    def stem(self, input_string, train=True):
        """
        remove stopwords
        stem the sentence
        return a vector (list) of tokens

        eg. "Hello my name is XY" -> ["hello", "name", "XY"]
        """
        # extract single words
        splits = input_string.split()
        stemmed = []

        languages = {
            "en": "english",
            "it": "italian",
            "de": "german"
        }

        stemmer = SnowballStemmer(languages[self.language])

        # if word is not a stop word, save it in a new list (vector)
        for word in splits:
            if word.lower() not in self.stopwords:
                stem = stemmer.stem(word)
                stemmed.append(stem)

                # add word to vocabulary
                if train is True:
                    if stem not in self.vocabulary:
                        self.vocabulary[stem] = len(self.vocabulary)

        return stemmed

    def stem2vec(self, sentence_vector, voc=None):
        """
        given a sentence in the form of a list of stemmed tokens
        this method calculates a term frequency vector based on
        the vocabulary.

        Input: vector of tokens
        Output: term frequency vector
        """
        if voc is None:
            voc = self.vocabulary

        vec = np.zeros(len(voc))

        for word in sentence_vector:
            index = voc[word]
            vec[index] += 1

        return vec

    # PUBLIC METHODS--------------------------------------------------------

    def compute_matrix(self, process=False):
        """
        given all the sentences added, the bag of words will compute
        the term frequency matrix
        """
        # calculate matrix
        for i, text in enumerate(self.texts):
            vec = self.stem2vec(text)
            self.matrix.append(vec)
            if process:
                progress_bar(
                    i+1, len(self.texts), prefix="Indexing", length=40
                )

        self.matrix = np.array(self.matrix)

    def add_sentence(self, sentence):
        """
        adds a sentence to the texts.
        After adding all sentences it is necessary to
        call the method compute_matrix()
        """
        cleaned = self.clean_string(sentence)
        stemmed = self.stem(cleaned)
        self.texts.append(stemmed)

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
        nj = (x > 0).sum(axis=0) * np.ones(x.shape)

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
        stemmed = self.stem(cleaned, train=False)

        if not set(stemmed).intersection(set(self.vocabulary.keys())):
            return None

        else:
            difference = set(stemmed) - set(self.vocabulary.keys())
            to_append = np.zeros((self.matrix.shape[0], len(difference)))
            matrix = np.append(self.matrix, to_append, axis=1)

            new_voc = copy.deepcopy(self.vocabulary)
            for word in difference:
                if word not in new_voc:
                    new_voc[word] = len(new_voc)

            question_vector = self.stem2vec(stemmed, new_voc)
            result = np.matmul(matrix, question_vector)
            return np.argmax(result)


if __name__ == "__main__":
    # collect stopwords
    with open("stopwords-iso.json") as stopwords:
        stopwords = json.load(stopwords)

    stopwords = stopwords["en"]

    texts = [
        "I was going to the shop and saw a new car",
        "my new car is ideal for going to the shop",
        "my computer is less expensive than my car",
        "my car is more expensive than my house"
    ]

    bag = BagWords("en", stopwords)

    for text in texts:
        bag.add_sentence(text)

    bag.compute_matrix()
    print("Bag of Words:")
    print('\t'.join(bag.vocabulary))
    for i, row in enumerate(bag.matrix):
        row = '\t'.join([str(int(i)) for i in row])
        print(f"{row}\t{texts[i]}")

    query = "I drive to the shop with my car"
    print(f"Query:\t{query}")
    result = bag.similarity(query)  # returns index
    print(f"Most similar text:\t{texts[result]}")
