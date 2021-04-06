Github: https://github.com/sebag90/bag_of_words

# Bag of Words:
A search engine based on a bag  of words model written in python.
The model takes as input every document in the input folder and a query from the user and creates a vector space.  
For each query it returns the most similar document based on vector similarity (dot product).

## Dependencies:
- Natural Language Toolkit: https://www.nltk.org/

## Requirements:
* all requirements are saved in envoroment.yml

## Quickstart:
* install [Miniconda](https://docs.conda.io/en/latest/miniconda.html)
* cd into the root directory of this repository
* create a new environment from the environment.yml file
```
conda env create -f environment.yml
```

* activate the new environment
```
conda activate bow
```


## Supported languages:
- German
- English 
- Italian 

### Stopwords:
Source: https://github.com/stopwords-iso/stopwords-iso

## Use:
place your text files in the input folder and start main.py:  
```
search.py language
es. python3 search.py de
```
