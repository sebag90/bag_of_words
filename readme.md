Github: https://github.com/sebag90/bag_of_words

# Bag of Words:
A search engine based on a bag  of words model written in python.
The model takes as input every document in the input folder and a query from the user and creates a vector space.  
For each query it returns the most similar document based on vector similarity (dot product).

## Dependencies:
- Natural Language Toolkit: https://www.nltk.org/

### Install:
pip3 install -r requirements.txt

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
es. python3 main.py de
```