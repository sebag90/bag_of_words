Github: https://github.com/sebag90/bag_of_words

# Bag of Words:
A bag of words model written in python.  
The model takes as input every document in the input folder and a query from the user and creates a vector space.  
For each query it returns the most similar document based on vector similarity (dot product).

## Dependencies:
- Natural Language Toolkit: https://www.nltk.org/
- Progress: https://pypi.org/project/progress/

## Supported languages:
- German
- English 
- Italian 

### Note:
2 Tokenizers are available for the German language:  
- Snowball (from nltk)
- CISTEM https://github.com/LeonieWeissweiler/CISTEM

### Stopwords:
Source: https://github.com/stopwords-iso/stopwords-iso
