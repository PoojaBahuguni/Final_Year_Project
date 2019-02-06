import pandas as pd
from nltk import word_tokenize, sent_tokenize, pos_tag
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer


def tokenize_words():
    data = 'Having individual dictionary entries per version would be highly redundant and inefficient, especially since, once we convert to numbers, the "value" is going to be identical.'
    return word_tokenize(data.lower())

def tokenize_sentences():
    data = "Hello Mr. Smith, how are you doing today? The weather is great, and Python is awesome. The sky is pinkish-blue. You shouldn't eat cardboard."
    return sent_tokenize(data.lower())

def remove_stopwords(word_tokens):
    stop_words = set(stopwords.words('english'))
    filtered_sentence = [w for w in word_tokens if not w in stop_words]
    return filtered_sentence

def stemmer(word_tokens):
    stemmed_words = []
    ps = PorterStemmer()
    for w in word_tokens:
        stemmed_words.append(ps.stem(w))
    return stemmed_words

def pos_tagging(tokens):
    pos_tagged = []
    try:
        tagged = pos_tag(tokens)

    except Exception as e:
        print(str(e))
    
    return tagged


if __name__ == '__main__':
    tokens = tokenize_words()
    print(tokenize_sentences())
    print(tokens)
    print(remove_stopwords(tokens))
    print(stemmer(tokens))
    print(pos_tagging(tokens))
