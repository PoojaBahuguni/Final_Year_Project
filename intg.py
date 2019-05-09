import numpy as np
import pandas as pdpip3
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
import preprocess
import ssl
from rake_nltk import Rake
from nltk import pos_tag
import nltk
import spacy
nlp = spacy.load('en_core_web_sm')

def summarize(article):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    #nltk.download('averaged_perceptron_tagger')
    sentences = preprocess.tokenize_sentences(article)
    clean_sentences = pdpip3.Series(sentences).str.replace("[^a-zA-Z]", " ")
    clean_sentences = [s.lower() for s in clean_sentences]
    clean_sentences = [preprocess.remove_stopwords(r.split()) for r in clean_sentences]

    word_embeddings = {}
    f = open('/Users/apple/Downloads/glove.6B/glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()

    sentence_vectors = []
    for i in clean_sentences:
        if len(i) != 0:
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)

    sim_mat = np.zeros([len(sentences), len(sentences)])
    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)

    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)

    # r = Rake()

    ques = []

    for i in range(len(ranked_sentences)):
        tokens = []
        print(ranked_sentences[i][1])
        article = ranked_sentences[i][1]
        print("Article:" , article)
        # r.extract_keywords_from_text(ranked_sentences[i][1])
        # print("*********************")
        # print(r.get_ranked_phrases()) # To get keyword phrases ranked highest to lowest.
        # tokens.extend(r.get_ranked_phrases())
        # lis = []
        # for i in range(len(tokens)):
        #     if len(tokens[i].split()) > 1:
        #         lis.extend(nltk.word_tokenize(tokens[i]))
        #
        #     else:
        #         lis.append(tokens[i])
        # print("Parts of speech tagging: ", pos_tag(lis))
        # for i in range(len(ranked_sentences)):
        doc = nlp(article)
        print("DOC",doc.ents)
        print([(X.text, X.label_) for X in doc.ents])
        for X in doc.ents:
            if X.label_:
                print("Inside for")
                article = article.replace(X.text, "__________")
                ques.append(article)
                break
                #print(ques)
                #print(type(ques))
        print(i+1,":" ,article)

    print(ques)
    return ques


if __name__ == '__main__':
    #article = "Maria Sharapova has basically no friends as tennis players on the WTA Tour. The Russian player has no problems in openly speaking about it and in a recent interview she said: 'I don't really hide any feelings too much. I think everyone knows this is my job here. When I'm on the courts or when I'm on the court playing, I'm a competitor and I want to beat every single person whether they're in the locker room or across the net.So I'm not the one to strike up a conversation about the weather and know that in the next few minutes I have to go and try to win a tennis match. I'm a pretty competitive girl. I say my hellos, but I'm not sending any players flowers as well. Uhm, I'm not really friendly or close to many players. I have not a lot of friends away from the courts.' When she said she is not really close to a lot of players, is that something strategic that she is doing? Is it different on the men's tour than the women's tour? 'No, not at all. I think just because you're in the same sport doesn't mean that you have to be friends with everyone just because you're categorized, you're a tennis player, so you're going to get along with tennis players. I think every person has different interests. I have friends that have completely different jobs and interests, and I've met them in very different parts of my life. I think everyone just thinks because we're tennis players we should be the greatest of friends. But ultimately tennis is just a very small part of what we do. There are so many other things that we're interested in, that we do.'"
    article = "Google was founded in 1998 by Larry Page and Sergey Brin while they were Ph.D. students at Stanford University in California. Together they own about 14 percent of its shares and control 56 percent of the stockholder voting power through supervoting stock. They incorporated Google as a privately held company on September 4, 1998. An initial public offering (IPO) took place on August 19, 2004, and Google moved to its headquarters in Mountain View, California, nicknamed the Googleplex. In August 2015, Google announced plans to reorganize its various interests as a conglomerate called Alphabet Inc. Google is Alphabet's leading subsidiary and will continue to be the umbrella company for Alphabet's Internet interests. Sundar Pichai was appointed CEO of Google, replacing Larry Page who became the CEO of Alphabet. "
    summarize(article)


