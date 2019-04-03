import spacy
nlp = spacy.load('en_core_web_sm')


def summarize(article):
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    ques = []
    doc = nlp(article)
    print([(X.text, X.label_) for X in doc.ents])
    for X in doc.ents:
        if X.label_:
            article=article.replace(X.text,"__________")
            print(article)
            ques.append(str(article))
            print(ques)
            print(type(ques))
            return ques
