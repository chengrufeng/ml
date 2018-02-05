import gensim as gensim
import numpy as np
from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

'''
gensim的lda模型实践
'''

tokenizer = RegexpTokenizer(r'\w+')

with open("stopwords.txt", "r", encoding='UTF-8') as f:  # 读文本
# create English stop words list
    en_stop=f.readlines()
    en_stop=list(map(lambda x:x.replace("\n", ""),en_stop))

    # Create p_stemmer 词干分析器 of class PorterStemmer词干提取就是去除这些词的词缀而得到词根
    p_stemmer = PorterStemmer()

    # create sample documents
    doc_a = "Brocolli is good to eat. My brother likes to eat good brocolli, but not my mother."
    doc_b = "My mother spends a lot of time driving my brother around to baseball practice."
    doc_c = "Some health experts suggest that driving may cause increased tension and blood pressure."
    doc_d = "I often feel pressure to perform well at school, but my mother never seems to drive my brother to do better."
    doc_e = "Health professionals say that brocolli is good for your health."

    # compile sample documents into a list
    doc_set = [doc_a, doc_b, doc_c, doc_d, doc_e]

    # list for tokenized 标记化的 documents in loop
    texts = []

    # loop through document list
    for i in doc_set:
        # clean and tokenize document string
        raw = i.lower()
        tokens = tokenizer.tokenize(raw)

        # remove stop words from tokens
        stopped_tokens = [i for i in tokens if not i in en_stop]

        # stem tokens
        stemmed_tokens = [p_stemmer.stem(i) for i in stopped_tokens]

        # add tokens to list
        texts.append(stemmed_tokens)

    # turn our tokenized documents into a id <-> term dictionary
    dictionary = corpora.Dictionary(texts)

    # convert tokenized documents into a document-term matrix
    corpus = [dictionary.doc2bow(text) for text in texts]

    # generate LDA model
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=20)

    print(ldamodel.print_topics(num_topics=2, num_words=3))

    print(123)


