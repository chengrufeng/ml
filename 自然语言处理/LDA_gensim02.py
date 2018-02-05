import gensim as gensim
import numpy as np
from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
import gensim

'''
gensim的lda模型实践02
'''

tokenizer = RegexpTokenizer(r'\w+')

with open("stopwords.txt", "r", encoding='UTF-8') as f:  # 读文本
# create English stop words list
    en_stop=f.readlines()
    en_stop=list(map(lambda x:x.replace("\n", ""),en_stop))

    # Create p_stemmer 词干分析器 of class PorterStemmer词干提取就是去除这些词的词缀而得到词根
    p_stemmer = PorterStemmer()

    # create sample documents
    documents = ["Shipment of gold damaged in a fire",
              "Delivery of silver arrived in a silver truck",
              "Shipment of gold arrived in a truck"]

    # list for tokenized 标记化的 documents in loop
    texts = []

    # loop through document list
    for i in documents:
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

    # convert tokenized documents into a document-term matrix 文献语料库
    corpus = [dictionary.doc2bow(text) for text in texts]

    #基于这些“训练文档”计算一个TF-IDF“模型”：
    tfidf = models.TfidfModel(corpus)
    corpus_tfidf = tfidf[corpus]
    for doc in corpus_tfidf:
        print(doc)

#有了tf-idf值表示的文档向量，我们就可以训练一个LSI模型，和Latent Semantic Indexing (LSI) A Fast Track Tutorial中的例子相似，我们设置topic数为2：
    lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=2)
    print(lsi.print_topics(num_topics=2, num_words=3))

    corpus_lsi = lsi[corpus_tfidf]
    for doc in corpus_lsi:
        print(doc)#显示属于不同主题topic的概率

#好了，我们回到LSI模型，有了LSI模型，我们如何来计算文档直接的相似度，或者换个角度，给定一个查询Query，如何找到最相关的文档？当然首先是建索引了：
    index = similarities.MatrixSimilarity(lsi[corpus])

#还是以这篇英文tutorial中的查询Query为例：gold silver truck。首先将其向量化：
    query = "gold silver truck"
    query_bow = dictionary.doc2bow(query.lower().split())
    print(query_bow)
#再用之前训练好的LSI模型将其映射到二维的topic空间：
    query_lsi = lsi[query_bow]
    print(query_lsi)
    #最后就是计算其和index中doc的余弦相似度了：
    sims = index[query_lsi]
    sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])
    print(sort_sims)

    # # generate LDA model
    # ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=2, id2word=dictionary, passes=20)
    # print(ldamodel.print_topics(num_topics=2, num_words=3))

    print(123)


