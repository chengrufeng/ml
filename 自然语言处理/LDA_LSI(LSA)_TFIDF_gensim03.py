import gensim as gensim
import numpy as np
from nltk.tokenize import RegexpTokenizer
# from stop_words import get_stop_words
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models, similarities
import gensim
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.lancaster import LancasterStemmer
from gensim import corpora, models, similarities

'''
gensim的lda模型实践03

'''

tokenizer = RegexpTokenizer(r'\w+')
# en_stop=[]
# with open("stopwords.txt", "r", encoding='UTF-8') as f:  # 读文本
# # create English stop words list
#     en_stop=f.readlines()
courses =[]
with open("coursera_corpus.txt", "r", encoding='UTF-8') as f:  # 读文本
    # create English stop words list
    courses =f.readlines()
courses_name = [course.split('\t')[0] for course in courses]
# print(courses_name[0:10])
# print(brown.readme())
# print(brown.words()[0:10])
# print(brown.tagged_words()[0:10])
# print(len(brown.words()))

texts_lower = [[word for word in document.lower().split()] for document in courses]

#注意其中很多标点符号和单词是没有分离的，所以我们引入nltk的word_tokenize函数，并处理相应的数据：
texts_tokenized = [[word.lower() for word in word_tokenize(document)] for document in courses]
english_stopwords = stopwords.words('english')
texts_filtered_stopwords = [[word for word in document if not word in english_stopwords] for document in
                            texts_tokenized]
#停用词被过滤了，不过发现标点符号还在，这个好办，我们首先定义一个标点符号list:
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%', '...']
texts_filtered = [[word for word in document if not word in english_punctuations] for document in texts_filtered_stopwords]

#更进一步，我们对这些英文单词词干化（Stemming)
st = LancasterStemmer()
texts_stemmed = [[st.stem(word) for word in docment] for docment in texts_filtered]

#去掉在整个语料库中出现次数为1的低频词
all_stems = sum(texts_stemmed, [])
stems_once = set(stem for stem in set(all_stems) if all_stems.count(stem) == 1)
texts = [[stem for stem in text if stem not in stems_once] for text in texts_stemmed]

dictionary = corpora.Dictionary(texts)
corpus = [dictionary.doc2bow(text) for text in texts]
tfidf = models.TfidfModel(corpus)
corpus_tfidf = tfidf[corpus]

lsi = models.LsiModel(corpus_tfidf, id2word=dictionary, num_topics=10)#这里我们拍脑门决定训练topic数量为10的LSI模型：
index = similarities.MatrixSimilarity(lsi[corpus])
ml_course = texts[210]
ml_bow = dictionary.doc2bow(ml_course)
ml_lsi = lsi[ml_bow]
sims = index[ml_lsi]
sort_sims = sorted(enumerate(sims), key=lambda item: -item[1])

# print(sort_sims[0:10])
#取按相似度排序的前10门课程：
for i,v in sort_sims[0:10]:
    print(courses_name[i],v)