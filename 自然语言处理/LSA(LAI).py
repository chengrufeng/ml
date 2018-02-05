from operator import itemgetter

from numpy import zeros
from scipy.linalg import svd
from math import log
from numpy import asarray, sum
import matplotlib.pyplot as plt

titles =[
    "The Neatest Little Guide to Stock Market Investing",
    "Investing For Dummies, 4th Edition",
    "The Little Book of Common Sense Investing: The Only Way to Guarantee Your Fair Share of Stock Market Returns",
    "The Little Book of Value Investing",
    "Value Investing: From Graham to Buffett and Beyond",
    "Rich Dad's Guide to Investing: What the Rich Invest in, That the Poor and the Middle Class Do Not!",
    "Investing in Real Estate, 5th Edition",
    "Stock Investing For Dummies",
    "Rich Dad's Advisors: The ABC's of Real Estate Investing: The Secrets of Finding Hidden Profits Most Investors Miss"
]
#Stopwords 是停用词 ignorechars是无用的标点
stopwords = ['and', 'edition', 'for', 'in', 'little', 'of', 'the', 'to']
ignorechars = ''''',:'!'''

# print(ignorechars)
#这里定义了一个LSA的类，包括其初始化过程wdict是词典，dcount用来记录文档号。
class LSA(object):
    #这里定义了一个LSA的类，包括其初始化过程wdict是词典，dcount用来记录文档号。
    def __init__(self, stopwords, ignorechars):
        self.stopwords = stopwords
        self.ignorechars = ignorechars
        self.wdict = {}
        self.dcount = 0

    #这个函数就是把文档拆成词并滤除停用词和标点，剩下的词会把其出现的文档号填入到wdict中去，
    # 例如，词book出现在标题3和4中，则我们有self.wdict['book'] = [3, 4]。相当于建了一下倒排。
    def parse(self, doc):
        words = doc.split();
        for w in words:
            w = w.lower().translate(self.ignorechars)
            if w in self.stopwords:
                continue
            elif w in self.wdict:
                self.wdict[w].append(self.dcount)
            else:
                self.wdict[w] = [self.dcount]
        self.dcount += 1

    #所有的文档被解析之后，所有出现的词（也就是词典的keys）被取出并且排序。
    # 建立一个矩阵，其行数是词的个数，列数是文档个数。
    # 最后，所有的词和文档对所对应的矩阵单元的值被统计出来。
    def build(self):
        self.keys = [k for k in self.wdict.keys() if len(self.wdict[k]) > 1]
        self.keys.sort()
        self.A = zeros([len(self.keys), self.dcount])
        for i, k in enumerate(self.keys):
            for d in self.wdict[k]:
                self.A[i, d] += 1

    #In sophisticated Latent Semantic Analysis systems,
    # the raw matrix counts are usually modified so that
    # rare words are weighted more heavily than common words. F
    def TFIDF(self):
        WordsPerDoc = sum(self.A, axis=0)
        DocsPerWord = sum(asarray(self.A > 0, 'i'), axis = 1)#将数据类型转换为int
        rows, cols = self.A.shape
        for i in range(rows):
            for j in range(cols):
                self.A[i, j] = (self.A[i, j] / WordsPerDoc[j]) * log(float(cols) / DocsPerWord[i])

    def calc(self):
        self.U, self.S, self.Vt = svd(self.A)

    def printA(self):
        print(self.A)
    def pltS(self):
        print(self.S)
        s_plt=self.S
        s_plt=s_plt/sum(s_plt)
        x=range(len(self.S))
        plt.bar(x,s_plt)
        plt.show()






mylsa = LSA(stopwords, ignorechars)
for t in titles:
    mylsa.parse(t)

mylsa.build()
mylsa.printA()
mylsa.TFIDF()
mylsa.calc()
mylsa.pltS()



