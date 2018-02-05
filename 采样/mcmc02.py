#coding=utf-8
'''
基本MCMC算法以及M-H算法的python实现
@author: whz
p:输入的概率分布，离散情况采用元素为概率值的数组表示
N:认为迭代N次马尔可夫链收敛
Nlmax:马尔可夫链收敛后又取的服从p分布的样本数
isMH:是否采用MH算法，默认为True
'''
from __future__ import division
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
from array import array

def mcmc(p,N=100,Nlmax=100,isMH=True):
    A = np.array([[1.0 / len(p) for x in range(len(p))] for y in range(len(p))], dtype=np.float64)
    #取q为[1/3,1/3,1/3]
    #A为[q,q,q]
    X0 = np.random.randint(len(p))
    count = 0
    samplecount = 0
    L = array("i",[X0])
    l = array("d")
    while True:
        X = L[samplecount]
        cur = np.argmax(np.random.multinomial(1,A[X]))
        count += 1
        if isMH:
            a = (p[cur]*A[cur][X])/(p[X]*A[X][cur])
            alpha = min(a,1)
        else:
            alpha = p[cur]*A[cur][X]
        u = np.random.uniform(0,1)
        if u<alpha:
            samplecount += 1
            L.append(cur)
            if count>N:
                l.append(cur)
        if len(l)>=Nlmax:
            break
        else:
            continue
    La = np.frombuffer(L)
    la = np.frombuffer(l)
    return La,la
def count(q,n):
    L = array("d")
    l1 = array("d")
    l2 = array("d")
    for e in q:
        L.append(e)
    for e in range(n):
        l1.append(L.count(e))
    for e in l1:
        l2.append(e/sum(l1))
    return l1,l2


p = np.array([0.9,0.05,0.05])#初试采样数据
a = mcmc(p,Nlmax=100)[1]
l1 = ['state%d'% x for x in range(len(p))]
plt.pie(count(a,len(p))[0],labels=l1,labeldistance=0.3,autopct='%1.2f%%')
plt.title("sampling")
plt.show()