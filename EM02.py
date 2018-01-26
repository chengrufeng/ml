#! /usr/bin/env python
# ! -*- coding=utf-8 -*-

# 估计两个正态分布的均值和方差
# 均值和方差都变化

from numpy import *
import numpy as np
import random
import copy

EPS = 0.0000001 #误差控制
EPS_s = 0.0000001 #误差控制

# 生成方差相同,均值不同的样本
def generate_data():
    Miu1 = 30
    SIGMA1= 2
    SIGMA2= 5
    Miu2 = 40
    N = 1000 #产生1000个数
    X = mat(zeros((N, 1)))

    for i in range(N):
        temp = random.uniform(0, 1) #从均匀分布中读取样本
        if (temp > 0.5):
            X[i]=np.random.normal(Miu1,SIGMA1)
            # X[i] = temp * SIGMA1 + Miu1 #均值变为Miu1，方差变为SIGMA

        else:
            X[i]=np.random.normal(Miu2,SIGMA2)
            # X[i] = temp * SIGMA2 + Miu2
    return X #混合了两种分布的数据，均值不同，方差相同


# EM算法
def my_EM(X):
    k = 2 #类别数量，2个正太分布
    N = len(X) #数组大小
    m_x = mean(X)
    v_x = var(X)
    Miu = np.random.rand(k, 1)*m_x #随机猜测两个数，作为均值
    Miu2 = np.random.rand(k, 1)*m_x #随机猜测两个数，作为均值
    SIGMA = np.random.rand(k, 1)*m_x #随机猜测两个数，作为均值
    Posterior = mat(zeros((N, 2))) #p(y|x),属于对应分布的概率
    dominator = 0 #分母
    numerator = 0 #分子
    iters=1000 #循环迭代次数
    # 先求后验概率
    for iter in range(iters):
        # E步
        for i in range(N):
            dominator = 0 #计算累计p(x)
            numerator=0
            for j in range(k):
                dominator = dominator + np.exp(-1.0 / (2.0 * SIGMA[j] ** 2) * (X[i] - Miu[j]) ** 2)/SIGMA[j]
                # 分母和分子都没有必要计算前面乘的部分，因为SIGMA不相同，分子分母会抵消。
                # print dominator,-1/(2*SIGMA**2) * (X[i] - Miu[j])**2,2*SIGMA**2,(X[i] - Miu[j])**2
                # return
            for j in range(k):
                numerator = np.exp(-1.0 / (2.0 * SIGMA[j] ** 2) * (X[i] - Miu[j]) ** 2)/SIGMA[j]#计算p(x,y)
                Posterior[i, j] = numerator / dominator #计算得到p(y|x)
        oldMiu = copy.deepcopy(Miu)
        oldSIGMA = copy.deepcopy(SIGMA)
        # 最大化，M步
        for j in range(k):
            numerator = 0
            dominator = 0
            for i in range(N):
                numerator = numerator + Posterior[i, j] * X[i] #某一个数属于当前类别的累加和
                dominator = dominator + Posterior[i, j] #当前类别发生的概率累加和
            Miu[j] = numerator / dominator #数学期望，计算得到均值

        for j in range(k):
            numerator = 0
            dominator = 0
            for i in range(N):
                numerator = numerator + (Posterior[i, j] * X[i]**2)#某一个数属于当前类别的累加和
                dominator = dominator + Posterior[i, j] #当前类别发生的概率累加和
                Miu2[j] = numerator / dominator #平方的期望

        SIGMA =np.sqrt( Miu2-Miu**2) #期望的平方的期望-期望的平方
        print(Miu,SIGMA)
        # print(abs(Miu - oldMiu).sum(),abs(SIGMA - oldSIGMA).sum())#计算均值的变化
        # print '\n'
        if (abs(Miu - oldMiu)).sum()+abs(SIGMA - oldSIGMA).sum()< EPS:#均值变化很小，退出循环
            print(Miu,SIGMA,iter)
            break


if __name__ == '__main__':
    X = generate_data()
    my_EM(X)
