#! /usr/bin/env python
# ! -*- coding=utf-8 -*-

# 模拟两个正态分布的均值估计
# 只有均值不同，方差相同都为1

from numpy import *
import numpy as np
import random
import copy

SIGMA = 1
EPS = 0.0000001


# 生成方差相同,均值不同的样本
def generate_data():
    Miu1 = 10
    Miu2 = 40
    N = 1000 #产生1000个数
    X = mat(zeros((N, 1)))

    for i in range(N):
        temp = random.uniform(0, 1) #产生服从正太分布的随机数
        if (temp > 0.5):
            X[i] = temp * SIGMA + Miu1 #均值变为Miu1，方差变为SIGMA
        else:
            X[i] = temp * SIGMA + Miu2
    return X #混合了两种分布的数据，均值不同，方差相同


# EM算法
def my_EM(X):
    k = 2 #类别数量，2个正太分布
    N = len(X) #数组大小
    m_x = mean(X)
    Miu = np.random.rand(k, 1)*m_x #随机猜测两个数，作为均值
    Posterior = mat(zeros((N, 2))) #p(y|x),属于对应分布的概率
    dominator = 0 #分母
    numerator = 0 #分子
    iters=1000 #循环迭代次数
    # 先求后验概率
    for iter in range(iters):
        # E步
        for i in range(N):
            dominator = 0 #计算累计p(x)
            for j in range(k):
                dominator = dominator + np.exp(-1.0 / (2.0 * SIGMA ** 2) * (X[i] - Miu[j]) ** 2)
                # 分母和分子都没有必要计算前面乘的部分，因为SIGMA相同，分子分母会抵消。
                # print dominator,-1/(2*SIGMA**2) * (X[i] - Miu[j])**2,2*SIGMA**2,(X[i] - Miu[j])**2
                # return
            for j in range(k):
                numerator = np.exp(-1.0 / (2.0 * SIGMA ** 2) * (X[i] - Miu[j]) ** 2)#计算p(x,y)
                Posterior[i, j] = numerator / dominator #计算得到p(y|x)
        oldMiu = copy.deepcopy(Miu)
        # 最大化，M步
        for j in range(k):
            numerator = 0
            dominator = 0
            for i in range(N):
                numerator = numerator + Posterior[i, j] * X[i] #某一个数属于当前类别的累加和
                dominator = dominator + Posterior[i, j] #当前类别发生的概率累加和
            Miu[j] = numerator / dominator #数学期望，计算得到均值
        print(abs(Miu - oldMiu).sum())#计算均值的变化
        # print '\n'
        if (abs(Miu - oldMiu)).sum() < EPS:#均值变化很小，退出循环
            print(Miu, iter)
            break


if __name__ == '__main__':
    X = generate_data()
    my_EM(X)
