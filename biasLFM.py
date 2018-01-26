import random
import math


class BiasLFM(object):
    '''
    “偏置”这东西反应的是事物固有的、不受外界影响的属性，用公式11去预估用户对物品的评分时没有考虑这个用户是宽容的还是苛刻的，
    他倾向于给物品打高分还是打低分，所以在公式1111加入了偏置bubu。
    带偏置的LFM又被称为SVD法，
    其他和LFM比基本没有变化
    '''
    def __init__(self, rating_data, F, alpha=0.1, lmbd=0.1, max_iter=500):
        '''rating_data是list<(user,list<(position,rate)>)>类型
        '''
        self.F = F
        self.P = dict()
        self.Q = dict()  # 相当于博客中Q的转置
        self.bu = dict() #bu是用户偏置，代表一个用户评分的平均值
        self.bi = dict() #bi是物品偏置，代表一个物品被评分的平均值
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.rating_data = rating_data
        self.mu = 0.0 #μ表示训练集中的所有评分的平均值

        '''随机初始化矩阵P和Q'''
        cnt = 0
        for user, rates in self.rating_data:
            self.P[user] = [random.random() / math.sqrt(self.F)
                            for x in range(self.F)]
            self.bu[user] = 0
            cnt += len(rates)
            for item, rate in rates:
                self.mu += rate
                if item not in self.Q:
                    self.Q[item] = [random.random() / math.sqrt(self.F)
                                    for x in range(self.F)]
                self.bi[item] = 0
        self.mu /= cnt

    def train(self):
        '''随机梯度下降法训练参数P和Q
        '''
        for step in range(self.max_iter):
            for user, rates in self.rating_data:
                for item, rui in rates:
                    hat_rui = self.predict(user, item)
                    err_ui = rui - hat_rui
                    self.bu[user] += self.alpha * (err_ui - self.lmbd * self.bu[user])
                    self.bi[item] += self.alpha * (err_ui - self.lmbd * self.bi[item])
                    for f in range(self.F):
                        self.P[user][f] += self.alpha * (err_ui * self.Q[item][f] - self.lmbd * self.P[user][f])
                        self.Q[item][f] += self.alpha * (err_ui * self.P[user][f] - self.lmbd * self.Q[item][f])
            self.alpha *= 0.9  # 每次迭代步长要逐步缩小

    def predict(self, user, item):
        '''预测用户user对物品item的评分
        '''
        return sum(self.P[user][f] * self.Q[item][f] for f in range(self.F)) + self.bu[user] + self.bi[item] + self.mu


if __name__ == '__main__':
    '''用户有A B C，物品有a b c d'''
    rating_data = list()
    rate_A = [('a', 1.0), ('b', 1.0)]
    rating_data.append(('A', rate_A))
    rate_B = [('b', 1.0), ('c', 1.0)]
    rating_data.append(('B', rate_B))
    rate_C = [('c', 1.0), ('d', 1.0)]
    rating_data.append(('C', rate_C))

    lfm = BiasLFM(rating_data, 2)
    lfm.train()
    for user,_ in rating_data:
        for item in ['a', 'b', 'c', 'd']:
            print(user,item, lfm.predict(user, item))  # 计算用户A对各个物品的喜好程度
        print()
