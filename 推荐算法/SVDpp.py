import random
import math


class SVDPP(object):
    '''
    SVD++认为任何用户只要对物品ii有过评分，不论评分是多少，就已经在一定程度上反应了他对各个隐因子的喜好程度
    yi=(yi1,yi2,...,yiF,)yi=(yi1,yi2,...,yiF,)，y是物品所携带的属性
    '''
    def __init__(self, rating_data, F, alpha=0.1, lmbd=0.1, max_iter=500):
        '''rating_data是list<(user,list<(position,rate)>)>类型
        '''
        self.F = F
        self.P = dict()
        self.Q = dict()  # 相当于博客中Q的转置
        self.Y = dict() #y是物品所携带的属性
        self.bu = dict()
        self.bi = dict()
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.rating_data = rating_data
        self.mu = 0.0

        '''随机初始化矩阵P、Q、Y'''
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
                if item not in self.Y:
                    self.Y[item] = [random.random() / math.sqrt(self.F)
                                    for x in range(self.F)]
                self.bi[item] = 0
        self.mu /= cnt

    def train(self):
        '''随机梯度下降法训练参数P和Q
        '''
        for step in range(self.max_iter):
            for user, rates in self.rating_data:
                z = [0.0 for f in range(self.F)]
                for item, _ in rates:
                    for f in range(self.F):
                        z[f] += self.Y[item][f]
                ru = 1.0 / math.sqrt(1.0 * len(rates))
                s = [0.0 for f in range(self.F)]
                for item, rui in rates:
                    hat_rui = self.predict(user, item, rates)
                    err_ui = rui - hat_rui
                    self.bu[user] += self.alpha * (err_ui - self.lmbd * self.bu[user])
                    self.bi[item] += self.alpha * (err_ui - self.lmbd * self.bi[item])
                    for f in range(self.F):
                        s[f] += self.Q[item][f] * err_ui
                        self.P[user][f] += self.alpha * (err_ui * self.Q[item][f] - self.lmbd * self.P[user][f])
                        self.Q[item][f] += self.alpha * (
                        err_ui * (self.P[user][f] + z[f] * ru) - self.lmbd * self.Q[item][f])
                for item, _ in rates:
                    for f in range(self.F):
                        self.Y[item][f] += self.alpha * (s[f] * ru - self.lmbd * self.Y[item][f])
            self.alpha *= 0.9  # 每次迭代步长要逐步缩小

    def predict(self, user, item, ratedItems):
        '''预测用户user对物品item的评分
        '''
        z = [0.0 for f in range(self.F)]
        for ri, _ in ratedItems:
            for f in range(self.F):
                z[f] += self.Y[ri][f]
        return sum(
            (self.P[user][f] + z[f] / math.sqrt(1.0 * len(ratedItems))) * self.Q[item][f] for f in range(self.F)) + \
               self.bu[user] + self.bi[item] + self.mu


if __name__ == '__main__':
    '''用户有A B C，物品有a b c d'''
    rating_data = list()
    rate_A = [('a', 1.0), ('b', 1.0)]
    rating_data.append(('A', rate_A))
    rate_B = [('b', 1.0), ('c', 1.0)]
    rating_data.append(('B', rate_B))
    rate_C = [('c', 1.0), ('d', 1.0)]
    rating_data.append(('C', rate_C))

    lfm = SVDPP(rating_data, 2)
    lfm.train()
    for user,rate_t in rating_data:
        for item in ['a', 'b', 'c', 'd']:
            print(user,item, lfm.predict(user, item,rate_t))  # 计算用户A对各个物品的喜好程度
        print()