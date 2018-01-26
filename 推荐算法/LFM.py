# coding:utf-8
import random
import math

class LFM(object):
    '''
    LFM即隐因子模型，我们可以把隐因子理解为主题模型中的主题、HMM中的隐藏变量。
    比如一个用户喜欢《推荐系统实践》这本书，背后的原因可能是该用户喜欢推荐系统、或者是喜欢数据挖掘、亦或者是喜欢作者项亮本人等等，
    假如真的是由于这3个原因导致的，那如果项亮出了另外一本数据挖掘方面的书，我们可以推测该用户也会喜欢，
    这“背后的原因”我们称之为隐因子。
    '''
    def __init__(self, rating_data, F, alpha=0.1, lmbd=0.1, max_iter=1000):
        '''rating_data是list<(user,list<(position,rate)>)>类型
        F是隐因子的个数
        alpha是随机梯度的系数
        lmbd是损失函数的系数
        max_iter是最大迭代次数
        每轮更新的时间复杂度是m*f*n
        '''
        self.F = F
        self.P = dict()  # R=PQ^T，代码中的Q相当于博客中Q的转置（m*f）,这里用dict存储，查找快速方便
        self.Q = dict() #(f,n),这里是（n,f），转置后就可以乘了。
        self.alpha = alpha
        self.lmbd = lmbd
        self.max_iter = max_iter
        self.rating_data = rating_data

        '''随机初始化矩阵P和Q'''
        for user, rates in self.rating_data:
            self.P[user] = [random.random()
                            / math.sqrt(self.F)#要除以这个数，这样可以保证对每个商品的推荐程度在（0-1）
                            for x in range(self.F)]#对p进行随机初始化（m*f）
            for item, _ in rates:
                if item not in self.Q:#对Q初始化，按照商品来搞，所以这里是（n*f）
                    self.Q[item] = [random.random()
                                    / math.sqrt(self.F)
                                    for x in range(self.F)]
        print('初始化完成')

    def train(self):
        '''随机梯度下降法训练参数P和Q
        随机梯度下降法，一次只更新一个p，然后更新一个q（会用到更新的p），并且每次只用到一个err_ui
        '''
        for step in range(self.max_iter):#循环次数
            for user, rates in self.rating_data:#m次
                for item, rui in rates:#n次
                    hat_rui = self.predict(user, item)
                    err_ui = rui - hat_rui
                    for f in range(self.F):#f次
                        self.P[user][f] += self.alpha * (err_ui * self.Q[item][f] - self.lmbd * self.P[user][f])
                        self.Q[item][f] += self.alpha * (err_ui * self.P[user][f] - self.lmbd * self.Q[item][f])
            self.alpha *= 0.95  # 每次迭代步长要逐步缩小
            # self.alpha = random.random() # 每次迭代步长随机变化（不可以，随机变化导致没有规律）

    def predict(self, user, item):
        '''预测用户user对物品item的评分
        实时计算r（u,i）
        '''
        return sum(self.P[user][f] * self.Q[item][f] for f in range(self.F))


if __name__ == '__main__':
    '''用户有A B C，物品有a b c d'''
    rating_data = list()
    rate_A = [('a', 1.0), ('b', 1.0)]
    rating_data.append(('A', rate_A))
    rate_B = [('b', 1.0), ('c', 1.0)]
    rating_data.append(('B', rate_B))
    rate_C = [('c', 1.0), ('d', 1.0)]
    rating_data.append(('C', rate_C))

    lfm = LFM(rating_data, 3)
    # lfm = LFM(rating_data, 3,alpha=1, lmbd=0.5)
    lfm.train()
    for user,_ in rating_data:
        for item in ['a', 'b', 'c', 'd']:
            print(user,item, lfm.predict(user, item))  # 计算用户A对各个物品的喜好程度
        print()