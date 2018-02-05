import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from math import *


'''
二维正态分布
二元联合正态分布，均值为0，方差为1
两个变量之间存在相关系数,rho
'''


def genexp(lamb):
    return (-1.0 / lamb) * log(np.random.random())


def gennor(mu, sigma):#均值为mu，方差为sigma的正太分布
    theta = np.random.random() * 2 * pi
    rsq = genexp(0.5)
    z = sqrt(rsq) * cos(theta) #这种计算方法是由均匀分布产生正太分布。结果是标准正太分布，因此，要根据均值和方差来调整
    return mu + z * sigma


n = 1000  # 样本个数
rho = 0.5  #相关系数
x = 0 #变量x
y = 0 #变量y
sig = sqrt(1 - rho * rho)
a_x=[]
a_y=[]
for i in range(n):
    x = gennor(rho * y, sig) #根据条件概率x|y产生对应的x
    a_x.append(x)
    y = gennor(rho * x, sig) #根据条件概率y|x产生对应的y
    a_y.append(y)
    print(x, y)

a_x_a=[]
a_y_a=[]
min_t=abs(min(a_x))+1
for v in a_x:
    a_x_a.append(v+min_t)
min_t=abs(min(a_y))+1
for v in a_y:
    a_y_a.append(v+min_t)

ax = sns.distplot(a_x_a)
# plt.show()
ay = sns.distplot(a_y_a)
plt.show()

jaxy=sns.jointplot(np.array(a_x_a),np.array(a_y_a))
# jaxy=plt.scatter(a_x_a,a_y_a)
plt.show()