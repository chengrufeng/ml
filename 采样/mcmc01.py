import numpy as np
import scipy.special as ss
import matplotlib.pyplot as plt
'''
mcmc（Metropolis-Hastings采样算法）采样算法示例，
算法是对beta分布进行采样
beta分布有两个参数，
'''
def beta_s(x, a, b):
    return x**(a-1)*(1-x)**(b-1)
def beta(x, a, b):
    return beta_s(x, a, b)/ss.beta(a, b)

def plot_mcmc(a, b):
    cur = np.random.rand()
    states = [cur]
    for i in range(10**5):
        next, u = np.random.rand(),np.random.rand()
        if u < np.min((beta_s(next, a, b)/beta_s(cur, a, b), 1)):
            states.append(next)
            cur = next
    x = np.arange(0, 1, .01)
    plt.figure(figsize=(10, 5))
    plt.plot(x, beta(x, a, b), lw=2, label='real dist: a={}, b={}'.format(a, b))
    plt.hist(states[-1000:], 25, normed=True, label='simu mcmc: a={}, b={}'.format(a, b))
    plt.show()

# if __name__ == '__main__':
plot_mcmc(0.1, 0.1)
plot_mcmc(1, 1)
plot_mcmc(2, 3)