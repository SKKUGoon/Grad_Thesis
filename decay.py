import numpy as np
import matplotlib.pyplot as plt


# Insert this in to LaTeX
def exp_decay(rate, x, init_val=1):
    return init_val * np.exp(-rate * x)


def inv_decay(rate, x, init_val=1):
    return init_val / (1 + rate * x)

t = list(range(1,100))

res1, res2, res3 = [[] for _ in range(3)]

res4, res5, res6 = [[] for _ in range(3)]

for i in t:
    res1.append(exp_decay(0.1, i))
    res2.append(exp_decay(0.2, i))
    res3.append(exp_decay(0.3, i))

    res4.append(inv_decay(0.1, i))
    res5.append(inv_decay(0.2, i))
    res6.append(inv_decay(0.3, i))

plt.plot(t, res1, label = 'exp, .1')
plt.plot(t, res2, label = 'exp, .2')
plt.plot(t, res3, label = 'exp, .3')


plt.plot(t, res4, label = 'inv, .1')
plt.plot(t, res5, label = 'inv, .2')
plt.plot(t, res6, label = 'inv, .3')
plt.legend()
plt.show()