import scipy
import numpy as np

def binomcdf(p,n):
    p = 0.5
    n = 100
    x = 0
    for a in range(10):
        print(scipy.stats.binom.cdf(x, n, p))
        x += 10

delta = 0.1
e = 2.718

target_prob = delta / e

N= 1750
Score_list = [0.0576	,0.1236,	0.2412,	0.254,	0.2643,	0.2679,	0.2596,	0.272,	0.2721,	0.2729,	0.2702,	0.2732,	0.272]
R_hat_list = np.array([1.]*len(Score_list)) - np.array(Score_list)
alpha_list = []
for R_hat in R_hat_list:
    t = np.ceil(N*R_hat)
    res = 0.
    res_gap = 1e10
    for ind in range(1000):
        alpha = ind * 0.001
        cdf = scipy.stats.binom.cdf(t, N, alpha)
        if abs(cdf-target_prob)<res_gap:
            res_gap = abs(cdf-target_prob)
            res = alpha
    alpha_list.append(res)

print(f'Empirical risk: {R_hat_list}\nControlled risk alpha: {alpha_list}')