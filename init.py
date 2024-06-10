#%%
import numpy as np
from scipy.stats import truncnorm
rng = np.random.default_rng()
generator = np.random.default_rng()

def init_params(data):
    K=1
    J, N = data.shape
    A0 = np.zeros([1, J])
    B0 = np.zeros([1, J])
    
    for j in range(J):
        #A0[0][j] = np.random.uniform(0.4, 4)
        #B0[0][j] = np.random.uniform(-2, 2)
        A0[0][j] = rng.lognormal(-0.5, 0.2) # 対数正規分布か

    a, b = -2, 2
    mean = 0
    std = np.sqrt(1.0) # 標準偏差
    # 標準化する。 (scipy.stats.truncnorm のドキュメントの Notes 参照)
    a, b = (a - mean) / std, (b - mean) / std
    B0[0] = truncnorm.rvs(a, b, loc=mean, scale=std, size=J)

    sig_theta0 = 1
    rnd2 = generator.normal(size=N*K)
    theta = rnd2
    eps0 = np.ones([N,1]) * B0 - np.array(theta * np.array(A0).T).T

    return A0, B0, eps0
# %%
